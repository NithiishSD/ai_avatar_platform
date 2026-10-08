#!/usr/bin/env python3
"""
Load test against a RUNNING server (N-03 job initiation, N-04 capacity).

Two numbers, measured separately because they answer different questions:

* **Initiation latency (N-03)**: how long a client waits until the server has
  accepted a job (HTTP 202). Measured for a render job (queued, answered at
  once) and for a synthesis job. In the development queue (``QUEUE_BACKEND=
  in_memory``) the synthesis task runs *inside* the request, so its "initiation"
  is the whole synthesis; only a Celery deployment answers 202 immediately.
  Both are reported with the queue mode, never one as the other.
* **Capacity (N-04)**: requests per minute the server serves with many
  concurrent clients, counting what it accepted (2xx) and what the rate limiter
  refused (429) separately. Run it once against the default limiter and once
  with the limiter lifted (``RATE_LIMIT_RPM=100000``) to see the policy and the
  raw capacity.

Standard library only. Start the server first:

    cd backend && PYTHONPATH=. ./.conda/bin/python -m uvicorn app:app --port 8000
    python scripts/load_test.py --seconds 20 --clients 8

What it writes: one JSON report to
``outputs/benchmarks/load-test-<YYYYmmdd-HHMMSS>.json`` and the same JSON to
stdout. The report records the queue backend and security settings read from
``/health``, so a number is never quoted without the conditions it was taken in.

Concepts used here, explained once:

**Percentiles.** p50 is the median: half the requests were faster. p95 and p99
say how slow the worst 5% and 1% were. A mean hides a few very slow requests;
p99 shows them, and it is the number users notice.

**Keep-alive.** HTTP/1.1 lets one TCP connection carry many requests. Reusing
it means each timing measures the server's work, not a fresh TCP handshake.
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

# parents[1] of scripts/load_test.py is the repository root, so outputs land in
# the same place whichever directory the script is run from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def percentile(values: List[float], q: float) -> float:
    """
    Return the ``q``-th percentile (0-100) by the nearest-rank method.

    Nearest rank picks an actual measured value instead of interpolating
    between two, which is enough for latency reports and needs no numpy.
    """
    ordered = sorted(values)
    if not ordered:
        # An empty list (every request refused) reports 0 rather than raising.
        return 0.0
    # q/100 * (n-1) maps 0..100 onto index 0..n-1; min() guards against
    # rounding past the last element.
    return ordered[min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))]


def summary(ms: List[float]) -> Dict[str, Any]:
    """Summarise a list of millisecond timings as count, mean, p50/p95/p99 and max."""
    if not ms:
        # Only the count: a mean of nothing would be a made-up number.
        return {"n": 0}
    return {
        # fmean is the float-only mean: faster than statistics.mean and enough here.
        "n": len(ms), "meanMs": round(statistics.fmean(ms), 1), "p50Ms": round(percentile(ms, 50), 1),
        "p95Ms": round(percentile(ms, 95), 1), "p99Ms": round(percentile(ms, 99), 1), "maxMs": round(max(ms), 1),
    }


class Client:
    """One keep-alive connection, so the numbers are the server's, not connect() time."""

    def __init__(self, api: str) -> None:
        """Open one persistent connection to the host and port in ``api``."""
        parsed = urlparse(api)
        # Defaults cover a bare "http://localhost" with no explicit port.
        self.host, self.port = parsed.hostname or "localhost", parsed.port or 80
        # 300 s because a synthesis request in the in-memory queue runs the
        # whole model inside the request and can take minutes on CPU.
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=300)

    def call(self, method: str, path: str, body: Any = None) -> Tuple[int, float, Any]:
        """
        Send one request and time it.

        Returns ``(status, elapsed_ms, parsed_json_or_None)``. A network failure
        is returned as status 0 instead of raised, so one dropped connection
        counts as a failed request rather than ending the whole run.
        """
        data = json.dumps(body) if body is not None else None
        # perf_counter is a monotonic, high-resolution clock meant for measuring
        # intervals; time.time() can jump if the system clock is adjusted.
        started = time.perf_counter()
        try:
            self.conn.request(method, path, body=data, headers={"content-type": "application/json"})
            response = self.conn.getresponse()
            # The body must be read in full before the connection can be reused
            # for the next request; the timing therefore includes it.
            payload = response.read()
        except (OSError, http.client.HTTPException):
            # A broken keep-alive connection cannot be reused. Replace it so the
            # next call starts clean, and report this one as status 0.
            self.conn.close()
            self.conn = http.client.HTTPConnection(self.host, self.port, timeout=300)
            return 0, (time.perf_counter() - started) * 1000, None
        elapsed = (time.perf_counter() - started) * 1000
        try:
            return response.status, elapsed, json.loads(payload) if payload else None
        except ValueError:
            # Non-JSON bodies (an HTML error page, say) still count; the status
            # and timing are what the test needs.
            return response.status, elapsed, None


def initiation(api: str, jobs: int) -> Dict[str, Any]:
    """
    Measure N-03: time until the server accepts a job.

    First synthesises one short clip, because a render job needs real audio
    and phoneme timestamps to point at. Then submits ``jobs`` render jobs and
    times each 202. Synthesis POST latency is reported alongside, labelled as
    its own number because in the in-memory queue it includes the synthesis.
    """
    client = Client(api)
    status, ms, speech = client.call("POST", "/api/v1/audio/synthesize", {
        "text": "Load test clip.", "mode": "fast", "language": "en", "returnAlignment": True, "outputFilename": "loadtest.wav"})
    # The first call can include a lazy model load; it is kept but reported
    # separately as firstMs so it does not inflate the warm numbers.
    synth = [ms]
    for _ in range(2):  # warm second and third calls
        synth.append(client.call("POST", "/api/v1/audio/synthesize", {
            "text": "Load test clip.", "mode": "fast", "language": "en", "returnAlignment": False, "outputFilename": "loadtest.wav"})[1])
    # Poll the first task until it ends: its alignment is needed below. The
    # later warm calls used returnAlignment False, so only the first is waited on.
    state = speech
    while state and state.get("status") not in ("SUCCESS", "FAILED"):
        time.sleep(1)
        state = client.call("GET", f"/api/v1/audio/synthesize/{speech['taskId']}")[2]
    if not state or state.get("status") != "SUCCESS":
        # Without audio the render jobs below would be meaningless; stop loudly.
        raise SystemExit(f"setup synthesis failed: {state}")
    audio_url = f"{api}/outputs/loadtest.wav"
    # A timestamp in the jobId keeps ids unique across runs, so a rerun does not
    # collide with jobs already stored by an earlier one.
    stamp = int(time.time())
    latencies, codes = [], {}
    for i in range(jobs):
        # An AvatarRenderJob (backend/contracts.py), the only audio-to-vision
        # interface. PREVIEW keeps each queued render as cheap as possible.
        job = {
            "jobId": f"load-{stamp}-{i}", "avatarId": "demo", "audioUrl": audio_url, "sampleRate": 24000,
            "durationSeconds": state["durationSeconds"], "phonemeTimestamps": state["phonemeTimestamps"],
            "emotionVector": {"happy": 0, "neutral": 1, "eyeblinkRate": 1}, "renderQuality": "PREVIEW", "targetFps": 25,
        }
        code, ms, _ = client.call("POST", "/api/v1/avatar/render-job", job)
        codes[code] = codes.get(code, 0) + 1
        # Only accepted jobs count toward latency; refusals are visible in codes.
        if code == 202:
            latencies.append(ms)
    return {
        # under500ms is False when nothing was accepted: an empty run must not pass.
        "renderJobInitiation": {**summary(latencies), "statusCodes": codes, "under500ms": bool(latencies) and max(latencies) < 500},
        "synthesisPostLatency": {"firstMs": round(synth[0], 1), "warmMs": [round(x, 1) for x in synth[1:]], "httpStatus": status},
        "queuedRenders": jobs,
    }


def capacity(api: str, seconds: float, clients: int, path: str) -> Dict[str, Any]:
    """
    Measure N-04: requests per minute with ``clients`` concurrent threads.

    Each thread loops on GET ``path`` with its own keep-alive connection until
    the deadline. Accepted (2xx) and refused (429) are counted apart, so the
    rate limiter's policy is not mistaken for the server's raw capacity.
    """
    # One shared deadline, so every thread stops at the same moment.
    stop = time.perf_counter() + seconds
    # The lock protects the shared list and dict: several threads updating a
    # dict entry (read, add, write) at once could lose counts.
    lock = threading.Lock()
    ok_ms: List[float] = []
    codes: Dict[int, int] = {}

    def worker() -> None:
        """Loop requests on one connection until the deadline, recording each result."""
        # One connection per thread: http.client connections are not thread-safe.
        client = Client(api)
        while time.perf_counter() < stop:
            code, ms, _ = client.call("GET", path)
            with lock:
                codes[code] = codes.get(code, 0) + 1
                if 200 <= code < 300:
                    ok_ms.append(ms)

    started = time.perf_counter()
    # Threads, not processes: the work is waiting on the network, and the GIL is
    # released while a thread blocks on a socket.
    threads = [threading.Thread(target=worker) for _ in range(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Divide by the measured window, not the requested one: the last requests
    # finish a little after the deadline.
    window = time.perf_counter() - started
    accepted = sum(n for c, n in codes.items() if 200 <= c < 300)
    return {
        "path": path, "clients": clients, "seconds": round(window, 1), "statusCodes": codes,
        "acceptedPerMinute": round(accepted / window * 60, 1),
        "attemptedPerMinute": round(sum(codes.values()) / window * 60, 1),
        "refused429": codes.get(429, 0), "latencyOfAccepted": summary(ok_ms),
    }


def main() -> int:
    """Parse arguments, check the server is up, run both measurements and write the report."""
    # argparse builds --help from these declarations. description=__doc__ reuses
    # the module docstring, and RawDescriptionHelpFormatter keeps its line breaks.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000")
    # type= converts the command-line string; without it the value stays a str.
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=20, help="render jobs to submit for the initiation measurement")
    # A cheap read endpoint by default, so capacity measures the HTTP stack and
    # limiter rather than a model.
    parser.add_argument("--path", default="/api/v1/audio/languages?limit=5")
    # store_true makes a flag that is False unless given.
    parser.add_argument("--skip-initiation", action="store_true")
    args = parser.parse_args()

    # Fail fast with a clear message rather than timing connection errors.
    health = Client(args.api).call("GET", "/health")
    if health[0] != 200:
        print(f"error: no server answering at {args.api} (GET /health -> {health[0]})", file=sys.stderr)
        return 1
    # Record the conditions with the numbers: the queue backend changes what
    # "initiation" means, and the security block shows the rate limit in force.
    report: Dict[str, Any] = {
        "api": args.api, "queueBackend": health[2].get("queueBackend"), "security": health[2].get("security"),
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not args.skip_initiation:
        report.update(initiation(args.api, args.jobs))
    report["capacity"] = capacity(args.api, args.seconds, args.clients, args.path)
    out = PROJECT_ROOT / "outputs" / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    # A timestamped name keeps every run instead of overwriting the last one.
    destination = out / f"load-test-{time.strftime('%Y%m%d-%H%M%S')}.json"
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"written: {destination}")
    return 0


if __name__ == "__main__":
    # sys.exit passes main()'s return value to the shell as the exit code.
    sys.exit(main())
