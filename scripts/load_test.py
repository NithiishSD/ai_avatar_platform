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

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))]


def summary(ms: List[float]) -> Dict[str, Any]:
    if not ms:
        return {"n": 0}
    return {
        "n": len(ms), "meanMs": round(statistics.fmean(ms), 1), "p50Ms": round(percentile(ms, 50), 1),
        "p95Ms": round(percentile(ms, 95), 1), "p99Ms": round(percentile(ms, 99), 1), "maxMs": round(max(ms), 1),
    }


class Client:
    """One keep-alive connection, so the numbers are the server's, not connect() time."""

    def __init__(self, api: str) -> None:
        parsed = urlparse(api)
        self.host, self.port = parsed.hostname or "localhost", parsed.port or 80
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=300)

    def call(self, method: str, path: str, body: Any = None) -> Tuple[int, float, Any]:
        data = json.dumps(body) if body is not None else None
        started = time.perf_counter()
        try:
            self.conn.request(method, path, body=data, headers={"content-type": "application/json"})
            response = self.conn.getresponse()
            payload = response.read()
        except (OSError, http.client.HTTPException):
            self.conn.close()
            self.conn = http.client.HTTPConnection(self.host, self.port, timeout=300)
            return 0, (time.perf_counter() - started) * 1000, None
        elapsed = (time.perf_counter() - started) * 1000
        try:
            return response.status, elapsed, json.loads(payload) if payload else None
        except ValueError:
            return response.status, elapsed, None


def initiation(api: str, jobs: int) -> Dict[str, Any]:
    client = Client(api)
    status, ms, speech = client.call("POST", "/api/v1/audio/synthesize", {
        "text": "Load test clip.", "mode": "fast", "language": "en", "returnAlignment": True, "outputFilename": "loadtest.wav"})
    synth = [ms]
    for _ in range(2):  # warm second and third calls
        synth.append(client.call("POST", "/api/v1/audio/synthesize", {
            "text": "Load test clip.", "mode": "fast", "language": "en", "returnAlignment": False, "outputFilename": "loadtest.wav"})[1])
    state = speech
    while state and state.get("status") not in ("SUCCESS", "FAILED"):
        time.sleep(1)
        state = client.call("GET", f"/api/v1/audio/synthesize/{speech['taskId']}")[2]
    if not state or state.get("status") != "SUCCESS":
        raise SystemExit(f"setup synthesis failed: {state}")
    audio_url = f"{api}/outputs/loadtest.wav"
    stamp = int(time.time())
    latencies, codes = [], {}
    for i in range(jobs):
        job = {
            "jobId": f"load-{stamp}-{i}", "avatarId": "demo", "audioUrl": audio_url, "sampleRate": 24000,
            "durationSeconds": state["durationSeconds"], "phonemeTimestamps": state["phonemeTimestamps"],
            "emotionVector": {"happy": 0, "neutral": 1, "eyeblinkRate": 1}, "renderQuality": "PREVIEW", "targetFps": 25,
        }
        code, ms, _ = client.call("POST", "/api/v1/avatar/render-job", job)
        codes[code] = codes.get(code, 0) + 1
        if code == 202:
            latencies.append(ms)
    return {
        "renderJobInitiation": {**summary(latencies), "statusCodes": codes, "under500ms": bool(latencies) and max(latencies) < 500},
        "synthesisPostLatency": {"firstMs": round(synth[0], 1), "warmMs": [round(x, 1) for x in synth[1:]], "httpStatus": status},
        "queuedRenders": jobs,
    }


def capacity(api: str, seconds: float, clients: int, path: str) -> Dict[str, Any]:
    stop = time.perf_counter() + seconds
    lock = threading.Lock()
    ok_ms: List[float] = []
    codes: Dict[int, int] = {}

    def worker() -> None:
        client = Client(api)
        while time.perf_counter() < stop:
            code, ms, _ = client.call("GET", path)
            with lock:
                codes[code] = codes.get(code, 0) + 1
                if 200 <= code < 300:
                    ok_ms.append(ms)

    started = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    window = time.perf_counter() - started
    accepted = sum(n for c, n in codes.items() if 200 <= c < 300)
    return {
        "path": path, "clients": clients, "seconds": round(window, 1), "statusCodes": codes,
        "acceptedPerMinute": round(accepted / window * 60, 1),
        "attemptedPerMinute": round(sum(codes.values()) / window * 60, 1),
        "refused429": codes.get(429, 0), "latencyOfAccepted": summary(ok_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=20, help="render jobs to submit for the initiation measurement")
    parser.add_argument("--path", default="/api/v1/audio/languages?limit=5")
    parser.add_argument("--skip-initiation", action="store_true")
    args = parser.parse_args()

    health = Client(args.api).call("GET", "/health")
    if health[0] != 200:
        print(f"error: no server answering at {args.api} (GET /health -> {health[0]})", file=sys.stderr)
        return 1
    report: Dict[str, Any] = {
        "api": args.api, "queueBackend": health[2].get("queueBackend"), "security": health[2].get("security"),
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not args.skip_initiation:
        report.update(initiation(args.api, args.jobs))
    report["capacity"] = capacity(args.api, args.seconds, args.clients, args.path)
    out = PROJECT_ROOT / "outputs" / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"load-test-{time.strftime('%Y%m%d-%H%M%S')}.json"
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"written: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
