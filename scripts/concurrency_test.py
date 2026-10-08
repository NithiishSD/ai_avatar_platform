#!/usr/bin/env python3
"""
Concurrency test against a RUNNING server (N-08): many render jobs at once.

Submits ``--jobs`` distinct render jobs from ``--jobs`` threads released
together, plus ``--duplicates`` deliberate repeats of already-used ids, then
waits for the queue to drain and checks the invariants a queue must keep:

* every distinct job was accepted (202) exactly once, every repeat refused (409);
* every accepted job reaches COMPLETED: none lost, none stuck, none failed;
* each job produced its own video file.

Throughput is the time from the first submission to the last COMPLETED.
Renders are one at a time by design (one heavy model, golden rule 5), so the
point is correctness under a burst, not parallel speed-up.

    python scripts/concurrency_test.py --jobs 60 --duplicates 12
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_test import PROJECT_ROOT, Client, summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--jobs", type=int, default=60)
    parser.add_argument("--duplicates", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()

    setup = Client(args.api)
    status, _, speech = setup.call("POST", "/api/v1/audio/synthesize", {
        "text": "Concurrency test clip.", "mode": "fast", "language": "en", "returnAlignment": True, "outputFilename": "concurrency.wav"})
    while speech and speech.get("status") not in ("SUCCESS", "FAILED"):
        time.sleep(1)
        speech = setup.call("GET", f"/api/v1/audio/synthesize/{speech['taskId']}")[2]
    if not speech or speech.get("status") != "SUCCESS":
        print(f"error: setup synthesis failed (HTTP {status}): {speech}", file=sys.stderr)
        return 1

    stamp = int(time.time())
    ids = [f"conc-{stamp}-{i}" for i in range(args.jobs)]
    repeats = [ids[i % args.jobs] for i in range(args.duplicates)]

    def payload(job_id: str) -> Dict[str, Any]:
        return {
            "jobId": job_id, "avatarId": "demo", "audioUrl": f"{args.api}/outputs/concurrency.wav", "sampleRate": 24000,
            "durationSeconds": speech["durationSeconds"], "phonemeTimestamps": speech["phonemeTimestamps"],
            "emotionVector": {"happy": 0, "neutral": 1, "eyeblinkRate": 1}, "renderQuality": "PREVIEW", "targetFps": 25,
        }

    # Distinct jobs first (all released together), then the repeats, also together:
    # a repeat can only be judged once the original is known to exist.
    results: Dict[str, List[int]] = {}
    latencies: List[float] = []
    lock = threading.Lock()

    def fire(batch: List[str]) -> None:
        barrier = threading.Barrier(len(batch))

        def go(job_id: str) -> None:
            client = Client(args.api)
            barrier.wait()
            code, ms, _ = client.call("POST", "/api/v1/avatar/render-job", payload(job_id))
            with lock:
                results.setdefault(job_id, []).append(code)
                latencies.append(ms)

        threads = [threading.Thread(target=go, args=(j,)) for j in batch]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    started = time.perf_counter()
    fire(ids)
    submitted = time.perf_counter() - started
    fire(repeats)

    # Drain.
    poller = Client(args.api)
    final: Dict[str, Dict[str, Any]] = {}
    deadline = time.perf_counter() + args.timeout
    while len(final) < len(ids) and time.perf_counter() < deadline:
        for job_id in ids:
            if job_id in final:
                continue
            state = poller.call("GET", f"/api/v1/avatar/render-job/{job_id}")[2] or {}
            if state.get("status") in ("COMPLETED", "FAILED"):
                final[job_id] = state
        time.sleep(1)
    drained = time.perf_counter() - started

    accepted = [j for j in ids if 202 in results.get(j, [])]
    refused_repeats = sum(1 for j in set(repeats) if results[j].count(409) == repeats.count(j))
    files = {Path(s.get("videoUrl", "")).name for s in final.values() if s.get("status") == "COMPLETED"}
    checks = {
        "everyDistinctJobAccepted202": len(accepted) == len(ids) and all(results[j].count(202) == 1 for j in ids),
        "everyRepeatRefused409": refused_repeats == len(set(repeats)) and all(c in (202, 409) for v in results.values() for c in v),
        "noJobLost": len(final) == len(ids),
        "noJobFailed": all(s.get("status") == "COMPLETED" for s in final.values()),
        "eachJobOwnVideoFile": len(files) == len(ids),
    }
    report = {
        "api": args.api, "distinctJobs": len(ids), "repeatSubmissions": len(repeats),
        "submitAllSeconds": round(submitted, 2), "drainAllSeconds": round(drained, 1),
        "jobsPerMinute": round(len(final) / drained * 60, 1), "acceptLatency": summary(latencies),
        "statusCodes": {str(c): sum(v.count(c) for v in results.values()) for c in sorted({c for v in results.values() for c in v})},
        "completed": sum(1 for s in final.values() if s.get("status") == "COMPLETED"),
        "failed": [(j, s.get("error")) for j, s in final.items() if s.get("status") == "FAILED"][:5],
        "checks": checks, "passed": all(checks.values()),
    }
    out = PROJECT_ROOT / "outputs" / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"concurrency-{time.strftime('%Y%m%d-%H%M%S')}.json"
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"written: {destination}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
