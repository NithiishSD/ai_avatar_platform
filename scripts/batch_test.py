#!/usr/bin/env python3
"""
Live check of the batch render endpoint against a RUNNING server (T8.1).

Sends one batch of ``--jobs`` good render jobs plus three deliberately bad ones (an unknown avatar, a
job with no phonemes, a repeated id) and checks what the endpoint promises: the bad items are
refused by index with the reason, every good item is accepted and reaches COMPLETED, none is lost,
and each makes its own video file.

    python scripts/batch_test.py --jobs 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_test import PROJECT_ROOT, Client  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    api = Client(args.api)

    _, _, speech = api.call("POST", "/api/v1/audio/synthesize", {
        "text": "Batch test clip.", "mode": "fast", "language": "en", "returnAlignment": True, "outputFilename": "batch.wav"})
    while speech and speech.get("status") not in ("SUCCESS", "FAILED"):
        time.sleep(1)
        speech = api.call("GET", f"/api/v1/audio/synthesize/{speech['taskId']}")[2]
    if not speech or speech.get("status") != "SUCCESS":
        print(f"error: setup synthesis failed: {speech}", file=sys.stderr)
        return 1

    def job(job_id: str, **override: Any) -> Dict[str, Any]:
        body = {
            "jobId": job_id, "avatarId": "demo", "audioUrl": f"{args.api}/outputs/batch.wav", "sampleRate": 24000,
            "durationSeconds": speech["durationSeconds"], "phonemeTimestamps": speech["phonemeTimestamps"],
            "emotionVector": {"happy": 0, "neutral": 1, "eyeblinkRate": 1}, "renderQuality": "PREVIEW", "targetFps": 25,
        }
        body.update(override)
        return body

    stamp = int(time.time())
    good = [job(f"batch-{stamp}-{i}") for i in range(args.jobs)]
    bad_shape = job(f"batch-{stamp}-noph")
    del bad_shape["phonemeTimestamps"]
    batch = good + [job(f"batch-{stamp}-ghost", avatarId="ghost"), bad_shape, good[0]]

    started = time.perf_counter()
    status, accept_ms, body = api.call("POST", "/api/v1/avatar/render-batch", {"jobs": batch})
    print(f"POST render-batch -> {status} in {accept_ms:.0f} ms: accepted {body['accepted']}, rejected {body['rejected']}")
    for item in body["jobs"]:
        if not item["accepted"]:
            print(f"  item {item['index']} refused ({item['httpStatus']}): {item['detail']}")

    final: Dict[str, Any] = {}
    while time.perf_counter() - started < args.timeout:
        final = api.call("GET", f"/api/v1/avatar/render-batch/{body['batchId']}")[2]
        if final["done"]:
            break
        time.sleep(1)
    elapsed = time.perf_counter() - started
    print(f"batch {body['batchId']} counts {final['counts']} done={final['done']} after {elapsed:.1f} s")

    videos = {j["videoUrl"] for j in final["jobs"] if j.get("videoUrl")}
    files = [(PROJECT_ROOT / v.lstrip("/")).exists() for v in videos]
    ok = (
        status == 202 and body["accepted"] == args.jobs and body["rejected"] == 3
        and final["counts"] == {"COMPLETED": args.jobs, "REJECTED": 3}
        and len(videos) == args.jobs and all(files)
    )
    print("PASS" if ok else "FAIL", f"({len(videos)} distinct videos, {sum(files)} on disk)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
