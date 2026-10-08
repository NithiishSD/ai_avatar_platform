#!/usr/bin/env python3
"""
Live check of the batch render endpoint against a RUNNING server (T8.1).

Sends one batch of ``--jobs`` good render jobs plus three deliberately bad ones (an unknown avatar, a
job with no phonemes, a repeated id) and checks what the endpoint promises: the bad items are
refused by index with the reason, every good item is accepted and reaches COMPLETED, none is lost,
and each makes its own video file.

    python scripts/batch_test.py --jobs 10

How it works, in order:

1. Synthesise one short clip (``outputs/batch.wav``) with the fast TTS route and ask for its
   phoneme alignment, because a render job needs both the audio and the timings.
2. Build ``--jobs`` good jobs that share that clip, then add the three bad items.
3. POST them as one batch to ``/api/v1/avatar/render-batch`` and poll the batch until done.
4. Compare the counts and the video files on disk with what should have happened.

Writes nothing of its own: the server writes the clip and the renders. Prints PASS or FAIL and
exits 0 or 1, so a shell script or CI step can gate on it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Scripts in this folder are not a package, so put the folder itself on the import path to
# reuse load_test's HTTP client instead of writing a second one.
sys.path.insert(0, str(Path(__file__).resolve().parent))
# E402 ("import not at top of file") is expected: the import has to follow the path change.
from load_test import PROJECT_ROOT, Client  # noqa: E402


def main() -> int:
    """Run the batch check once; return 0 on PASS and 1 on FAIL or failed setup."""
    # argparse turns command-line flags into ``args.<name>`` attributes. Passing the module
    # docstring as the description makes ``--help`` print the usage text above, and
    # RawDescriptionHelpFormatter keeps its line breaks instead of re-wrapping them.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--jobs", type=int, default=10)
    # Seconds to wait for the whole batch. Renders run one at a time, so this scales with --jobs.
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    # Client.call returns (HTTP status, latency in ms, decoded JSON body); index [2] below is the body.
    api = Client(args.api)

    # Setup: synthesis is asynchronous. The POST returns a task id at once and the audio is made
    # in the background, so poll once a second until the task reaches a terminal state.
    _, _, speech = api.call("POST", "/api/v1/audio/synthesize", {
        "text": "Batch test clip.", "mode": "fast", "language": "en", "returnAlignment": True, "outputFilename": "batch.wav"})
    while speech and speech.get("status") not in ("SUCCESS", "FAILED"):
        time.sleep(1)
        speech = api.call("GET", f"/api/v1/audio/synthesize/{speech['taskId']}")[2]
    # Without a clip there is nothing to render, so stop here rather than report a misleading FAIL.
    if not speech or speech.get("status") != "SUCCESS":
        print(f"error: setup synthesis failed: {speech}", file=sys.stderr)
        return 1

    def job(job_id: str, **override: Any) -> Dict[str, Any]:
        """
        One AvatarRenderJob body (see backend/contracts.py) for the shared clip.

        ``**override`` collects any extra keyword arguments into a dict, so a caller can replace
        one field (for example ``avatarId="ghost"``) and keep every other field valid.
        """
        body = {
            "jobId": job_id, "avatarId": "demo", "audioUrl": f"{args.api}/outputs/batch.wav", "sampleRate": 24000,
            "durationSeconds": speech["durationSeconds"], "phonemeTimestamps": speech["phonemeTimestamps"],
            "emotionVector": {"happy": 0, "neutral": 1, "eyeblinkRate": 1}, "renderQuality": "PREVIEW", "targetFps": 25,
        }
        body.update(override)
        return body

    # A timestamp in every id keeps ids unique across runs, so a re-run is not refused as a repeat.
    stamp = int(time.time())
    good = [job(f"batch-{stamp}-{i}") for i in range(args.jobs)]
    # Bad item 2: a job missing a required field, which the contract validation must refuse.
    bad_shape = job(f"batch-{stamp}-noph")
    del bad_shape["phonemeTimestamps"]
    # Bad items 1 and 3: an avatar that does not exist, and good[0] sent a second time (a repeated id).
    # They go last, so their indexes in the response are len(good), len(good)+1 and len(good)+2.
    batch = good + [job(f"batch-{stamp}-ghost", avatarId="ghost"), bad_shape, good[0]]

    # perf_counter is a monotonic clock: it never jumps when the system clock is adjusted,
    # which makes it the right clock for measuring elapsed time.
    started = time.perf_counter()
    status, accept_ms, body = api.call("POST", "/api/v1/avatar/render-batch", {"jobs": batch})
    print(f"POST render-batch -> {status} in {accept_ms:.0f} ms: accepted {body['accepted']}, rejected {body['rejected']}")
    # Each refused item names its index, status code and reason, so the user can see which one failed and why.
    for item in body["jobs"]:
        if not item["accepted"]:
            print(f"  item {item['index']} refused ({item['httpStatus']}): {item['detail']}")

    # Poll the batch summary until the server says every accepted job has finished, or time runs out.
    final: Dict[str, Any] = {}
    while time.perf_counter() - started < args.timeout:
        final = api.call("GET", f"/api/v1/avatar/render-batch/{body['batchId']}")[2]
        if final["done"]:
            break
        time.sleep(1)
    elapsed = time.perf_counter() - started
    print(f"batch {body['batchId']} counts {final['counts']} done={final['done']} after {elapsed:.1f} s")

    # A set comprehension removes duplicates: two jobs that wrote the same file would count once,
    # so "one distinct video per job" shows up as a short set.
    videos = {j["videoUrl"] for j in final["jobs"] if j.get("videoUrl")}
    # videoUrl is a server path such as "/outputs/renders/x.mp4"; strip the leading slash so it
    # joins onto the project root instead of replacing it.
    files = [(PROJECT_ROOT / v.lstrip("/")).exists() for v in videos]
    # Every promise at once: 202 for the batch, exactly the 3 bad items refused, all good ones
    # COMPLETED, and one distinct video file on disk per good job.
    ok = (
        status == 202 and body["accepted"] == args.jobs and body["rejected"] == 3
        and final["counts"] == {"COMPLETED": args.jobs, "REJECTED": 3}
        and len(videos) == args.jobs and all(files)
    )
    print("PASS" if ok else "FAIL", f"({len(videos)} distinct videos, {sum(files)} on disk)")
    return 0 if ok else 1


# Runs only when executed as a script, not when imported. sys.exit passes the return code to the shell.
if __name__ == "__main__":
    sys.exit(main())
