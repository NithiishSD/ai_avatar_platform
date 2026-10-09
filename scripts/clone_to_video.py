#!/usr/bin/env python3
"""
Cloned voice -> lip-synced video, through the public HTTP API only.

This is integration Gate 2 as one command: it clones a consented reference
voice, hands the result to the render worker as the frozen AvatarRenderJob
contract, waits for the video, and then measures both halves - lip sync with
SyncNet and voice similarity with ECAPA-TDNN. It talks to a running server
exactly as any client would, so it exercises routing, validation, the queue
and the static file mount, not just the Python functions behind them.

    cd backend && PYTHONPATH=. ./.conda/bin/python -m uvicorn app:app --port 8000
    backend/.conda/bin/python scripts/clone_to_video.py \\
        --text "Hello, this is my cloned voice." \\
        --voice inputs/ljspeech_reference.wav --face demo --clone-engine openvoice-v2

Standard library only (urllib), so it runs from any Python with no install.
Exit code 0 when the video rendered, 1 on any failure (the reason is printed).
It writes nothing itself: the server writes the audio and video under ``outputs/``, and this
script prints the evidence (or JSON with ``--json``).

Concepts used here:

**Polling** means asking the server "is it done yet?" on a fixed interval. Rendering takes
minutes, longer than one HTTP request should stay open, so the API answers at once with a job
id and the client checks the job's status until it is terminal.

**SSRF** (server-side request forgery) is when a client tricks a server into fetching a URL of
the attacker's choosing, such as an internal service. The render worker refuses any audio URL
that is not one of this server's own ``/outputs/`` files, which is why the URL is built below
from the path the server itself returned.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# The repository root. The server publishes PROJECT_ROOT/outputs at the /outputs/ URL, so this is
# needed to turn a file path from a response back into a URL.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ApiError(RuntimeError):
    """An HTTP error from the server, carrying its status and `detail`."""


def call(api: str, method: str, path: str, body: Optional[Dict[str, Any]] = None, timeout: float = 1800) -> Dict[str, Any]:
    """One JSON request. Raises ApiError with the server's own explanation."""
    # urllib sends bytes, not dicts. data=None makes a request with no body, as GET needs.
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(api + path, data=data, method=method, headers={"content-type": "application/json"})
    try:
        # The default timeout is a generous 30 minutes: in in_memory mode the server synthesises
        # before it answers, which can take minutes on CPU. The `with` closes the connection even
        # on an exception.
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as err:
        # The API's errors say how to fix themselves; surface them verbatim.
        # An HTTPError is also a readable response, so its JSON body can be parsed. FastAPI
        # puts the explanation under "detail".
        try:
            detail = json.load(err).get("detail")
        # Not JSON (for example a proxy's HTML error page): fall back to the status phrase.
        except ValueError:
            detail = err.reason
        # `from err` chains the original exception, so a traceback still shows the raw HTTP error.
        raise ApiError(f"{method} {path} -> HTTP {err.code}: {detail}") from err


def wait_for(api: str, path: str, done: Tuple[str, ...], every: float = 1.0, limit: float = 3600) -> Dict[str, Any]:
    """Poll a job until its status is one of `done` (or give up after `limit` s)."""
    # monotonic() is a clock that never jumps backwards (unlike wall time when the system clock
    # is adjusted), so it is the right one for measuring elapsed time.
    started = time.monotonic()
    while True:
        state = call(api, "GET", path)
        if state.get("status") in done:
            return state
        if time.monotonic() - started > limit:
            raise ApiError(f"{path} still {state.get('status')} after {limit:.0f}s")
        time.sleep(every)


def main() -> int:
    """Clone, render and measure, printing the evidence. Returns the process exit code."""
    # description=__doc__ shows the module docstring as --help; the Raw formatter keeps its layout.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", required=True, help="reference recording with a provenance sidecar")
    parser.add_argument("--face", required=True, help="avatarId registered with scripts/make_avatar.py")
    parser.add_argument("--language", default="en")
    # None means "let the server pick its default clone engine"; choices= makes argparse reject typos.
    parser.add_argument("--clone-engine", default=None, choices=["xtts-v2", "openvoice-v2"])
    parser.add_argument("--engine", default="wav2lip", choices=["blendshape", "wav2lip", "sadtalker"], help="lip-sync engine")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--json", action="store_true", help="print the evidence as JSON")
    args = parser.parse_args()

    # A timestamped id keeps repeated runs from overwriting each other's outputs.
    job_id = args.job_id or f"clone-{int(time.time())}"
    # The server opens the reference file itself, so it needs an absolute path, not one relative
    # to wherever this script was launched from.
    voice = str(Path(args.voice).resolve())
    timings: Dict[str, float] = {}
    try:
        # 1. Clone. In in_memory mode the server runs the task eagerly and
        #    answers SUCCESS at once; under Celery it answers QUEUED and we poll.
        # perf_counter() is the highest-resolution clock available, meant for timing a span.
        started = time.perf_counter()
        # returnAlignment asks for per-phoneme timings: the lip-sync step needs them.
        body: Dict[str, Any] = {
            "text": args.text, "mode": "clone", "language": args.language, "speakerWav": voice,
            "returnAlignment": True, "outputFilename": f"{job_id}.wav",
        }
        if args.clone_engine:
            body["cloneEngine"] = args.clone_engine
        speech = call(args.api, "POST", "/api/v1/audio/synthesize", body)
        if speech.get("status") not in ("SUCCESS", "FAILED"):
            # UNKNOWN is terminal too: polling a task id the server no longer knows would never end.
            speech = wait_for(args.api, f"/api/v1/audio/synthesize/{speech['taskId']}", ("SUCCESS", "FAILED", "UNKNOWN"))
        if speech.get("status") != "SUCCESS":
            raise ApiError(f"synthesis ended {speech.get('status')}: {speech}")
        timings["speechSeconds"] = round(time.perf_counter() - started, 2)
        if not speech.get("phonemeTimestamps"):
            raise ApiError("synthesis returned no phoneme timestamps, so there is nothing to lip-sync")

        # 2. Render. The frozen contract, built from the speech response. The
        #    worker only reads this server's own /outputs/ URLs (no SSRF), so
        #    the audio is addressed exactly as the server published it.
        # relative_to raises ValueError if the audio is not under outputs/, which the except
        # clause below reports rather than sending the worker a URL it would refuse.
        relative = Path(speech["outputPath"]).resolve().relative_to(PROJECT_ROOT / "outputs")
        job = {
            "jobId": job_id, "avatarId": args.face,
            # quote() percent-encodes characters such as spaces that are not allowed in a URL.
            "audioUrl": f"{args.api}/outputs/{urllib.parse.quote(relative.as_posix())}",
            "sampleRate": 24000, "durationSeconds": speech["durationSeconds"],
            "phonemeTimestamps": speech["phonemeTimestamps"],
            # A fixed neutral expression, so the measurement reflects the voice and lips only.
            "emotionVector": {"happy": 0.0, "neutral": 1.0, "eyeblinkRate": 1.0},
            "renderQuality": "PREVIEW", "targetFps": 25,
        }
        started = time.perf_counter()
        # The POST only queues the job; its response is ignored and the job is polled by id.
        call(args.api, "POST", f"/api/v1/avatar/render-job?engine={args.engine}", job)
        render = wait_for(args.api, f"/api/v1/avatar/render-job/{urllib.parse.quote(job_id)}", ("COMPLETED", "FAILED"))
        timings["renderSeconds"] = round(time.perf_counter() - started, 2)
        if render["status"] != "COMPLETED":
            raise ApiError(f"render failed: {render.get('error')}")

        # 3. Measure both halves, each naming its method.
        sync = call(args.api, "POST", f"/api/v1/avatar/render-job/{urllib.parse.quote(job_id)}/lipsync-score")["score"]
        voice_match = call(args.api, "POST", "/api/v1/audio/voice-similarity",
                           {"referencePath": voice, "generatedPath": speech["outputPath"]})["report"]
    # URLError: the server is not reachable. KeyError: a response lacked an expected field.
    # ValueError: a body was not JSON, or the audio path was outside outputs/.
    except (ApiError, urllib.error.URLError, KeyError, ValueError) as err:
        print(f"FAILED: {err}", file=sys.stderr)
        return 1

    # One flat record of what ran and what it scored. Each score keeps its `method` so the number
    # can be judged as evidence (golden rule 2). `**timings` merges the timing keys in.
    evidence = {
        "jobId": job_id, "model": speech.get("modelUsed"), "backend": (speech.get("language") or {}).get("backend"),
        "alignmentMethod": speech.get("alignmentMethod"), "audioSeconds": speech["durationSeconds"],
        "videoUrl": render.get("videoUrl"), "renderEngine": render.get("engine"), **timings,
        "lipSync": {k: sync.get(k) for k in ("offsetFrames", "lseC", "lseD", "method")},
        "voiceSimilarity": {k: voice_match.get(k) for k in ("similarityPercent", "method", "passesTarget")},
    }
    if args.json:
        print(json.dumps(evidence, indent=2))
    else:
        print(f"voice      {evidence['model']} ({evidence['backend']}), {evidence['audioSeconds']:.2f}s, "
              f"alignment {evidence['alignmentMethod']}, {timings['speechSeconds']}s")
        print(f"video      {evidence['videoUrl']} via {evidence['renderEngine']}, {timings['renderSeconds']}s")
        print(f"lip sync   offset {sync.get('offsetFrames')} frames, LSE-C {sync.get('lseC')}, LSE-D {sync.get('lseD')} ({sync.get('method')})")
        print(f"similarity {voice_match.get('similarityPercent')}% ({voice_match.get('method')}), "
              f"target met: {voice_match.get('passesTarget')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
