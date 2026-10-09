#!/usr/bin/env python
"""
Text + face -> talking avatar MP4, in one command (task G2-05).

    PYTHONPATH=backend backend/.conda/bin/python scripts/render_avatar.py \\
        --text "Hello, I am a synthetic avatar." --face demo

    # cloned voice (needs XTTS-v2 weights and a consented reference)
    ... --voice inputs/reference.wav

    # other language / emotion / engine
    ... --language hin --emotion joy --engine wav2lip --quality 1080P_HQ

The script is the pipeline the API runs, in order and in one process:
synthesize speech with forced alignment, build the frozen ``AvatarRenderJob``
from the result, hand it to the render worker. It prints which speech model
and which render engine actually ran, and ``--metric`` scores the result with
SyncNet.

Exit codes: 0 rendered, 1 failed, 2 alignment produced no timestamps.

What it writes: the speech WAV and the MP4 under ``outputs/renders/`` (named after the job id,
``cli-<date>-<time>`` unless ``--job-id`` is given), plus whatever the render worker writes beside
the video. On the terminal it prints a summary, or one JSON document with ``--json``.

Concepts used here:

* **Forced alignment** finds when each phoneme starts and ends in the synthesised audio. Those
  timestamps are what move the mouth; without them there is nothing to animate, hence exit code 2.
* **The render job** (``AvatarRenderJob`` in ``backend/contracts.py``) is the one frozen interface
  between the audio side and the vision side. This script builds it exactly as the API does, so a
  render here is the same render the server would make.
* **Render engines**: ``blendshape`` moves a face mesh from viseme weights; ``wav2lip`` regenerates the
  mouth region from the audio with a neural network. ``--engine`` picks one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

# parents[1] of scripts/render_avatar.py is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The backend modules import each other by bare name, so backend/ goes on the import path.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


def background_spec(value: str) -> dict:
    """``--background`` is a colour when it looks like one, otherwise a file."""
    # A hex colour such as "#1e293b" starts with "#"; a file path never needs to.
    if value.startswith("#"):
        return {"color": value}
    # The contract takes the image as a URL; as_uri() turns an absolute path into "file:///...".
    # resolve() first, because as_uri() refuses a relative path.
    return {"imageUrl": Path(value).resolve().as_uri()}


def main(argv: Optional[List[str]] = None) -> int:
    """
    Parse the arguments, run speech then render, and report; returns the exit code.

    ``argv`` defaults to the real command line; passing a list lets a test call ``main`` directly.
    """
    # RawDescriptionHelpFormatter keeps the module docstring's line breaks in --help.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Every option below maps onto one field of the speech request or the render job.
    parser.add_argument("--text", required=True)
    parser.add_argument("--face", required=True, help="avatarId registered with scripts/make_avatar.py")
    parser.add_argument("--voice", default=None, help="reference WAV to clone (switches mode to 'clone')")
    parser.add_argument("--mode", default=None, help="fast | clone | high_quality | dialogue | multilingual")
    parser.add_argument("--language", default="en")
    parser.add_argument("--emotion", default=None)
    parser.add_argument("--engine", default=None, help="blendshape (default) or wav2lip")
    parser.add_argument("--quality", default="PREVIEW", choices=["PREVIEW", "1080P_HQ"])
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--job-id", default=None)
    parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="force every model onto this device (use cpu when another process holds the GPU)",
    )
    parser.add_argument("--no-label", action="store_true", help="do not burn in the AI-generated label")
    parser.add_argument("--metric", action="store_true", help="score lip sync with SyncNet (LSE-C / LSE-D)")
    parser.add_argument("--jitter", action="store_true", help="measure frame-to-frame landmark jitter (N-09)")
    parser.add_argument(
        "--background", default=None, help="replace the photo's background: '#rrggbb' or an image under outputs/ or inputs/"
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)

    # AVATAR_DEVICE is read by gpu_utils when each model picks its device, so it must be set
    # before any model module is imported or loaded below.
    if args.device:
        # os is imported only on this path, where it is used.
        import os

        os.environ["AVATAR_DEVICE"] = args.device

    # Imported after argument parsing, so --help and argument errors return without loading them.
    from avatar_store import AvatarConsentError, AvatarError, AvatarNotFound, AvatarStore
    from contracts import AvatarRenderJob
    import render_engine

    # The job id names the output files; a timestamp keeps separate runs from overwriting each other.
    job_id = args.job_id or f"cli-{time.strftime('%Y%m%d-%H%M%S')}"

    # Fail on the face before spending time on speech.
    try:
        # require_usable refuses a face whose provenance sidecar does not allow animation
        # (golden rule 3: consent before use).
        AvatarStore().require_usable(args.face)
        # validate_engine rejects an unknown --engine name and lists the valid ones.
        render_engine.validate_engine(args.engine)
    except (AvatarError, AvatarNotFound, AvatarConsentError, render_engine.RenderError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    # The speech stack is imported only after the face check has passed.
    from emotion_engine import to_render_emotion_vector
    from voice_engine import VoiceEngineRouter

    # The speech engine prints progress banners; with --json, stdout has to
    # carry the JSON document and nothing else.
    # Keep a handle on the real stdout to print the JSON to at the end; every other print
    # in the meantime goes to stderr.
    real_stdout = sys.stdout
    if args.json:
        sys.stdout = sys.stderr

    # An explicit --mode wins; otherwise a reference voice implies cloning, and plain text uses
    # the fast engine.
    mode = args.mode or ("clone" if args.voice else "fast")
    # The router picks the speech model for the mode and reports which one ran (speech.model).
    speech = VoiceEngineRouter(device=args.device).synthesize(
        text=args.text,
        mode=mode,
        speaker_wav=args.voice,
        language=args.language,
        emotion=args.emotion,
        # The phoneme timestamps are required: they are what drives the mouth.
        return_alignment=True,
        # Name the WAV after the video it belongs to, so the two sit together in outputs/.
        output_filename=f"{render_engine.output_path_for(job_id).stem}.wav",
    )
    # An empty list means the aligner ran but found nothing; render nothing rather than a still face.
    if not speech.phoneme_timestamps:
        print(
            "error: forced alignment returned no timestamps, so there is nothing "
            "to drive the mouth. See the alignment warning above.",
            file=sys.stderr,
        )
        return 2

    audio_path = Path(speech.output_path).resolve()
    # The contract rejects a timestamp that ends after the audio does, so trailing stamps that
    # overrun the measured duration are dropped rather than failing the whole job.
    duration_ms = speech.duration_seconds * 1000.0
    timestamps = [t for t in speech.phoneme_timestamps if t["endMs"] <= duration_ms]
    # The speech side's emotion vector is projected onto the contract's EmotionVector shape.
    # "or {}" covers both a missing emotion result and one without a vector.
    emotion_vector = to_render_emotion_vector((speech.emotion or {}).get("vector") or {})

    try:
        # model_validate runs Pydantic's checks on the camelCase dictionary, the same wire shape
        # the API receives, and either returns a valid job or raises.
        job = AvatarRenderJob.model_validate(
            {
                "jobId": job_id,
                "avatarId": args.face,
                "audioUrl": audio_path.as_uri(),
                "sampleRate": speech.sample_rate,
                "durationSeconds": speech.duration_seconds,
                "phonemeTimestamps": timestamps,
                "emotionVector": emotion_vector,
                "renderQuality": args.quality,
                "targetFps": args.fps,
                # Add the optional field only when asked: spreading an empty dict adds nothing,
                # which keeps "no background" distinct from "background: null".
                **({"background": background_spec(args.background)} if args.background else {}),
            }
        )
        # label=True burns in the visible "AI-generated" label; --no-label turns it off.
        result = render_engine.render_job(job, engine=args.engine, label=not args.no_label)
    # Any failure in validation or rendering is printed as one line and becomes exit code 1.
    except Exception as err:  # noqa: BLE001 - the CLI reports, it does not trace
        print(f"error: {type(err).__name__}: {err}", file=sys.stderr)
        return 1

    # One dictionary holds everything reported, so the --json output and the text summary below
    # come from the same values.
    payload = {
        "speech": {
            "model": speech.model,
            "mode": speech.mode,
            "audioPath": str(audio_path),
            "durationSeconds": round(speech.duration_seconds, 3),
            "phonemeCount": len(timestamps),
            # Which aligner produced the timings; golden rule 1 says it must be reported, never hidden.
            "alignmentMethod": speech.alignment_method,
        },
        "render": result.to_dict(),
    }

    # Optional measurements. Each failure is recorded in the report instead of failing the run:
    # the video was already rendered and is still worth returning.
    if args.metric:
        # SyncNet: compares the mouth movement with the audio and gives LSE-C / LSE-D.
        import lipsync_metric

        try:
            payload["lipSync"] = lipsync_metric.score_video(result.output_path).to_dict()
        except Exception as err:  # noqa: BLE001
            payload["lipSync"] = {"error": f"{type(err).__name__}: {err}"}

    if args.jitter:
        # Jitter: how much the face landmarks shake from one frame to the next (N-09).
        import jitter_metric

        try:
            payload["jitter"] = jitter_metric.score_video(result.output_path).to_dict()
        except Exception as err:  # noqa: BLE001
            payload["jitter"] = {"error": f"{type(err).__name__}: {err}"}

    # In --json mode the document goes to the saved real stdout, and the human summary is skipped.
    if args.json:
        print(json.dumps(payload, indent=2), file=real_stdout)
        return 0

    # media holds what ffprobe read back from the finished file: the real stream durations and
    # codecs, so a gap between audio and video length is visible here.
    media = result.media
    print("\n" + "=" * 60)
    print(f"Video         : {result.output_path}")
    print(f"Speech model  : {speech.model} (mode {speech.mode}), {len(timestamps)} phonemes")
    print(f"Alignment     : {speech.alignment_method}")
    # The fallback guesses timing from the sound; say so loudly rather than pass it off as measured.
    if speech.alignment_method == "acoustic-fallback":
        print("Warning       : phoneme timing was ESTIMATED, not measured (see the aligner warning above)")
    print(f"Render engine : {result.engine}")
    print(f"Frames        : {result.frame_count} at {result.fps} fps, {result.width}x{result.height}")
    print(
        f"Streams       : video {media['videoDuration']} s ({media['videoCodec']}), "
        f"audio {media['audioDuration']} s ({media['audioCodec']}), gap {media['durationGap']} s"
    )
    # The real-time factor is render seconds per second of video; below 1.0 is faster than real time.
    print(f"Render time   : {result.render_seconds:.2f} s ({result.realtime_factor:.2f}x real time)")
    # Measured only for wav2lip, the engine that uses the GPU, and None without CUDA.
    if result.peak_vram_mb is not None:
        print(f"Peak VRAM     : {result.peak_vram_mb} MiB")
    # Everything the renderer noticed going less than perfectly is listed, never hidden.
    for warning in result.warnings:
        print(f"Warning       : {warning}")
    # .get returns None when the measurement was not requested, which skips its line.
    sync = payload.get("lipSync")
    if sync:
        if "error" in sync:
            print(f"Lip sync      : not measured - {sync['error']}")
        else:
            print(
                f"Lip sync      : LSE-C {sync['lseC']}  LSE-D {sync['lseD']}  "
                f"offset {sync['offsetFrames']} frames ({sync['method']})"
            )
    jitter = payload.get("jitter")
    if jitter:
        if "error" in jitter:
            print(f"Jitter        : not measured - {jitter['error']}")
        else:
            # Golden rule 8: a missed target is stated plainly, in capitals.
            verdict = "meets" if jitter["meetsTarget"] else "MISSES"
            print(
                f"Jitter        : mean {jitter['meanPct']}%  p95 {jitter['p95Pct']}%  max {jitter['maxPct']}%  "
                f"({verdict} the < {jitter['targetPct']}% target)"
            )
    print("=" * 60)
    return 0


# raise SystemExit(code) ends the process with main()'s return value as the exit status.
if __name__ == "__main__":
    raise SystemExit(main())
