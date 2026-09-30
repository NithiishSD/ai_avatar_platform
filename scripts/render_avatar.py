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
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    parser.add_argument("--no-label", action="store_true", help="do not burn in the AI-generated label")
    parser.add_argument("--metric", action="store_true", help="score lip sync with SyncNet (LSE-C / LSE-D)")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)

    from avatar_store import AvatarConsentError, AvatarError, AvatarNotFound, AvatarStore
    from contracts import AvatarRenderJob
    import render_engine

    job_id = args.job_id or f"cli-{time.strftime('%Y%m%d-%H%M%S')}"

    # Fail on the face before spending time on speech.
    try:
        AvatarStore().require_usable(args.face)
        render_engine.validate_engine(args.engine)
    except (AvatarError, AvatarNotFound, AvatarConsentError, render_engine.RenderError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    from emotion_engine import to_render_emotion_vector
    from voice_engine import VoiceEngineRouter

    mode = args.mode or ("clone" if args.voice else "fast")
    speech = VoiceEngineRouter().synthesize(
        text=args.text,
        mode=mode,
        speaker_wav=args.voice,
        language=args.language,
        emotion=args.emotion,
        return_alignment=True,
        output_filename=f"{render_engine.output_path_for(job_id).stem}.wav",
    )
    if not speech.phoneme_timestamps:
        print(
            "error: forced alignment returned no timestamps, so there is nothing "
            "to drive the mouth. See the alignment warning above.",
            file=sys.stderr,
        )
        return 2

    audio_path = Path(speech.output_path).resolve()
    duration_ms = speech.duration_seconds * 1000.0
    timestamps = [t for t in speech.phoneme_timestamps if t["endMs"] <= duration_ms]
    emotion_vector = to_render_emotion_vector((speech.emotion or {}).get("vector") or {})

    try:
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
            }
        )
        result = render_engine.render_job(job, engine=args.engine, label=not args.no_label)
    except Exception as err:  # noqa: BLE001 - the CLI reports, it does not trace
        print(f"error: {type(err).__name__}: {err}", file=sys.stderr)
        return 1

    payload = {
        "speech": {
            "model": speech.model,
            "mode": speech.mode,
            "audioPath": str(audio_path),
            "durationSeconds": round(speech.duration_seconds, 3),
            "phonemeCount": len(timestamps),
        },
        "render": result.to_dict(),
    }

    if args.metric:
        import lipsync_metric

        try:
            payload["lipSync"] = lipsync_metric.score_video(result.output_path).to_dict()
        except Exception as err:  # noqa: BLE001
            payload["lipSync"] = {"error": f"{type(err).__name__}: {err}"}

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    media = result.media
    print("\n" + "=" * 60)
    print(f"Video         : {result.output_path}")
    print(f"Speech model  : {speech.model} (mode {speech.mode}), {len(timestamps)} phonemes")
    print(f"Render engine : {result.engine}")
    print(f"Frames        : {result.frame_count} at {result.fps} fps, {result.width}x{result.height}")
    print(
        f"Streams       : video {media['videoDuration']} s ({media['videoCodec']}), "
        f"audio {media['audioDuration']} s ({media['audioCodec']}), gap {media['durationGap']} s"
    )
    print(f"Render time   : {result.render_seconds:.2f} s ({result.realtime_factor:.2f}x real time)")
    if result.peak_vram_mb is not None:
        print(f"Peak VRAM     : {result.peak_vram_mb} MiB")
    for warning in result.warnings:
        print(f"Warning       : {warning}")
    sync = payload.get("lipSync")
    if sync:
        if "error" in sync:
            print(f"Lip sync      : not measured - {sync['error']}")
        else:
            print(
                f"Lip sync      : LSE-C {sync['lseC']}  LSE-D {sync['lseD']}  "
                f"offset {sync['offsetFrames']} frames ({sync['method']})"
            )
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
