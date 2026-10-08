#!/usr/bin/env python
"""
Every GPU measurement the project still owes, in one run on the host (M-04, N-06, N-10, N-17, N-20).

The development sandbox cannot see the GPU, so these numbers can only come from the owner's machine.
Run it there, from the project root, after `nvidia-smi` works:

    PYTHONPATH=backend backend/.conda/bin/python scripts/gpu_benchmark.py

It measures, each with its method written into the report:

1. **Speech speed (N-17):** Kokoro, model already loaded, on ~27 s of speech, with and without the
   audio watermark. Target: under 2 s per 30 s of audio.
2. **30 s video render (N-20):** the blendshape renderer on ~27 s of speech, marks off and on.
   Target: under 5 s per 30 s of video.
3. **60 s of video end to end (N-06):** synthesis + alignment + render with the marks on, models warm
   (as a running server would be). Target: under 30 s.
4. **Peak VRAM (N-10):** ``torch.cuda.max_memory_allocated`` for each step, including a Wav2Lip
   render. Target: under 6 GB. (CUDA's own context, ~0.3-0.5 GB, is not counted by torch; the
   report also gives ``max_memory_reserved`` and nvidia-smi's reading for that reason.)
5. **Wav2Lip lip sync on the GPU:** SyncNet offset, LSE-C and the share of seconds within one frame.

Writes ``outputs/benchmarks/gpu-<date>.json`` and prints a short summary to paste back. Exits 2,
without measuring anything, if PyTorch cannot see a GPU (so a CPU run can never pass as a GPU one);
``--allow-cpu --quick`` exists only to test the script itself on a machine without one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

SENTENCE = "This is a sentence for the timing test of the avatar platform, and it is spoken at a natural pace. "


def nvidia_smi() -> Optional[str]:
    """The GPU name, driver and memory in use as nvidia-smi reports them, or None when it is missing."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def timed(step: Callable[[], Any], cuda: bool) -> Tuple[Any, float, Optional[int], Optional[int]]:
    """Run ``step``; return (result, seconds, peak allocated MiB, peak reserved MiB) for that step alone."""
    import torch

    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = step()
    if cuda:
        # GPU work is asynchronous: wait for it to finish, or the clock stops early.
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if not cuda:
        return result, seconds, None, None
    return (result, seconds, int(torch.cuda.max_memory_allocated() // 2**20), int(torch.cuda.max_memory_reserved() // 2**20))


def set_marks(on: bool) -> None:
    # One switch for both watermarks, read each time a file is marked (watermark_engine.enabled).
    os.environ["WATERMARK_ENABLED"] = "true" if on else "false"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-cpu", action="store_true", help="test the script without a GPU (numbers are then NOT GPU numbers)")
    parser.add_argument("--quick", action="store_true", help="short texts, for testing the script only")
    parser.add_argument("--skip-wav2lip", action="store_true")
    args = parser.parse_args()

    import torch

    cuda = torch.cuda.is_available()
    env = {"torch": torch.__version__, "torchCuda": torch.version.cuda, "cudaAvailable": cuda,
           "device": torch.cuda.get_device_name(0) if cuda else None,
           "vramTotalMiB": int(torch.cuda.get_device_properties(0).total_memory // 2**20) if cuda else None,
           "nvidiaSmi": nvidia_smi()}
    print(json.dumps(env, indent=2))
    if not cuda and not args.allow_cpu:
        print("\nPyTorch cannot see a GPU, so nothing was measured. Check, in this order:\n"
              "  1. nvidia-smi                       -> must list the RTX 4050 (if 'command not found', install the driver libraries:\n"
              "                                         sudo apt install libnvidia-compute-595 nvidia-utils-595, then reboot)\n"
              "  2. backend/.conda/bin/python -c \"import torch; print(torch.cuda.is_available())\"   -> must print True\n"
              "  3. run this from a normal terminal, not the editor's sandboxed one\n"
              "  4. unset CUDA_VISIBLE_DEVICES and AVATAR_DEVICE if either is set", file=sys.stderr)
        return 2

    import lipsync_metric
    import render_engine
    from contracts import AvatarRenderJob
    from emotion_engine import to_render_emotion_vector
    from voice_engine import VoiceEngineRouter

    repeats_30, repeats_60 = (1, 2) if args.quick else (4, 8)
    router = VoiceEngineRouter()
    report: Dict[str, Any] = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "environment": env,
                              "note": "CPU run for testing the script, not GPU evidence" if not cuda else "GPU run"}

    # Warm-up: load Kokoro, the aligner and the watermark models once, as a running server would have.
    set_marks(True)
    router.synthesize("Warm up.", mode="fast", return_alignment=True, output_filename="gpu-warm.wav")

    def speak(text: str, name: str, align: bool):
        return router.synthesize(text, mode="fast", return_alignment=align, output_filename=f"{name}.wav")

    # 1. Speech speed (N-17)
    text30 = (SENTENCE * repeats_30).strip()
    speech_rows = {}
    for marks in (False, True):
        set_marks(marks)
        result, seconds, alloc, reserved = timed(lambda m=marks: speak(text30, f"gpu-speech-{m}", False), cuda)
        speech_rows["marked" if marks else "unmarked"] = {
            "audioSeconds": round(result.duration_seconds, 2), "seconds": round(seconds, 2),
            "secondsPer30s": round(seconds * 30 / result.duration_seconds, 2), "peakAllocMiB": alloc, "peakReservedMiB": reserved}
    report["speech_N17"] = {"model": "kokoro", "target": "< 2 s per 30 s of audio", **speech_rows}

    def job_for(speech, job_id: str, quality: str = "PREVIEW") -> AvatarRenderJob:
        stamps = [t for t in speech.phoneme_timestamps if t["endMs"] <= speech.duration_seconds * 1000]
        return AvatarRenderJob.model_validate({
            "jobId": job_id, "avatarId": "demo", "audioUrl": Path(speech.output_path).resolve().as_uri(), "sampleRate": speech.sample_rate,
            "durationSeconds": speech.duration_seconds, "phonemeTimestamps": stamps,
            "emotionVector": to_render_emotion_vector((speech.emotion or {}).get("vector") or {}), "renderQuality": quality, "targetFps": 25})

    # 2. 30 s video render (N-20)
    set_marks(True)
    speech30 = speak(text30, "gpu-render30", True)
    render_rows = {}
    for marks in (False, True):
        set_marks(marks)
        result, seconds, alloc, reserved = timed(lambda m=marks: render_engine.render_job(job_for(speech30, f"gpu-r30-{m}"), engine="blendshape"), cuda)
        render_rows["marked" if marks else "unmarked"] = {
            "videoSeconds": round(result.duration_seconds, 2), "renderSeconds": round(result.render_seconds, 2),
            "secondsPer30s": round(result.render_seconds * 30 / result.duration_seconds, 2), "peakAllocMiB": alloc, "peakReservedMiB": reserved}
    report["render30_N20"] = {"engine": "blendshape", "target": "< 5 s per 30 s of video", **render_rows}

    # 3. 60 s of video end to end (N-06), marks on, models warm
    set_marks(True)
    text60 = (SENTENCE * repeats_60).strip()

    def end_to_end():
        speech = speak(text60, "gpu-e2e60", True)
        return render_engine.render_job(job_for(speech, "gpu-e2e60"), engine="blendshape")

    result, seconds, alloc, reserved = timed(end_to_end, cuda)
    report["video60_N06"] = {"target": "< 30 s for 60 s of video", "videoSeconds": round(result.duration_seconds, 2), "wallSeconds": round(seconds, 2),
                             "secondsPer60s": round(seconds * 60 / result.duration_seconds, 2), "watermarks": "on",
                             "peakAllocMiB": alloc, "peakReservedMiB": reserved, "nvidiaSmiAfter": nvidia_smi()}

    # 4 + 5. Wav2Lip on the GPU: peak VRAM (N-10) and lip sync
    if not args.skip_wav2lip:
        set_marks(True)
        speech = speak("The quick brown fox jumps over the lazy dog.", "gpu-w2l", True)
        result, seconds, alloc, reserved = timed(lambda: render_engine.render_job(job_for(speech, "gpu-w2l"), engine="wav2lip"), cuda)
        score = lipsync_metric.score_video(result.output_path).to_dict()
        report["wav2lip_N10"] = {"target": "peak VRAM < 6144 MiB", "renderSeconds": round(result.render_seconds, 2), "peakAllocMiB": alloc,
                                 "peakReservedMiB": reserved, "rendererPeakVramMiB": result.peak_vram_mb, "nvidiaSmiAfter": nvidia_smi(),
                                 "video": result.output_path, "offsetFrames": score["offsetFrames"], "lseC": score["lseC"],
                                 "secondsWithinOneFrame": score["secondsWithinOneFrame"]}

    out = PROJECT_ROOT / "outputs" / "benchmarks" / f"gpu-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("\n===== PASTE EVERYTHING BELOW THIS LINE =====")
    print(json.dumps(report, indent=2))
    print(f"(saved to {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
