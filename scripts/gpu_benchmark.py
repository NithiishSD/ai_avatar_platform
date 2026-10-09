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

Concepts used here:

* **VRAM** is the GPU's own memory. The host GPU has about 6 GB, which is why golden rule 5 keeps one
  heavy model resident at a time and why peak usage is measured per step.
* **Allocated vs reserved.** PyTorch keeps a cache of GPU memory blocks so it does not have to ask the
  driver every time. ``max_memory_allocated`` is the peak held by live tensors; ``max_memory_reserved``
  is the peak the cache held, which is what other programs actually lose. Reserved is always at least
  allocated.
* **Warm vs cold.** The first call to a model pays for loading weights from disk. A running server pays
  that once, so the targets are measured after a warm-up call.
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
# Put backend/ on the import path so "import render_engine" works when run from the project root.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# One sentence of roughly 7 s of speech; repeating it builds the ~27 s and ~60 s texts below.
# The trailing space keeps the repeated words apart.
SENTENCE = "This is a sentence for the timing test of the avatar platform, and it is spoken at a natural pace. "


def nvidia_smi() -> Optional[str]:
    """The GPU name, driver and memory in use as nvidia-smi reports them, or None when it is missing."""
    try:
        # nvidia-smi sees all memory on the card, CUDA's own context included, which torch's
        # counters do not. The timeout keeps a hung driver from hanging the whole benchmark.
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        # OSError covers "command not found" when the driver utilities are not installed.
        return None
    return out.stdout.strip() or None


def timed(step: Callable[[], Any], cuda: bool) -> Tuple[Any, float, Optional[int], Optional[int]]:
    """Run ``step``; return (result, seconds, peak allocated MiB, peak reserved MiB) for that step alone."""
    import torch

    if cuda:
        # CUDA synchronize: block until every GPU job queued so far has finished. Without it,
        # leftover work from the previous step would run inside this step's timing.
        torch.cuda.synchronize()
        # The peak-memory counters remember the highest value since the last reset. Resetting
        # here makes the peak belong to this step alone, not to the whole run.
        torch.cuda.reset_peak_memory_stats()
    # perf_counter is a monotonic, high-resolution clock meant for measuring intervals.
    started = time.perf_counter()
    result = step()
    if cuda:
        # GPU work is asynchronous: wait for it to finish, or the clock stops early.
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if not cuda:
        return result, seconds, None, None
    # 2**20 bytes is one MiB; integer division keeps the report in whole MiB.
    return (result, seconds, int(torch.cuda.max_memory_allocated() // 2**20), int(torch.cuda.max_memory_reserved() // 2**20))


def set_marks(on: bool) -> None:
    """Turn the audio and video watermarks on or off for the steps that follow."""
    # One switch for both watermarks, read each time a file is marked (watermark_engine.enabled).
    os.environ["WATERMARK_ENABLED"] = "true" if on else "false"


def main() -> int:
    """Measure every step, write the JSON report and print it; returns the process exit code."""
    # RawDescriptionHelpFormatter keeps the module docstring's line breaks in --help.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-cpu", action="store_true", help="test the script without a GPU (numbers are then NOT GPU numbers)")
    parser.add_argument("--quick", action="store_true", help="short texts, for testing the script only")
    parser.add_argument("--skip-wav2lip", action="store_true")
    args = parser.parse_args()

    # torch is imported after argument parsing, so --help does not wait for torch to load.
    import torch

    cuda = torch.cuda.is_available()
    # Record the software and hardware with the numbers, so a result can be traced to its machine.
    env = {"torch": torch.__version__, "torchCuda": torch.version.cuda, "cudaAvailable": cuda,
           "device": torch.cuda.get_device_name(0) if cuda else None,
           "vramTotalMiB": int(torch.cuda.get_device_properties(0).total_memory // 2**20) if cuda else None,
           "nvidiaSmi": nvidia_smi()}
    print(json.dumps(env, indent=2))
    # Refuse to measure on the CPU unless asked: a CPU timing filed as GPU evidence would be a
    # silent fallback (golden rule 1).
    if not cuda and not args.allow_cpu:
        print("\nPyTorch cannot see a GPU, so nothing was measured. Check, in this order:\n"
              "  1. nvidia-smi                       -> must list the RTX 4050 (if 'command not found', install the driver libraries:\n"
              "                                         sudo apt install libnvidia-compute-595 nvidia-utils-595, then reboot)\n"
              "  2. backend/.conda/bin/python -c \"import torch; print(torch.cuda.is_available())\"   -> must print True\n"
              "  3. run this from a normal terminal, not the editor's sandboxed one\n"
              "  4. unset CUDA_VISIBLE_DEVICES and AVATAR_DEVICE if either is set", file=sys.stderr)
        return 2

    # The heavy project modules load only once a GPU is confirmed.
    import lipsync_metric
    import render_engine
    from contracts import AvatarRenderJob
    from emotion_engine import to_render_emotion_vector
    from voice_engine import VoiceEngineRouter

    # How many times SENTENCE is repeated: 4 gives ~27 s of speech, 8 gives ~60 s of video.
    repeats_30, repeats_60 = (1, 2) if args.quick else (4, 8)
    router = VoiceEngineRouter()
    report: Dict[str, Any] = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "environment": env,
                              "note": "CPU run for testing the script, not GPU evidence" if not cuda else "GPU run"}

    # Warm-up: load Kokoro, the aligner and the watermark models once, as a running server would have.
    set_marks(True)
    router.synthesize("Warm up.", mode="fast", return_alignment=True, output_filename="gpu-warm.wav")

    def speak(text: str, name: str, align: bool):
        """Synthesise ``text`` with Kokoro into ``<name>.wav``; ``align`` also returns phoneme timings."""
        return router.synthesize(text, mode="fast", return_alignment=align, output_filename=f"{name}.wav")

    # 1. Speech speed (N-17)
    text30 = (SENTENCE * repeats_30).strip()
    speech_rows = {}
    for marks in (False, True):
        set_marks(marks)
        # m=marks binds the current value now. A plain closure would read the loop variable
        # later, which is the classic late-binding pitfall with lambdas in loops.
        result, seconds, alloc, reserved = timed(lambda m=marks: speak(text30, f"gpu-speech-{m}", False), cuda)
        speech_rows["marked" if marks else "unmarked"] = {
            "audioSeconds": round(result.duration_seconds, 2), "seconds": round(seconds, 2),
            "secondsPer30s": round(seconds * 30 / result.duration_seconds, 2), "peakAllocMiB": alloc, "peakReservedMiB": reserved}
    # The ** spreads the marked/unmarked rows into the same dictionary as the model and target.
    report["speech_N17"] = {"model": "kokoro", "target": "< 2 s per 30 s of audio", **speech_rows}

    def job_for(speech, job_id: str, quality: str = "PREVIEW") -> AvatarRenderJob:
        """Build the frozen render contract (``AvatarRenderJob``) from one synthesis result."""
        # Drop any timestamp that ends after the audio does; the contract's validator rejects
        # those ("phoneme timestamp exceeds durationSeconds").
        stamps = [t for t in speech.phoneme_timestamps if t["endMs"] <= speech.duration_seconds * 1000]
        return AvatarRenderJob.model_validate({
            "jobId": job_id, "avatarId": "demo", "audioUrl": Path(speech.output_path).resolve().as_uri(), "sampleRate": speech.sample_rate,
            "durationSeconds": speech.duration_seconds, "phonemeTimestamps": stamps,
            "emotionVector": to_render_emotion_vector((speech.emotion or {}).get("vector") or {}), "renderQuality": quality, "targetFps": 25})

    # 2. 30 s video render (N-20)
    # Synthesise once, then render the same speech twice, so only the render is timed.
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
        """Synthesis, alignment and render together, the way one API request runs them."""
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
        # SyncNet scores the rendered video itself, so the lip-sync number comes from the output file.
        score = lipsync_metric.score_video(result.output_path).to_dict()
        report["wav2lip_N10"] = {"target": "peak VRAM < 6144 MiB", "renderSeconds": round(result.render_seconds, 2), "peakAllocMiB": alloc,
                                 "peakReservedMiB": reserved, "rendererPeakVramMiB": result.peak_vram_mb, "nvidiaSmiAfter": nvidia_smi(),
                                 "video": result.output_path, "offsetFrames": score["offsetFrames"], "lseC": score["lseC"],
                                 "secondsWithinOneFrame": score["secondsWithinOneFrame"]}

    # A timestamped name, so a second run never overwrites the first.
    out = PROJECT_ROOT / "outputs" / "benchmarks" / f"gpu-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("\n===== PASTE EVERYTHING BELOW THIS LINE =====")
    print(json.dumps(report, indent=2))
    print(f"(saved to {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
