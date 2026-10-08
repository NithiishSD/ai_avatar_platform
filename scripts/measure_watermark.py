#!/usr/bin/env python3
"""
How good is the audio watermark? Three questions, each answered with a number and its method.

1. Does it hurt the speech?  SQUIM MOS / PESQ / STOI estimates on the same clips before and after
   marking (the change is the number that matters), plus how far below the speech the mark sits.
2. Does it survive handling?  The fraction of marked clips still detected after each transform:
   a 16-bit WAV, 8 kHz resampling, AAC and MP3 and Opus re-encoding (what an MP4 soundtrack goes through),
   added noise, trimming, quieter volume, low-pass, a speed change.
3. Does it cry wolf?  The fraction of UNMARKED audio detected (human speech, TTS speech, noise, silence,
   tones) and of clips marked with a different key.

Clips: 8 human segments (the LJSpeech reference) and 8 Kokoro sentences synthesised with the watermark
switched off, so every original is genuinely unmarked.

    PYTHONPATH=backend backend/.conda/bin/python scripts/measure_watermark.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "She sells sea shells by the sea shore every single summer morning.",
    "Our quarterly revenue grew by thirty seven percent across all regions.",
    "Please confirm whether the shipment arrived before the eleventh of March.",
    "Thick fog rolled through the valley while the church bells rang loudly.",
    "The committee will review the proposal and publish its decision next week.",
    "A gentle breeze moved the curtains as the evening light faded slowly away.",
    "Remember to bring your passport and the signed forms to the front desk.",
]


def main() -> int:
    os.environ["WATERMARK_ENABLED"] = "false"  # the synthesis below must produce UNMARKED originals
    import numpy as np
    import soundfile as sf

    import watermark_engine as w
    from quality_auditor import SpeechQualityAuditor
    from voice_engine import VoiceEngineRouter

    marker = w.AudioWatermarker()
    auditor = SpeechQualityAuditor()
    reference = PROJECT_ROOT / "inputs" / "ljspeech_reference.wav"

    # ---- the clips
    human, rate = sf.read(str(reference), dtype="float32")
    clips: Dict[str, np.ndarray] = {f"human-{i}": human[i * 5 * rate : i * 5 * rate + 4 * rate] for i in range(8)}
    router = VoiceEngineRouter(device="cpu")
    for i, sentence in enumerate(SENTENCES):
        result = router.synthesize(sentence, mode="fast", output_filename=f"benchmark/wm-src-{i}.wav")
        audio, sr = sf.read(result.output_path, dtype="float32")
        assert sr == rate, "all clips are used at one rate"
        clips[f"tts-{i}"] = audio

    marked = {name: marker.embed(clip, rate) for name, clip in clips.items()}

    # ---- 1. quality
    work = Path(tempfile.mkdtemp(prefix="wm-measure-"))
    quality: List[Dict[str, Any]] = []
    ratios: List[float] = []
    for name, clip in clips.items():
        a, b = work / f"{name}-orig.wav", work / f"{name}-marked.wav"
        sf.write(a, clip, rate, subtype="PCM_16")
        sf.write(b, marked[name], rate, subtype="PCM_16")
        ra, rb = auditor.audit(a, reference_path=reference), auditor.audit(b, reference_path=reference)
        ratios.append(10 * np.log10(np.sum(clip**2) / np.sum((marked[name] - clip) ** 2)))
        quality.append({
            "clip": name,
            **{f"{k}Delta": round((getattr(rb, k) or 0) - (getattr(ra, k) or 0), 3) for k in ("mos", "pesq", "stoi")},
            "mosBefore": round(ra.mos or 0, 3), "mosAfter": round(rb.mos or 0, 3),
        })
    method = auditor.audit(work / "human-0-orig.wav", reference_path=reference).method

    def mean(key: str, subset: str | None = None) -> float:
        rows = [q[key] for q in quality if subset is None or q["clip"].startswith(subset)]
        return round(float(np.mean(rows)), 3)

    # ---- 2. robustness
    def ffmpeg(codec_args: List[str], suffix: str) -> Callable[[np.ndarray], np.ndarray]:
        def run(x: np.ndarray) -> np.ndarray:
            src, out = work / "t-in.wav", work / f"t-out{suffix}"
            sf.write(src, x, rate, subtype="PCM_16")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *codec_args, str(out)], check=True)
            decoded = work / "t-dec.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-ar", str(rate), "-ac", "1", str(decoded)], check=True)
            return sf.read(str(decoded), dtype="float32")[0]
        return run

    rng = np.random.default_rng(7)

    def noise(snr_db: float) -> Callable[[np.ndarray], np.ndarray]:
        def run(x: np.ndarray) -> np.ndarray:
            power = float(np.mean(x**2))
            return (x + rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(power / 10 ** (snr_db / 10))).astype(np.float32)
        return run

    def lowpass(x: np.ndarray) -> np.ndarray:
        from scipy.signal import butter, sosfilt

        return np.asarray(sosfilt(butter(8, 4000, btype="low", fs=rate, output="sos"), x), dtype=np.float32)

    transforms: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "none (float32)": lambda x: x,
        "16-bit WAV": lambda x: np.round(x * 32767).astype(np.float32) / 32767,
        "resample 24k->8k->24k": lambda x: w._resample(w._resample(x, rate, 8000), 8000, rate),
        "AAC 128 kbps (MP4 soundtrack)": ffmpeg(["-c:a", "aac", "-b:a", "128k"], ".m4a"),
        "AAC 64 kbps": ffmpeg(["-c:a", "aac", "-b:a", "64k"], ".m4a"),
        "MP3 128 kbps": ffmpeg(["-c:a", "libmp3lame", "-b:a", "128k"], ".mp3"),
        "Opus 32 kbps": ffmpeg(["-c:a", "libopus", "-b:a", "32k"], ".opus"),
        "white noise, 30 dB SNR": noise(30),
        "white noise, 20 dB SNR": noise(20),
        "white noise, 10 dB SNR": noise(10),
        "trim first 0.7 s": lambda x: x[int(0.7 * rate):],
        "volume x0.3": lambda x: (x * 0.3).astype(np.float32),
        "low-pass 4 kHz": lowpass,
        "speed x1.10 (pitch shifts too)": lambda x: w._resample(x, int(rate * 1.10), rate),
    }
    robustness: Dict[str, Dict[str, Any]] = {}
    for label, transform in transforms.items():
        flags = []
        for name in clips:
            flags.append(marker.detect(transform(marked[name]), rate).detected)
        robustness[label] = {"detected": sum(flags), "of": len(flags), "rate": round(sum(flags) / len(flags), 3)}
        print(f"  {label:<34} {sum(flags):>2}/{len(flags)}", flush=True)

    # ---- 3. false positives
    nulls: Dict[str, List[bool]] = {
        "human speech (unmarked)": [marker.detect(clips[n], rate).detected for n in clips if n.startswith("human")],
        "TTS speech (unmarked)": [marker.detect(clips[n], rate).detected for n in clips if n.startswith("tts")],
        "white noise": [marker.detect(rng.standard_normal(rate * 3).astype(np.float32) * 0.1, rate).detected for _ in range(8)],
        "pure tones": [marker.detect((0.3 * np.sin(2 * np.pi * f * np.arange(rate * 3) / rate)).astype(np.float32), rate).detected for f in (110, 220, 440, 880, 1760, 3520)],
        "silence": [marker.detect(np.zeros(rate * 3, dtype=np.float32), rate).detected],
    }
    original_key = w.signing_key
    w.signing_key = lambda: b"a-completely-different-key"  # type: ignore[assignment]
    try:
        nulls["our clips checked with another key"] = [marker.detect(marked[n], rate).detected for n in clips]
    finally:
        w.signing_key = original_key  # type: ignore[assignment]
    false_positives = {k: {"detected": sum(v), "of": len(v)} for k, v in nulls.items()}
    for k, v in false_positives.items():
        print(f"  false positives: {k:<38} {v['detected']}/{v['of']}", flush=True)

    report = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"), "clips": len(clips), "rateHz": rate, "detector": w.METHOD,
        "decisionRule": f"probability >= {w.MIN_PROBABILITY} and >= {w.MIN_BITS_MATCHING}/16 bits equal the platform tag",
        "quality": {
            "method": method, "signalToMarkDbMean": round(float(np.mean(ratios)), 1), "signalToMarkDbMin": round(float(np.min(ratios)), 1),
            "mosDeltaMean": mean("mosDelta"), "mosDeltaMeanHuman": mean("mosDelta", "human"), "mosDeltaMeanTts": mean("mosDelta", "tts"),
            "pesqDeltaMean": mean("pesqDelta"), "stoiDeltaMean": mean("stoiDelta"), "perClip": quality,
        },
        "robustness": robustness, "falsePositives": false_positives,
    }
    out = PROJECT_ROOT / "outputs" / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"watermark-{time.strftime('%Y%m%d-%H%M%S')}.json"
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("robustness", "falsePositives")} | {"quality": {k: v for k, v in report["quality"].items() if k != "perClip"}}, indent=2))
    print(f"written: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
