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

What it writes: ``outputs/benchmarks/watermark-<YYYYmmdd-HHMMSS>.json`` with every number above and
the per-clip quality rows; a short summary and one line per transform go to stdout. It needs the
AudioSeal weights, the SQUIM models, Kokoro and ``ffmpeg`` on PATH; it loads real models, so it is a
measurement script, not a unit test.

Terms used below, explained once:

**MOS / PESQ / STOI.** MOS is a 1-5 "how natural does this sound" score; PESQ is perceived quality
against a reference; STOI is intelligibility (0-1). SQUIM predicts all three from the waveform, so no
human listening panel is needed. Here only the change before vs after marking matters.

**Signal-to-mark ratio (dB).** ``10*log10(speech energy / mark energy)``. Each +10 dB is the mark
being ten times weaker in energy than the speech; a high value means the mark is buried well below it.

**SNR (signal-to-noise ratio).** The same decibel ratio for added noise: 30 dB is faint hiss, 10 dB
is loud noise.
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

# The repository root (scripts/ is one level below it).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Eight varied sentences for the TTS half of the clips: different sounds and
# lengths, so a result is not an accident of one phrase.
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
    """Build the clips, run the three measurements and write the JSON report."""
    # Set before the backend modules are imported, so the synthesis engine reads
    # the switch when it starts and produces unmarked audio.
    os.environ["WATERMARK_ENABLED"] = "false"  # the synthesis below must produce UNMARKED originals
    # Heavy imports live inside main() so `--help`-style inspection and import
    # of this file stay cheap; they also must come after the env var above.
    import numpy as np
    import soundfile as sf

    import watermark_engine as w
    from quality_auditor import SpeechQualityAuditor
    from voice_engine import VoiceEngineRouter

    marker = w.AudioWatermarker()
    auditor = SpeechQualityAuditor()
    # SQUIM's subjective MOS needs a clean, non-matching reference clip; the
    # LJSpeech sample is the consented reference used throughout the project.
    reference = PROJECT_ROOT / "inputs" / "ljspeech_reference.wav"

    # ---- the clips
    human, rate = sf.read(str(reference), dtype="float32")
    # Eight 4-second windows, taken every 5 seconds so they never overlap.
    clips: Dict[str, np.ndarray] = {f"human-{i}": human[i * 5 * rate : i * 5 * rate + 4 * rate] for i in range(8)}
    router = VoiceEngineRouter(device="cpu")
    for i, sentence in enumerate(SENTENCES):
        result = router.synthesize(sentence, mode="fast", output_filename=f"benchmark/wm-src-{i}.wav")
        audio, sr = sf.read(result.output_path, dtype="float32")
        # One sample rate for every clip, so the same transforms and the same
        # detector call apply to all of them.
        assert sr == rate, "all clips are used at one rate"
        clips[f"tts-{i}"] = audio

    # The marked copy of every clip; the originals stay untouched for comparison.
    marked = {name: marker.embed(clip, rate) for name, clip in clips.items()}

    # ---- 1. quality
    # mkdtemp makes a fresh private directory; every intermediate file goes there.
    work = Path(tempfile.mkdtemp(prefix="wm-measure-"))
    quality: List[Dict[str, Any]] = []
    ratios: List[float] = []
    for name, clip in clips.items():
        a, b = work / f"{name}-orig.wav", work / f"{name}-marked.wav"
        # Both written as 16-bit PCM, so the before and after go through the
        # same quantisation and only the mark differs.
        sf.write(a, clip, rate, subtype="PCM_16")
        sf.write(b, marked[name], rate, subtype="PCM_16")
        ra, rb = auditor.audit(a, reference_path=reference), auditor.audit(b, reference_path=reference)
        # marked - clip is the mark alone; its energy against the speech's
        # energy is the signal-to-mark ratio in dB.
        ratios.append(10 * np.log10(np.sum(clip**2) / np.sum((marked[name] - clip) ** 2)))
        quality.append({
            "clip": name,
            # A metric the auditor could not compute comes back as None; `or 0`
            # keeps the subtraction from failing.
            **{f"{k}Delta": round((getattr(rb, k) or 0) - (getattr(ra, k) or 0), 3) for k in ("mos", "pesq", "stoi")},
            "mosBefore": round(ra.mos or 0, 3), "mosAfter": round(rb.mos or 0, 3),
        })
    # The auditor's method string goes into the report (golden rule 2: every
    # number records its method).
    method = auditor.audit(work / "human-0-orig.wav", reference_path=reference).method

    def mean(key: str, subset: str | None = None) -> float:
        """Mean of one quality column, optionally only over clips whose name starts with ``subset``."""
        rows = [q[key] for q in quality if subset is None or q["clip"].startswith(subset)]
        return round(float(np.mean(rows)), 3)

    # ---- 2. robustness
    # Each transform below is a function audio -> audio. The factories
    # (ffmpeg, noise) return such a function with their settings baked in: a
    # closure, so every entry in the table has the same call shape.
    def ffmpeg(codec_args: List[str], suffix: str) -> Callable[[np.ndarray], np.ndarray]:
        """Return a transform that encodes with ``codec_args`` and decodes back to mono WAV at ``rate``."""
        def run(x: np.ndarray) -> np.ndarray:
            """Round-trip ``x`` through the lossy codec, as a real upload would."""
            src, out = work / "t-in.wav", work / f"t-out{suffix}"
            sf.write(src, x, rate, subtype="PCM_16")
            # -y overwrites the scratch file; -loglevel error keeps the output to real failures.
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *codec_args, str(out)], check=True)
            decoded = work / "t-dec.wav"
            # Decode back at the original rate and to one channel, so the
            # detector sees the same shape it was given before encoding.
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-ar", str(rate), "-ac", "1", str(decoded)], check=True)
            return sf.read(str(decoded), dtype="float32")[0]
        return run

    # A fixed seed makes the noise, and therefore the result, reproducible.
    rng = np.random.default_rng(7)

    def noise(snr_db: float) -> Callable[[np.ndarray], np.ndarray]:
        """Return a transform that adds white noise at ``snr_db`` below the clip's own power."""
        def run(x: np.ndarray) -> np.ndarray:
            """Add Gaussian noise scaled so speech power / noise power equals the target SNR."""
            power = float(np.mean(x**2))
            # SNR in dB -> power ratio is 10**(dB/10); noise std is the square
            # root of the noise power.
            return (x + rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(power / 10 ** (snr_db / 10))).astype(np.float32)
        return run

    def lowpass(x: np.ndarray) -> np.ndarray:
        """Cut everything above 4 kHz with an 8th-order Butterworth filter (a telephone-like band)."""
        from scipy.signal import butter, sosfilt

        # output="sos" (second-order sections) is the numerically stable form
        # for a high-order filter.
        return np.asarray(sosfilt(butter(8, 4000, btype="low", fs=rate, output="sos"), x), dtype=np.float32)

    transforms: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        # The baseline: if this is not 100%, nothing below means anything.
        "none (float32)": lambda x: x,
        # Quantise to 16-bit integers and back, as saving a normal WAV does.
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
        # Treating the samples as if recorded at 1.10x the rate, then resampling,
        # plays them 10% faster and higher.
        "speed x1.10 (pitch shifts too)": lambda x: w._resample(x, int(rate * 1.10), rate),
    }
    robustness: Dict[str, Dict[str, Any]] = {}
    for label, transform in transforms.items():
        flags = []
        for name in clips:
            flags.append(marker.detect(transform(marked[name]), rate).detected)
        robustness[label] = {"detected": sum(flags), "of": len(flags), "rate": round(sum(flags) / len(flags), 3)}
        # flush=True prints each line as it is ready; the codec runs are slow.
        print(f"  {label:<34} {sum(flags):>2}/{len(flags)}", flush=True)

    # ---- 3. false positives
    # Audio that was never marked by us. Every detection here is a false alarm.
    nulls: Dict[str, List[bool]] = {
        "human speech (unmarked)": [marker.detect(clips[n], rate).detected for n in clips if n.startswith("human")],
        "TTS speech (unmarked)": [marker.detect(clips[n], rate).detected for n in clips if n.startswith("tts")],
        "white noise": [marker.detect(rng.standard_normal(rate * 3).astype(np.float32) * 0.1, rate).detected for _ in range(8)],
        "pure tones": [marker.detect((0.3 * np.sin(2 * np.pi * f * np.arange(rate * 3) / rate)).astype(np.float32), rate).detected for f in (110, 220, 440, 880, 1760, 3520)],
        "silence": [marker.detect(np.zeros(rate * 3, dtype=np.float32), rate).detected],
    }
    # The tag is derived from the signing key. Swapping the module's key
    # function makes the detector look for a different tag, which tests that our
    # marks are not confused with another deployment's.
    original_key = w.signing_key
    w.signing_key = lambda: b"a-completely-different-key"  # type: ignore[assignment]
    try:
        nulls["our clips checked with another key"] = [marker.detect(marked[n], rate).detected for n in clips]
    finally:
        # finally restores the real key even if detection raises.
        w.signing_key = original_key  # type: ignore[assignment]
    false_positives = {k: {"detected": sum(v), "of": len(v)} for k, v in nulls.items()}
    for k, v in false_positives.items():
        print(f"  false positives: {k:<38} {v['detected']}/{v['of']}", flush=True)

    report = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"), "clips": len(clips), "rateHz": rate, "detector": w.METHOD,
        # The rule is written out from the engine's own constants, so the report
        # cannot disagree with the code that made the decision.
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
    # Print a short version: the long tables were already printed line by line.
    # `|` merges two dicts (Python 3.9+); the right side wins on shared keys.
    print(json.dumps({k: v for k, v in report.items() if k not in ("robustness", "falsePositives")} | {"quality": {k: v for k, v in report["quality"].items() if k != "perClip"}}, indent=2))
    print(f"written: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
