#!/usr/bin/env python3
"""
How often do mouth shapes estimated from sound alone agree with real phoneme timing? (I-02)

Speaks sentences with Kokoro and forced alignment (the measured truth), then estimates the mouth shape
of every 40 ms window from the audio alone (``audio_visemes``) and compares families: closed, teeth
(s/sh/f/th), rounded (o/u), open (a), wide (e/i). Windows whose true viseme is a consonant that sound
cannot reveal (p, d, k, n, r) are skipped. Chance agreement over five families is about 20 %.

    PYTHONPATH=backend backend/.conda/bin/python scripts/measure_audio_visemes.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

FAMILY = {"viseme_sil": "closed", "viseme_SS": "teeth", "viseme_CH": "teeth", "viseme_FF": "teeth", "viseme_TH": "teeth",
          "viseme_O": "rounded", "viseme_U": "rounded", "viseme_aa": "open", "viseme_E": "wide", "viseme_I": "wide"}
SENTENCES = [
    "Hello, how are you today? I hope you are feeling well.",
    "She sells sea shells by the sea shore every summer.",
    "Our team will open the new office in October.",
    "Please look at the blue moon over the old pool.",
]


def main() -> int:
    import soundfile as sf

    import audio_visemes
    from voice_engine import VoiceEngineRouter

    router = VoiceEngineRouter()
    agree = total = 0
    confusion: Counter = Counter()
    for k, text in enumerate(SENTENCES):
        result = router.synthesize(text, mode="fast", return_alignment=True, output_filename=f"visemes-{k}.wav")
        audio, rate = sf.read(result.output_path, dtype="float32")
        hop = int(rate * audio_visemes.WINDOW_MS / 1000)
        stamps = result.phoneme_timestamps
        for start in range(0, len(audio) - hop, hop):
            middle_ms = (start + hop / 2) * 1000 / rate
            truth = next((s["viseme"] for s in stamps or [] if s["startMs"] <= middle_ms < s["endMs"]), "viseme_sil")
            if truth not in FAMILY:
                continue  # p, d, k, n, r: not visible in the sound alone
            guess = FAMILY.get(audio_visemes.classify(audio[start:start + hop], rate), "open")
            total += 1
            agree += guess == FAMILY[truth]
            confusion[(FAMILY[truth], guess)] += 1
    print(f"agreement {agree}/{total} = {100 * agree / max(1, total):.1f}% (chance ~20%)")
    for (truth, guess), n in sorted(confusion.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  truth {truth:8s} guessed {guess:8s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
