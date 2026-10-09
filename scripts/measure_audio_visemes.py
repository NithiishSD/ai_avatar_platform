#!/usr/bin/env python3
"""
How often do mouth shapes estimated from sound alone agree with real phoneme timing? (I-02)

Speaks sentences with Kokoro and forced alignment (the measured truth), then estimates the mouth shape
of every 40 ms window from the audio alone (``audio_visemes``) and compares families: closed, teeth
(s/sh/f/th), rounded (o/u), open (a), wide (e/i). Windows whose true viseme is a consonant that sound
cannot reveal (p, d, k, n, r) are skipped. Chance agreement over five families is about 20 %.

    PYTHONPATH=backend backend/.conda/bin/python scripts/measure_audio_visemes.py

Concepts used here:

* A **viseme** is the visible mouth shape for a sound, the way a phoneme is the audible unit. Several
  phonemes share one viseme ("p", "b" and "m" all close the lips).
* **Forced alignment** takes audio plus the words it contains and finds when each phoneme starts and
  ends. Because it knows the words, it is the reference here; ``audio_visemes`` has only the sound.

It writes nothing except the synthesised WAV files the voice router saves; the result is printed:
the agreement percentage and the ten most common (truth, guess) pairs, a small confusion table.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The backend modules import each other by bare name ("import audio_visemes"), so the backend
# folder has to be on the import path when this script is run from the project root.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Maps each viseme the aligner can emit to one of the five families compared here. Visemes missing
# from this table (PP, DD, kk, nn, RR, L) are consonants whose mouth shape the sound does not reveal.
FAMILY = {"viseme_sil": "closed", "viseme_SS": "teeth", "viseme_CH": "teeth", "viseme_FF": "teeth", "viseme_TH": "teeth",
          "viseme_O": "rounded", "viseme_U": "rounded", "viseme_aa": "open", "viseme_E": "wide", "viseme_I": "wide"}
# Between them the sentences exercise every family: open vowels, a hiss-heavy tongue twister,
# front vowels and rounded "oo" sounds.
SENTENCES = [
    "Hello, how are you today? I hope you are feeling well.",
    "She sells sea shells by the sea shore every summer.",
    "Our team will open the new office in October.",
    "Please look at the blue moon over the old pool.",
]


def main() -> int:
    """Synthesise each sentence, score every window, print the agreement; returns the exit code."""
    # Imported inside main(): the speech stack loads only when the measurement actually runs.
    import soundfile as sf

    import audio_visemes
    from voice_engine import VoiceEngineRouter

    router = VoiceEngineRouter()
    agree = total = 0
    # Counter counts hashable keys; here each key is a (true family, guessed family) pair.
    confusion: Counter = Counter()
    for k, text in enumerate(SENTENCES):
        # mode="fast" is Kokoro; return_alignment=True asks for the phoneme timestamps (the truth).
        result = router.synthesize(text, mode="fast", return_alignment=True, output_filename=f"visemes-{k}.wav")
        audio, rate = sf.read(result.output_path, dtype="float32")
        # One hop is one 40 ms window in samples, the same size audio_visemes classifies live.
        hop = int(rate * audio_visemes.WINDOW_MS / 1000)
        stamps = result.phoneme_timestamps
        for start in range(0, len(audio) - hop, hop):
            # Judge the window by the phoneme under its centre, so a boundary near an edge
            # does not decide which phoneme the window "is".
            middle_ms = (start + hop / 2) * 1000 / rate
            # next(generator, default) returns the first match; no stamp covering the moment
            # means a pause, which is the closed (silent) mouth.
            truth = next((s["viseme"] for s in stamps or [] if s["startMs"] <= middle_ms < s["endMs"]), "viseme_sil")
            if truth not in FAMILY:
                continue  # p, d, k, n, r: not visible in the sound alone
            # A classifier output that is not in the table is counted as the "open" family.
            guess = FAMILY.get(audio_visemes.classify(audio[start:start + hop], rate), "open")
            total += 1
            # A bool adds as 0 or 1, so this counts matches.
            agree += guess == FAMILY[truth]
            confusion[(FAMILY[truth], guess)] += 1
    # max(1, total) guards the division when no window was scorable.
    print(f"agreement {agree}/{total} = {100 * agree / max(1, total):.1f}% (chance ~20%)")
    for (truth, guess), n in sorted(confusion.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  truth {truth:8s} guessed {guess:8s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
