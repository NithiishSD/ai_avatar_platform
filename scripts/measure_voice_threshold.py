#!/usr/bin/env python3
"""
How similar are voices to each other on the ECAPA-TDNN encoder? (picks the protected-voice threshold, T8.9)

Prints a cosine-similarity matrix for the given recordings, plus the same speaker against itself
(the first and last parts of one long recording), so a threshold can be placed between "a different
voice" and "the same voice or a clone of it".

    PYTHONPATH=backend backend/.conda/bin/python scripts/measure_voice_threshold.py \\
        --long inputs/ljspeech_reference.wav --others outputs/gate3-hindi.wav outputs/gate3-tamil.wav ...
"""

from __future__ import annotations

import argparse
import itertools
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--long", required=True, help="one speaker's long recording; split into first 30 s and the rest")
    parser.add_argument("--others", nargs="+", required=True, help="recordings of other voices")
    args = parser.parse_args()

    import protected_voices
    from quality_auditor import SpeechQualityAuditor

    embed = SpeechQualityAuditor().embed_speaker
    wave, rate = sf.read(args.long, dtype="float32")
    cut = 30 * rate
    named = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, part in (("same-A", wave[:cut]), ("same-B", wave[cut:])):
            path = Path(tmp) / f"{name}.wav"
            sf.write(path, part, rate)
            named[name] = embed(path)
    for other in args.others:
        named[Path(other).name] = embed(other)

    names = list(named)
    print(f"{'same speaker, two parts of one recording':<45} {protected_voices.cosine(named['same-A'], named['same-B']):.3f}")
    different = []
    for a, b in itertools.combinations(names, 2):
        if {a, b} == {"same-A", "same-B"}:
            continue
        sim = protected_voices.cosine(named[a], named[b])
        different.append(sim)
        print(f"{a:<22} vs {b:<20} {sim:.3f}")
    print(f"\ndifferent-voice pairs: n={len(different)} max={max(different):.3f} mean={np.mean(different):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
