#!/usr/bin/env python3
"""
How similar are voices to each other on the ECAPA-TDNN encoder? (picks the protected-voice threshold, T8.9)

Prints a cosine-similarity matrix for the given recordings, plus the same speaker against itself
(the first and last parts of one long recording), so a threshold can be placed between "a different
voice" and "the same voice or a clone of it".

    PYTHONPATH=backend backend/.conda/bin/python scripts/measure_voice_threshold.py \\
        --long inputs/ljspeech_reference.wav --others outputs/gate3-hindi.wav outputs/gate3-tamil.wav ...

Concepts used here:

**A speaker embedding** is a fixed-length vector (192 numbers for ECAPA-TDNN) that an encoder
network produces from a recording. Two recordings of the same voice land close together; two
different voices land further apart. The words spoken should not matter, only who speaks them.

**Cosine similarity** compares two vectors by the angle between them: 1.0 means the same
direction, 0 means unrelated. It ignores vector length, so a loud and a quiet recording of one
voice still compare as similar.

Reading the output: the "same speaker" line is the high end, the "different-voice pairs" summary
is the low end. The threshold belongs in the gap between the largest different-voice score and
the same-speaker score. The chosen value lives in ``backend/protected_voices.py``
(``DEFAULT_THRESHOLD``). Writes nothing; output is printed only.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# parents[1] of scripts/this_file.py is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Lets the backend modules be imported by name even when PYTHONPATH is not set.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


def main() -> int:
    """Embed every recording, print each pair's similarity, and return the exit code."""
    # description=__doc__ reuses the module docstring as --help text; the Raw formatter keeps its
    # line breaks instead of re-wrapping the example command.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--long", required=True, help="one speaker's long recording; split into first 30 s and the rest")
    # nargs="+" collects one or more values into a list.
    parser.add_argument("--others", nargs="+", required=True, help="recordings of other voices")
    args = parser.parse_args()

    # Imported after argument parsing so `--help` answers instantly, without loading torch.
    import protected_voices
    from quality_auditor import SpeechQualityAuditor

    embed = SpeechQualityAuditor().embed_speaker
    wave, rate = sf.read(args.long, dtype="float32")
    # Samples per second times 30 seconds: the split point between the two halves.
    cut = 30 * rate
    named = {}
    # embed_speaker takes a file path, so each half is written to a temporary file that is
    # deleted when the `with` block ends.
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
    # combinations(names, 2) yields every unordered pair once: (a, b) but never also (b, a).
    for a, b in itertools.combinations(names, 2):
        # The same-speaker pair was already printed above; it must not count as "different".
        if {a, b} == {"same-A", "same-B"}:
            continue
        sim = protected_voices.cosine(named[a], named[b])
        different.append(sim)
        print(f"{a:<22} vs {b:<20} {sim:.3f}")
    # The max matters more than the mean: a threshold below the max would flag a different voice.
    print(f"\ndifferent-voice pairs: n={len(different)} max={max(different):.3f} mean={np.mean(different):.3f}")
    return 0


if __name__ == "__main__":
    # sys.exit turns main()'s return value into the process exit code.
    sys.exit(main())
