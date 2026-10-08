#!/usr/bin/env python
"""
Build a voice-cloning reference clip in ``inputs/``.

Two modes, because the reference serves two separable jobs.

``--smoke`` concatenates this project's own Kokoro output into a 30-60s clip
so the XTTS-v2 path can be wired and proven end to end without waiting on a
recording. It writes a provenance sidecar marking the clip SYNTHETIC, so the
benchmark measures it but never counts it toward the >85% threshold. Synthetic
speech is an artificially easy cloning target -- no room reverb, no mic
coloration, no breath noise -- so a score against it would flatter the system
and say nothing about cloning a real voice.

``--human FILE`` registers a real recording and records its consent basis,
which is what makes a published similarity number defensible. You do not need
your own voice for this: an openly licensed corpus carries consent in its
licence.

    # unblock the pipeline now
    PYTHONPATH=backend backend/.conda/bin/python scripts/make_reference.py --smoke

    # register a real reference later
    PYTHONPATH=backend backend/.conda/bin/python scripts/make_reference.py \
        --human ~/Downloads/LJ001-0001.wav \
        --speaker "LJSpeech (Linda Johnson)" \
        --licence "Public domain" --consent open-licence

What it writes: the clip under ``inputs/`` and, beside it, a provenance
sidecar (``provenance.sidecar_path`` names it). The sidecar is what every
later consent and admissibility check reads (golden rule 3: no voice without
one). It does not download anything and loads no model.

**Provenance sidecar.** A small file stored next to a recording that says where
it came from, who is speaking, under what licence and on what consent basis.
Keeping it beside the audio means the record travels with the file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Make backend/ importable without PYTHONPATH; must run before the import below.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# noqa: E402 silences "import not at top of file": it has to follow the path set-up.
import provenance  # noqa: E402

# The cloner only reads references from inputs/; Kokoro output lives in outputs/.
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# XTTS-v2 wants a clean 24 kHz mono reference; audio_utils converts anything
# else, but writing it correctly here keeps the smoke path free of surprises.
TARGET_SR = 24000
# Middle of the 30-60 s range the assignment asks for; collection stops once
# this much audio is gathered.
TARGET_SECONDS = 45.0

# Kokoro clips only. Mixing MMS output would blend speakers and produce a
# reference that represents no single voice.
SMOKE_GLOBS = ("kokoro_*.wav", "quality_*.wav", "api_*.wav", "speech.wav")


def collect_smoke_sources() -> List[Path]:
    """Same-voice Kokoro clips, longest first so fewer joins are needed."""
    found: List[Path] = []
    for pattern in SMOKE_GLOBS:
        # glob() matches shell-style wildcards in one directory (not recursive).
        found.extend(OUTPUTS_DIR.glob(pattern))
        found.extend((OUTPUTS_DIR / "benchmark").glob(pattern))
    # A set of resolved paths drops duplicates (two patterns can match one
    # file); sorting first makes the order repeatable between runs.
    unique = sorted({p.resolve() for p in found if p.is_file()})
    # File size stands in for duration: same format, so bigger means longer.
    return sorted(unique, key=lambda p: p.stat().st_size, reverse=True)


def build_smoke_reference(dest: Path) -> int:
    """
    Concatenate Kokoro clips into a synthetic reference at ``dest``; return the exit code.

    The sidecar marks it SYNTHETIC, so it can prove the pipeline runs but can
    never count as evidence for a similarity threshold.
    """
    # Imported here so --human, which needs neither, starts without them.
    import numpy as np
    import soundfile as sf

    sources = collect_smoke_sources()
    if not sources:
        print(
            f"No Kokoro output found under {OUTPUTS_DIR}. Generate some speech "
            "first, then re-run.",
            file=sys.stderr,
        )
        return 1

    # A short silence between clips keeps the join from sounding like a click,
    # which would otherwise show up as a transient in the speaker embedding.
    gap = np.zeros(int(0.25 * TARGET_SR), dtype=np.float32)
    chunks: List[np.ndarray] = []
    used: List[str] = []
    total = 0.0

    for path in sources:
        if total >= TARGET_SECONDS:
            break
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=False)
        # BLE001 is the lint rule against catching bare Exception. Any unreadable
        # file is skipped with its reason printed, not fatal to the whole build.
        except Exception as err:  # noqa: BLE001
            print(f"  skip {path.name}: {err}")
            continue
        # Stereo -> mono by averaging the channels.
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != TARGET_SR:
            # Resampling a mismatched clip would need librosa; skipping keeps
            # the smoke path dependency-light and the voice consistent.
            print(f"  skip {path.name}: {sr} Hz, expected {TARGET_SR}")
            continue
        chunks.extend([data, gap])
        used.append(path.name)
        total += len(data) / TARGET_SR

    if not chunks:
        print("No usable 24 kHz mono clips.", file=sys.stderr)
        return 1

    audio = np.concatenate(chunks)
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    # PCM_16 is the ordinary 16-bit WAV format every tool reads.
    sf.write(dest, audio, TARGET_SR, subtype="PCM_16")

    # The notes list every source file, so the clip can be traced back.
    sidecar = provenance.write(
        dest,
        source=provenance.SYNTHETIC,
        speaker="Kokoro v1.0 (synthetic)",
        licence="n/a - generated by this project",
        notes=(
            "Concatenated from this project's own Kokoro output to exercise the "
            f"XTTS-v2 cloning path. Built from {len(used)} clips: "
            f"{', '.join(used)}."
        ),
    )

    print(f"Wrote {dest.relative_to(PROJECT_ROOT)}  ({total:.1f}s from {len(used)} clips)")
    print(f"Wrote {sidecar.relative_to(PROJECT_ROOT)}  (source: synthetic)")
    if total < 30.0:
        print(
            f"\nNote: only {total:.1f}s available. The assignment asks for 30-60s; "
            "generate more speech for a fuller smoke reference."
        )
    print(
        "\nThis clip is for pipeline testing. The benchmark will measure it and "
        "report PIPELINE_TEST, never PASS."
    )
    return 0


def register_human_reference(args: argparse.Namespace) -> int:
    """
    Copy a real recording into ``inputs/`` and write its HUMAN provenance sidecar.

    Returns the exit code. The recording is not modified; the consent basis
    must be one ``provenance`` knows, or nothing is written.
    """
    # expanduser() turns "~/Downloads" into the real home directory path.
    source = Path(args.human).expanduser()
    if not source.is_file():
        print(f"Not a file: {source}", file=sys.stderr)
        return 1
    # Checked before any copy, so an invalid consent leaves no file behind.
    if args.consent not in provenance.CONSENT_BASES:
        print(
            f"--consent must be one of {sorted(provenance.CONSENT_BASES)}",
            file=sys.stderr,
        )
        return 1

    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = INPUTS_DIR / source.name
    # Skip the copy when the file is already in inputs/: copying a file onto
    # itself raises shutil.SameFileError. copy2 also keeps the timestamps.
    if dest.resolve() != source.resolve():
        shutil.copy2(source, dest)

    sidecar = provenance.write(
        dest,
        source=provenance.HUMAN,
        speaker=args.speaker,
        licence=args.licence,
        consent_basis=args.consent,
        notes=args.notes,
    )
    # Read the record back through the same function the benchmarks use, so
    # the printed verdict is the one they will reach.
    info = provenance.describe(dest)
    print(f"Registered {dest.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {sidecar.relative_to(PROJECT_ROOT)}")
    print(f"Admissible as acceptance evidence: {info['admissible']} ({info['reason']})")
    return 0


def main() -> int:
    """Parse the command line and run the chosen mode; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    # A mutually exclusive group: argparse refuses both --smoke and --human
    # together, and required=True refuses neither.
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="build a synthetic reference from this project's Kokoro output",
    )
    mode.add_argument("--human", metavar="FILE", help="register a real recording")
    parser.add_argument("--speaker", default="unknown", help="who is speaking")
    parser.add_argument("--licence", default="", help="licence covering the clip")
    parser.add_argument(
        "--consent",
        default="",
        help=f"consent basis, one of {sorted(provenance.CONSENT_BASES)}",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--name",
        default="smoke_reference_kokoro.wav",
        help="output filename for --smoke",
    )
    args = parser.parse_args()

    if args.smoke:
        return build_smoke_reference(INPUTS_DIR / args.name)
    return register_human_reference(args)


if __name__ == "__main__":
    # raise SystemExit(code) exits with main()'s return value, like sys.exit().
    raise SystemExit(main())
