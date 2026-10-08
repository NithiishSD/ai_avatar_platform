#!/usr/bin/env python
"""
Download the model weights the router routes to.

Three of the five routed models sat in the tree with only their config cached.
The router's fallbacks hid it, the tests mocked past it, and the Phase 3
benchmark measured Kokoro in their place. ``backend/model_registry.py`` now
detects that state; this script fixes it.

    PYTHONPATH=backend backend/.conda/bin/python scripts/fetch_models.py

Flags:
  --only kokoro,xtts-v2   fetch just these keys
  --all                   include models skipped by default (see SPECS)
  --dry-run               print the plan and the disk cost, download nothing
  --yes                   skip the confirmation prompt

Disk matters here: the full set is roughly 15 GB and the project disk was at
88% when this was written, so the script refuses to start a download that
would leave less than MIN_FREE_GB free.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model_registry import UNRUNNABLE, audit_model_weights  # noqa: E402
from openvoice_engine import OPENVOICE_REPO  # noqa: E402
from bark_engine import BARK_REPO, DIALOGUE_VOICES  # noqa: E402
from watermark_engine import AUDIOSEAL_FILES, AUDIOSEAL_REPO  # noqa: E402

# Refuse to fill the disk. A GPU box that runs out of room mid-download leaves
# a half-written cache that reports as present but fails to load.
MIN_FREE_GB = 5.0


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    approx_gb: float
    # Default-off models are the ones that cost a lot and buy little for the
    # demo; --all opts in.
    default: bool
    note: str
    fetch: Callable[[], None]


def _fetch_hf(repo_id: str, allow_patterns: Optional[List[str]] = None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns)


def _fetch_kokoro() -> None:
    _fetch_hf("hexgrad/Kokoro-82M")


def _fetch_xtts() -> None:
    """
    Coqui downloads XTTS-v2 on first construction, gated by a licence prompt.

    COQUI_TOS_AGREED is how Coqui exposes that acceptance for unattended runs;
    the caller has already been shown the licence note below.
    """
    import os

    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    from TTS.api import TTS

    TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")


def _fetch_openvoice() -> None:
    # The converter only (131 MB); the base speakers are MeloTTS's, unused here.
    _fetch_hf(OPENVOICE_REPO, allow_patterns=["converter/*"])


def _fetch_bark() -> None:
    # The model plus only the two English presets dialogue mode uses (of 784).
    presets = [f"speaker_embeddings/{voice}_*" for voice in DIALOGUE_VOICES.values()]
    _fetch_hf(BARK_REPO, allow_patterns=["*.json", "*.txt", "pytorch_model.bin", *presets])


def _fetch_audioseal() -> None:
    # Only the two 16-bit checkpoints (59 MB + 35 MB); the 453 MB 32 kHz model is not used.
    _fetch_hf(AUDIOSEAL_REPO, allow_patterns=list(AUDIOSEAL_FILES))


def _fetch_mms() -> None:
    """
    MMS-TTS is one checkpoint per language, fetched on demand at ~145 MB each.

    Pre-fetch only the four the Phase 3 benchmark already reports, so its
    numbers stay reproducible offline.
    """
    for iso3 in ("hin", "tam", "swh", "spa"):
        _fetch_hf(f"facebook/mms-tts-{iso3}")


SPECS: List[ModelSpec] = [
    ModelSpec(
        key="kokoro",
        name="Kokoro v1.0 (82M)",
        approx_gb=0.4,
        default=True,
        note="fast English path; already present in most checkouts",
        fetch=_fetch_kokoro,
    ),
    ModelSpec(
        key="xtts-v2",
        name="XTTS-v2 (voice cloning)",
        approx_gb=2.0,
        default=True,
        note="REQUIRED for Integration Gate 2. Coqui CPML licence: "
        "non-commercial use only",
        fetch=_fetch_xtts,
    ),
    ModelSpec(
        key="mms-tts",
        name="MMS-TTS (hin, tam, swh, spa)",
        approx_gb=0.6,
        default=True,
        note="the four languages the Phase 3 benchmark reports",
        fetch=_fetch_mms,
    ),
    ModelSpec(
        key="openvoice-v2",
        name="OpenVoice V2 converter (tone-colour cloning)",
        approx_gb=0.13,
        default=True,
        note="MIT; clones over Kokoro / MMS-TTS. Needs the openvoice package, see requirements.txt",
        fetch=_fetch_openvoice,
    ),
    ModelSpec(
        key="audioseal",
        name="AudioSeal (inaudible audio watermark)",
        approx_gb=0.1,
        default=True,
        note="MIT; every generated clip is watermarked with it. Needs `pip install --no-deps audioseal==0.2.0 omegaconf`",
        fetch=_fetch_audioseal,
    ),
    ModelSpec(
        key="bark",
        name="Bark small (two-speaker dialogue)",
        approx_gb=1.7,
        default=True,
        note="MIT; speaks [S1]/[S2] dialogue, replacing Dia, which cannot run here",
        fetch=_fetch_bark,
    ),
]


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1_000_000_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated model keys")
    parser.add_argument(
        "--all", action="store_true", help="include the large default-off models"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="skip confirmation")
    args = parser.parse_args()

    present = {s.key for s in audit_model_weights() if s.present}

    if args.only:
        wanted = {k.strip() for k in args.only.split(",") if k.strip()}
        # Higgs and Dia used to be offered here: 18 GB that this stack cannot
        # load (see model_registry.UNRUNNABLE). Refuse with the reason instead.
        refused = wanted & set(UNRUNNABLE)
        if refused:
            for key in sorted(refused):
                print(f"{key}: {UNRUNNABLE[key]}.", file=sys.stderr)
            return 1
        unknown = wanted - {s.key for s in SPECS}
        if unknown:
            parser.error(f"unknown model key(s): {', '.join(sorted(unknown))}")
        selected = [s for s in SPECS if s.key in wanted]
    else:
        selected = [s for s in SPECS if s.default or args.all]

    todo = [s for s in selected if s.key not in present]
    skipped = [s for s in selected if s.key in present]

    for spec in skipped:
        print(f"  have  {spec.key:<13} {spec.name}")
    if not todo:
        print("\nNothing to fetch: every selected model already has weights.")
        return 0

    total_gb = sum(s.approx_gb for s in todo)
    print("\nTo fetch:")
    for spec in todo:
        print(f"  ~{spec.approx_gb:>4.1f} GB  {spec.key:<13} {spec.name}")
        print(f"              {spec.note}")
    available = free_gb(PROJECT_ROOT)
    print(f"\nApprox download: {total_gb:.1f} GB   Free disk: {available:.1f} GB")

    if args.dry_run:
        return 0

    if available - total_gb < MIN_FREE_GB:
        print(
            f"\nRefusing to download: this would leave under {MIN_FREE_GB:.0f} GB "
            f"free. Free up space, or narrow the set with --only.",
            file=sys.stderr,
        )
        return 1

    if not args.yes:
        reply = input("\nProceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 1

    failures: List[str] = []
    for spec in todo:
        print(f"\n--- {spec.key}: {spec.name} ---")
        try:
            spec.fetch()
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            return 130
        except Exception as err:  # noqa: BLE001
            print(f"FAILED {spec.key}: {err}", file=sys.stderr)
            failures.append(spec.key)

    # Re-audit rather than trusting that the download said it succeeded: a
    # partial snapshot is exactly the state this script exists to detect.
    print("\n" + "=" * 60)
    print("Post-fetch audit")
    for status in audit_model_weights():
        mark = "OK  " if status.present else "MISS"
        print(f"  {mark}  {status.key:<13} {status.size_label:>8}  {status.detail}")
    print("=" * 60)

    if failures:
        print(f"\n{len(failures)} download(s) failed: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
