#!/usr/bin/env python
"""
Download the model files the vision pipeline loads.

The speech side has ``fetch_models.py``; this is its twin for everything the
face, lip-sync, metric and studio code needs. ``backend/model_registry.py``
(``audit_vision_weights``) reports what is missing and every engine error
names this script, so there is exactly one way to fix a missing vision model.

    PYTHONPATH=backend backend/.conda/bin/python scripts/fetch_vision_models.py

Flags:
  --only face-landmarker,syncnet   fetch just these keys
  --all                            include the large default-off models
  --accept-licence wav2lip         required for models under a restricted licence
  --dry-run                        print the plan and the disk cost, download nothing
  --yes                            skip the confirmation prompt

Nothing here is fetched silently at runtime. A licence-restricted model is
never downloaded unless a human names it with --accept-licence, the same rule
the speech side applies to XTTS-v2's CPML.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import model_registry  # noqa: E402
from model_registry import audit_vision_weights  # noqa: E402

# Same floor as fetch_models.py: a half-written cache on a full disk reports
# as present and then fails to load.
MIN_FREE_GB = 5.0

_MEDIAPIPE = "https://storage.googleapis.com/mediapipe-models"


@dataclass(frozen=True)
class VisionSpec:
    key: str
    name: str
    approx_gb: float
    default: bool
    licence: str
    note: str
    fetch: Callable[[], None]
    # Set for licences a human has to accept before the download starts.
    licence_gate: Optional[str] = None


def download_file(
    url: str,
    dest: Path,
    min_bytes: int = 0,
    sha256: Optional[str] = None,
) -> Path:
    """
    Download ``url`` to ``dest`` atomically.

    The body goes to ``<dest>.part`` and is renamed only after the size (and
    checksum, when one is pinned) checks pass, so an interrupted download can
    never leave a file the audit would count as present.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "ai-avatar-platform"})
    with urllib.request.urlopen(request, timeout=60) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last_percent = -1
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if total:
                percent = done * 100 // total
                if percent != last_percent and percent % 10 == 0:
                    print(f"    {dest.name}: {percent}% of {total / 1e6:.0f} MB")
                    last_percent = percent

    size = partial.stat().st_size
    if size < min_bytes:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"{dest.name}: downloaded {size} bytes, expected at least {min_bytes}. "
            f"The source may have moved: {url}"
        )
    if sha256 and digest.hexdigest() != sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"{dest.name}: checksum mismatch (got {digest.hexdigest()}, "
            f"expected {sha256}). Refusing to install an unverified model file."
        )
    partial.replace(dest)
    return dest


def _fetch_videoseal() -> None:
    """The VideoSeal 1.0 checkpoint (pinned size and SHA-256) and the small config file its wheel omits."""
    from video_watermark import ATTENUATION_FILE, ATTENUATION_YAML, CHECKPOINT, CHECKPOINT_BYTES, CHECKPOINT_SHA256, CHECKPOINT_URL

    download_file(CHECKPOINT_URL, CHECKPOINT, min_bytes=CHECKPOINT_BYTES, sha256=CHECKPOINT_SHA256)
    ATTENUATION_FILE.write_text(ATTENUATION_YAML)


def _fetch_face_landmarker() -> None:
    download_file(
        f"{_MEDIAPIPE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        model_registry.FACE_LANDMARKER_TASK,
        min_bytes=1_000_000,
    )


def _fetch_selfie_segmenter() -> None:
    download_file(
        f"{_MEDIAPIPE}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        model_registry.SELFIE_SEGMENTER_TFLITE,
        min_bytes=100_000,
    )


def _fetch_multiclass_segmenter() -> None:
    download_file(
        f"{_MEDIAPIPE}/image_segmenter/selfie_multiclass_256x256/float32/latest/"
        "selfie_multiclass_256x256.tflite",
        model_registry.MULTICLASS_SEGMENTER_TFLITE,
        min_bytes=1_000_000,
    )


def _fetch_syncnet() -> None:
    download_file(
        "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
        model_registry.SYNCNET_MODEL,
        min_bytes=10_000_000,
    )


def _fetch_sface() -> None:
    download_file(
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        model_registry.SFACE_MODEL,
        min_bytes=10_000_000,
    )


def _fetch_realesrgan() -> None:
    download_file(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        model_registry.SUPER_RESOLUTION_MODEL,
        min_bytes=4_000_000,
        sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
    )


def _fetch_wav2lip() -> None:
    download_file(
        "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth",
        model_registry.WAV2LIP_CHECKPOINTS[0],
        min_bytes=100_000_000,
    )


def _fetch_avatar_diffusion() -> None:
    """
    Half-precision safetensors only.

    The repo also carries full-precision and ``.bin`` copies of every
    component; fetching all of them is 30+ GB for weights a 6 GB card cannot
    hold anyway.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_registry.AVATAR_DIFFUSION_REPO,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "feature_extractor/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "safety_checker/config.json",
            "safety_checker/model.fp16.safetensors",
        ],
    )


SPECS: List[VisionSpec] = [
    VisionSpec(
        key="face-landmarker",
        name="MediaPipe Face Landmarker",
        approx_gb=0.004,
        default=True,
        licence="Apache-2.0",
        note="REQUIRED: 478 landmarks, head pose, 52 blendshapes",
        fetch=_fetch_face_landmarker,
    ),
    VisionSpec(
        key="selfie-segmenter",
        name="MediaPipe Selfie Segmenter",
        approx_gb=0.001,
        default=True,
        licence="Apache-2.0",
        note="background replacement",
        fetch=_fetch_selfie_segmenter,
    ),
    VisionSpec(
        key="multiclass-segmenter",
        name="MediaPipe Multiclass Segmenter",
        approx_gb=0.017,
        default=True,
        licence="Apache-2.0",
        note="hair / clothes masks for the customization studio",
        fetch=_fetch_multiclass_segmenter,
    ),
    VisionSpec(
        key="videoseal",
        name="VideoSeal 1.0 (256-bit invisible video watermark)",
        approx_gb=0.23,
        default=True,
        licence="MIT (Meta; code and weights)",
        note="every rendered video is watermarked with it; also needs the packages in video_watermark.VIDEOSEAL_PIP",
        fetch=_fetch_videoseal,
    ),
    VisionSpec(
        key="syncnet",
        name="SyncNet v2",
        approx_gb=0.055,
        default=True,
        licence="MIT (syncnet_python)",
        note="LSE-C / LSE-D, the only admissible lip-sync metric",
        fetch=_fetch_syncnet,
    ),
    VisionSpec(
        key="sface",
        name="SFace face recognition",
        approx_gb=0.039,
        default=True,
        licence="Apache-2.0 (OpenCV Zoo)",
        note="identity-preservation score for stylised avatars",
        fetch=_fetch_sface,
    ),
    VisionSpec(
        key="realesrgan",
        name="Real-ESRGAN general x4v3",
        approx_gb=0.005,
        default=True,
        licence="BSD-3-Clause (xinntao/Real-ESRGAN)",
        note="enlarges a small photo once so a 1080P_HQ render can reach 1080p",
        fetch=_fetch_realesrgan,
    ),
    VisionSpec(
        key="avatar-diffusion",
        name="Stable Diffusion 1.5 (fp16)",
        approx_gb=2.8,
        default=True,
        licence="CreativeML OpenRAIL-M (use-based restrictions apply)",
        note="avatar generator, style transfer, hair / clothing edits",
        fetch=_fetch_avatar_diffusion,
    ),
    VisionSpec(
        key="wav2lip",
        name="Wav2Lip GAN",
        approx_gb=0.44,
        default=False,
        licence="Research / non-commercial only (trained on LRS2)",
        note="neural lip sync; the blendshape engine renders without it",
        fetch=_fetch_wav2lip,
        licence_gate="wav2lip",
    ),
]


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1_000_000_000


def select_specs(only: Optional[str], include_all: bool) -> List[VisionSpec]:
    """The specs a run covers. Raises ``ValueError`` on an unknown key."""
    if only:
        wanted = {k.strip() for k in only.split(",") if k.strip()}
        unknown = wanted - {s.key for s in SPECS}
        if unknown:
            raise ValueError(f"unknown model key(s): {', '.join(sorted(unknown))}")
        return [s for s in SPECS if s.key in wanted]
    return [s for s in SPECS if s.default or include_all]


def licence_blocked(specs: List[VisionSpec], accepted: set) -> List[VisionSpec]:
    """Specs that need a licence acceptance the caller has not given."""
    return [s for s in specs if s.licence_gate and s.licence_gate not in accepted]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated model keys")
    parser.add_argument(
        "--all", action="store_true", help="include the default-off models"
    )
    parser.add_argument(
        "--accept-licence",
        default="",
        help="comma-separated licence gates a human has read and accepted (e.g. wav2lip)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="skip confirmation")
    args = parser.parse_args(argv)

    try:
        selected = select_specs(args.only, args.all)
    except ValueError as err:
        parser.error(str(err))

    present = {s.key for s in audit_vision_weights() if s.present}
    todo = [s for s in selected if s.key not in present]
    for spec in selected:
        if spec.key in present:
            print(f"  have  {spec.key:<21} {spec.name}")
    if not todo:
        print("\nNothing to fetch: every selected model is already on disk.")
        return 0

    total_gb = sum(s.approx_gb for s in todo)
    print("\nTo fetch:")
    for spec in todo:
        print(f"  ~{spec.approx_gb:>5.2f} GB  {spec.key:<21} {spec.name}")
        print(f"               licence: {spec.licence}")
        print(f"               {spec.note}")
    available = free_gb(PROJECT_ROOT)
    print(f"\nApprox download: {total_gb:.2f} GB   Free disk: {available:.1f} GB")

    accepted = {g.strip() for g in args.accept_licence.split(",") if g.strip()}
    blocked = licence_blocked(todo, accepted)
    if blocked:
        for spec in blocked:
            print(
                f"\n{spec.key} is under a restricted licence: {spec.licence}.\n"
                f"A person must read and accept it; re-run with "
                f"--accept-licence {spec.licence_gate}",
                file=sys.stderr,
            )
        if args.dry_run:
            return 0
        return 2

    if args.dry_run:
        return 0

    if available - total_gb < MIN_FREE_GB:
        print(
            f"\nRefusing to download: this would leave under {MIN_FREE_GB:.0f} GB "
            "free. Free up space, or narrow the set with --only.",
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

    # Re-audit instead of trusting the downloads: a partial file is exactly
    # the state the audit exists to detect.
    print("\n" + "=" * 60)
    print("Post-fetch audit")
    for status in audit_vision_weights():
        mark = "OK  " if status.present else "MISS"
        print(f"  {mark}  {status.key:<21} {status.size_label:>8}  {status.detail}")
    print("=" * 60)

    if failures:
        print(f"\n{len(failures)} download(s) failed: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
