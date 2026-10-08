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

What it writes: each model file at the path ``model_registry`` expects, plus
the Stable Diffusion files in the HuggingFace hub cache. It prints the plan, the download progress, and a fresh
audit at the end.

Exit codes: 0 done (or nothing to do, or a dry run), 1 a download failed, the
disk is too full or the user said no, 2 a licence was not accepted,
130 interrupted with Ctrl-C (the shell convention for SIGINT).

**Atomic download**, the core idea of ``download_file``: write to a temporary
``.part`` file and rename it to the real name only once it has been checked.
A rename within one folder is all-or-nothing, so the real path either holds a
complete, verified file or does not exist. A crash halfway leaves only a
``.part`` file, which the audit ignores.

**SHA-256 checksum**: a fingerprint of the file's bytes. Comparing it with a
value pinned in the code proves the download is exactly the file that was
reviewed, not a truncated copy, an error page or a changed upstream file.

To add a model: write a ``_fetch_<name>()`` function (usually one call to
``download_file`` with the URL, the destination path from ``model_registry``
and a ``min_bytes`` floor), add a ``VisionSpec`` to ``SPECS`` with the same
key the registry audit uses, and give it a ``licence_gate`` if its licence
is restricted. The audit in ``model_registry.audit_vision_weights`` must know
the key too, or the model will be fetched on every run.
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

# parents[1] is the folder above scripts/, i.e. the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Make backend/ importable so this script shares model paths with the server
# instead of keeping its own copy that could drift.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# E402 ("import not at top") is expected: these imports need the path set above.
import model_registry  # noqa: E402
from model_registry import audit_vision_weights  # noqa: E402

# Same floor as fetch_models.py: a half-written cache on a full disk reports
# as present and then fails to load.
MIN_FREE_GB = 5.0

# Base URL shared by the three MediaPipe models.
_MEDIAPIPE = "https://storage.googleapis.com/mediapipe-models"


# frozen=True makes instances read-only, so the SPECS table cannot be changed by accident at runtime.
@dataclass(frozen=True)
class VisionSpec:
    """
    One downloadable model: what it is, what it costs, and how to fetch it.

    ``fetch`` holds a function rather than a URL because the models come from
    different places: a single file over HTTPS, or a set of files from the
    HuggingFace hub. Each spec carries its own zero-argument fetcher.
    """

    # The id used on the command line (--only) and in the registry audit.
    key: str
    # Human-readable name for the plan and the logs.
    name: str
    # Rough download size, used for the plan and the free-disk check.
    approx_gb: float
    # True: fetched by a plain run. False: only with --all or --only.
    default: bool
    # Printed in the plan so the person running the script sees it before agreeing.
    licence: str
    # What the model is used for, so the reader can judge whether to skip it.
    note: str
    # Callable[[], None]: a function taking no arguments and returning nothing.
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

    ``min_bytes`` is a floor that catches error pages and truncated bodies.
    ``sha256``, when given, must match the file's hex digest exactly. Either
    failure deletes the ``.part`` file and raises ``RuntimeError`` naming the
    URL, so the message says where to look. Returns ``dest``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Same folder as the destination, so the final rename never crosses a disk.
    partial = dest.with_name(dest.name + ".part")
    # The hash is fed chunk by chunk while downloading, so the file is never read twice.
    digest = hashlib.sha256()
    # Send an explicit User-Agent naming the project rather than urllib's default one.
    request = urllib.request.Request(url, headers={"User-Agent": "ai-avatar-platform"})
    # Both resources in one `with`: the connection and the file are closed even if a read fails.
    with urllib.request.urlopen(request, timeout=60) as response, open(partial, "wb") as out:
        # Content-Length may be missing; 0 then means "unknown size" and progress is not printed.
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last_percent = -1
        while True:
            # 1 << 20 is 1 MiB. Reading in chunks keeps memory flat for files of hundreds of MB.
            chunk = response.read(1 << 20)
            # An empty read means the server has sent everything.
            if not chunk:
                break
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if total:
                # Integer percent; print only on each new multiple of 10 so the log stays short.
                percent = done * 100 // total
                if percent != last_percent and percent % 10 == 0:
                    print(f"    {dest.name}: {percent}% of {total / 1e6:.0f} MB")
                    last_percent = percent

    # Size floor: a moved file often returns a small HTML error page with status 200,
    # which would otherwise be installed as a "model".
    # Checks run after the `with` block has closed the file, so every byte is flushed to disk first.
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
    # The atomic step: replace() renames over any existing file in one operation.
    partial.replace(dest)
    return dest


# The _fetch_* functions below each fetch one model. The min_bytes values are
# floors well under the real size: big enough to catch an error page, small
# enough not to break when upstream re-publishes a slightly different build.


def _fetch_videoseal() -> None:
    """The VideoSeal 1.0 checkpoint (pinned size and SHA-256) and the small config file its wheel omits."""
    # The URL, size and hash live in video_watermark.py, next to the code that
    # loads the checkpoint, so there is one pinned value. Imported only when this model is fetched.
    from video_watermark import ATTENUATION_FILE, ATTENUATION_YAML, CHECKPOINT, CHECKPOINT_BYTES, CHECKPOINT_SHA256, CHECKPOINT_URL

    download_file(CHECKPOINT_URL, CHECKPOINT, min_bytes=CHECKPOINT_BYTES, sha256=CHECKPOINT_SHA256)
    ATTENUATION_FILE.write_text(ATTENUATION_YAML)


def _fetch_face_landmarker() -> None:
    """MediaPipe Face Landmarker: the one model a talking avatar cannot render without."""
    download_file(
        f"{_MEDIAPIPE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        model_registry.FACE_LANDMARKER_TASK,
        min_bytes=1_000_000,
    )


def _fetch_selfie_segmenter() -> None:
    """MediaPipe Selfie Segmenter: person-versus-background mask for background replacement."""
    download_file(
        f"{_MEDIAPIPE}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        model_registry.SELFIE_SEGMENTER_TFLITE,
        min_bytes=100_000,
    )


def _fetch_multiclass_segmenter() -> None:
    """MediaPipe multiclass segmenter: separate hair and clothes masks for the studio."""
    download_file(
        # Two adjacent string literals are joined by Python into one URL.
        f"{_MEDIAPIPE}/image_segmenter/selfie_multiclass_256x256/float32/latest/"
        "selfie_multiclass_256x256.tflite",
        model_registry.MULTICLASS_SEGMENTER_TFLITE,
        min_bytes=1_000_000,
    )


def _fetch_syncnet() -> None:
    """SyncNet v2 from the Oxford VGG group: scores how well lips match audio (LSE-C / LSE-D)."""
    download_file(
        "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
        model_registry.SYNCNET_MODEL,
        min_bytes=10_000_000,
    )


def _fetch_sface() -> None:
    """SFace (OpenCV Zoo, ONNX): face embeddings used to score identity after stylising."""
    download_file(
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        model_registry.SFACE_MODEL,
        min_bytes=10_000_000,
    )


def _fetch_realesrgan() -> None:
    """Real-ESRGAN x4 super-resolution, pinned to one release file by SHA-256."""
    download_file(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        model_registry.SUPER_RESOLUTION_MODEL,
        min_bytes=4_000_000,
        sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
    )


def _fetch_wav2lip() -> None:
    """Wav2Lip GAN checkpoint. Research-only licence, so its spec carries a licence gate."""
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
    # Lazy import: huggingface_hub is only needed for this one model.
    from huggingface_hub import snapshot_download

    # snapshot_download fetches a whole repo into the HuggingFace cache (where the
    # diffusers loader looks for it); allow_patterns limits it to the listed files.
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


# The catalogue. The keys must match the ones audit_vision_weights() reports,
# because main() compares the two to decide what is already on disk.
SPECS: List[VisionSpec] = [
    VisionSpec(
        # Required: the face analysis and every render need it.
        key="face-landmarker",
        name="MediaPipe Face Landmarker",
        approx_gb=0.004,
        default=True,
        licence="Apache-2.0",
        note="REQUIRED: 478 landmarks, head pose, 52 blendshapes",
        fetch=_fetch_face_landmarker,
    ),
    VisionSpec(
        # Optional feature: background replacement is off without it.
        key="selfie-segmenter",
        name="MediaPipe Selfie Segmenter",
        approx_gb=0.001,
        default=True,
        licence="Apache-2.0",
        note="background replacement",
        fetch=_fetch_selfie_segmenter,
    ),
    VisionSpec(
        # Optional feature: hair and clothing edits in the studio.
        key="multiclass-segmenter",
        name="MediaPipe Multiclass Segmenter",
        approx_gb=0.017,
        default=True,
        licence="Apache-2.0",
        note="hair / clothes masks for the customization studio",
        fetch=_fetch_multiclass_segmenter,
    ),
    VisionSpec(
        # The only spec with a pinned checksum and an extra config file (see _fetch_videoseal).
        key="videoseal",
        name="VideoSeal 1.0 (256-bit invisible video watermark)",
        approx_gb=0.23,
        default=True,
        licence="MIT (Meta; code and weights)",
        note="every rendered video is watermarked with it; also needs the packages in video_watermark.VIDEOSEAL_PIP",
        fetch=_fetch_videoseal,
    ),
    VisionSpec(
        # Measurement, not rendering: without it lip sync cannot be scored.
        key="syncnet",
        name="SyncNet v2",
        approx_gb=0.055,
        default=True,
        licence="MIT (syncnet_python)",
        note="LSE-C / LSE-D, the only admissible lip-sync metric",
        fetch=_fetch_syncnet,
    ),
    VisionSpec(
        # Measurement: how much of the person survives a style change.
        key="sface",
        name="SFace face recognition",
        approx_gb=0.039,
        default=True,
        licence="Apache-2.0 (OpenCV Zoo)",
        note="identity-preservation score for stylised avatars",
        fetch=_fetch_sface,
    ),
    VisionSpec(
        # Pinned by SHA-256 in _fetch_realesrgan.
        key="realesrgan",
        name="Real-ESRGAN general x4v3",
        approx_gb=0.005,
        default=True,
        licence="BSD-3-Clause (xinntao/Real-ESRGAN)",
        note="enlarges a small photo once so a 1080P_HQ render can reach 1080p",
        fetch=_fetch_realesrgan,
    ),
    VisionSpec(
        # By far the largest default download; narrow it out with --only if disk is short.
        key="avatar-diffusion",
        name="Stable Diffusion 1.5 (fp16)",
        approx_gb=2.8,
        default=True,
        licence="CreativeML OpenRAIL-M (use-based restrictions apply)",
        note="avatar generator, style transfer, hair / clothing edits",
        fetch=_fetch_avatar_diffusion,
    ),
    # The only default-off and licence-gated entry: non-commercial weights are
    # never fetched by a plain run, and never without --accept-licence wav2lip.
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
    """Free space, in decimal gigabytes, on the disk that holds ``path``."""
    return shutil.disk_usage(path).free / 1_000_000_000


def select_specs(only: Optional[str], include_all: bool) -> List[VisionSpec]:
    """
    The specs a run covers. Raises ``ValueError`` on an unknown key.

    ``only`` is the raw ``--only`` text (comma-separated keys) or None;
    ``include_all`` is ``--all``. With neither, the default-on models are chosen.
    """
    if only:
        # Split "a, b,," into {"a", "b"}: strip spaces and drop empty pieces.
        wanted = {k.strip() for k in only.split(",") if k.strip()}
        # Set difference: the requested keys that no spec has. A typo fails loudly instead of fetching nothing.
        unknown = wanted - {s.key for s in SPECS}
        if unknown:
            raise ValueError(f"unknown model key(s): {', '.join(sorted(unknown))}")
        # --only wins over `default`: naming a model fetches it even when it is default-off.
        return [s for s in SPECS if s.key in wanted]
    return [s for s in SPECS if s.default or include_all]


def licence_blocked(specs: List[VisionSpec], accepted: set) -> List[VisionSpec]:
    """
    Specs that need a licence acceptance the caller has not given.

    ``accepted`` holds the gate names from ``--accept-licence``. A spec with
    no gate is never blocked.
    """
    return [s for s in specs if s.licence_gate and s.licence_gate not in accepted]


def main(argv: Optional[List[str]] = None) -> int:
    """
    Plan, check and run the downloads; return the process exit code.

    ``argv`` defaults to the real command line; tests pass a list instead.
    The order matters: the plan is printed first, then the licence check, then
    the disk check, then the prompt, so a dry run can show every problem
    without downloading anything.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated model keys")
    # store_true: a flag with no value; present means True, absent means False.
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
        # parser.error prints the usage and the message, then exits with status 2.
        parser.error(str(err))

    # Ask the registry, not the file system directly: it applies the same
    # presence rules (such as minimum file sizes) that /health reports.
    present = {s.key for s in audit_vision_weights() if s.present}
    # Only missing models are fetched, so re-running the script is safe and cheap.
    todo = [s for s in selected if s.key not in present]
    for spec in selected:
        if spec.key in present:
            print(f"  have  {spec.key:<21} {spec.name}")
    if not todo:
        print("\nNothing to fetch: every selected model is already on disk.")
        return 0

    # The sum of the rough sizes; good enough for the plan and the disk check.
    total_gb = sum(s.approx_gb for s in todo)
    print("\nTo fetch:")
    # Format specs: `:<21` pads to 21 characters left-aligned, `:>5.2f` right-aligns a 2-decimal number.
    for spec in todo:
        print(f"  ~{spec.approx_gb:>5.2f} GB  {spec.key:<21} {spec.name}")
        print(f"               licence: {spec.licence}")
        print(f"               {spec.note}")
    # Measured on the disk that holds the project, where the model files land.
    available = free_gb(PROJECT_ROOT)
    print(f"\nApprox download: {total_gb:.2f} GB   Free disk: {available:.1f} GB")

    # The licence gate: a restricted model is never downloaded unless a person named it.
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
        # A dry run only reports, so a blocked licence is not an error there.
        if args.dry_run:
            return 0
        return 2

    if args.dry_run:
        return 0

    # Check the space left after the download, not before: the floor must still hold once the files land.
    if available - total_gb < MIN_FREE_GB:
        print(
            f"\nRefusing to download: this would leave under {MIN_FREE_GB:.0f} GB "
            "free. Free up space, or narrow the set with --only.",
            file=sys.stderr,
        )
        return 1

    if not args.yes:
        # Default is No: only an explicit y or yes proceeds.
        reply = input("\nProceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 1

    # One failed model does not stop the rest; failures are collected and reported at the end.
    failures: List[str] = []
    for spec in todo:
        print(f"\n--- {spec.key}: {spec.name} ---")
        try:
            spec.fetch()
        # Ctrl-C stops the whole run. KeyboardInterrupt is not an Exception subclass,
        # so it would skip the handler below anyway; catching it gives a clean message.
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            return 130
        # Broad catch on purpose (BLE001 silenced): any failure of one fetch is recorded, not fatal.
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


# SystemExit with an int sets the process exit status, same as sys.exit().
if __name__ == "__main__":
    raise SystemExit(main())
