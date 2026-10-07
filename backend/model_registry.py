"""
Model weight presence audit.

``VoiceEngineRouter`` loads every backend lazily and caches load failures so a
missing model degrades to a fallback instead of failing the request. That is
the right behaviour in production and the wrong behaviour during development:
three of the five routed models sat in the tree for several phases with only
their ``config.json`` cached, every test passed, and the benchmark silently
measured Kokoro in their place.

This module answers one narrow question without downloading anything or
importing torch: *are this model's weights actually on disk?* A HuggingFace
snapshot that holds no weight file is metadata only -- ``from_pretrained``
would reach the network, and offline it raises.

The audit runs at API startup (loudly) and is served from ``/health`` so the
frontend and the benchmark script can refuse to report a model as working when
its weights were never fetched.
"""

from __future__ import annotations

import logging
import os
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Extensions that hold actual trained parameters, as opposed to configs,
# tokenizer vocabularies or READMEs.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt", ".ckpt", ".onnx")

# A tokenizer vocabulary can be a .bin of a few hundred KB, so require at least
# one weight file big enough to be a real checkpoint shard.
MIN_CHECKPOINT_BYTES = 1_000_000

# ---------------------------------------------------------------------------
# Vision-side model files. They live in the project's own ``.models/`` tree
# rather than the HuggingFace cache, so the path *is* the contract: the engine
# that loads a file and the audit that reports on it read the same constant.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_DIR = PROJECT_ROOT / ".models"

MEDIAPIPE_DIR = LOCAL_MODEL_DIR / "mediapipe"
FACE_LANDMARKER_TASK = MEDIAPIPE_DIR / "face_landmarker.task"
SELFIE_SEGMENTER_TFLITE = MEDIAPIPE_DIR / "selfie_segmenter.tflite"
# Hair / skin / clothes classes, which the customization studio needs to know
# *what* to repaint. The plain selfie segmenter only separates person from
# background.
MULTICLASS_SEGMENTER_TFLITE = MEDIAPIPE_DIR / "selfie_multiclass_256x256.tflite"

WAV2LIP_DIR = LOCAL_MODEL_DIR / "wav2lip"
# Either checkpoint drives the same network. The GAN one looks sharper, the
# plain one syncs slightly tighter; the GAN file wins when both are present.
WAV2LIP_CHECKPOINTS = (WAV2LIP_DIR / "wav2lip_gan.pth", WAV2LIP_DIR / "wav2lip.pth")

SYNCNET_MODEL = LOCAL_MODEL_DIR / "syncnet" / "syncnet_v2.model"
SFACE_MODEL = LOCAL_MODEL_DIR / "sface" / "face_recognition_sface_2021dec.onnx"

# Overridable so a team with more VRAM, or a preferred checkpoint, can swap
# the generator without touching code. Must be a Stable Diffusion 1.x layout.
AVATAR_DIFFUSION_REPO = os.getenv(
    "AVATAR_DIFFUSION_REPO", "stable-diffusion-v1-5/stable-diffusion-v1-5"
)

VISION_FETCH_COMMAND = "python scripts/fetch_vision_models.py"
MODEL_FETCH_COMMAND = "python scripts/fetch_models.py"


@dataclass(frozen=True)
class ModelWeightStatus:
    """Whether one routed model's weights are on disk."""

    key: str
    name: str
    source: str
    present: bool
    size_bytes: int
    detail: str
    # How to fix it, when not present. Usually the fetch command; for a model
    # this stack cannot run at all, fetching would not help and this says so.
    fix: str = ""

    @property
    def size_label(self) -> str:
        if self.size_bytes >= 1_000_000_000:
            return f"{self.size_bytes / 1_000_000_000:.1f} GB"
        if self.size_bytes >= 1_000_000:
            return f"{self.size_bytes / 1_000_000:.0f} MB"
        if self.size_bytes > 0:
            return f"{self.size_bytes / 1_000:.0f} KB"
        return "0 B"

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "name": self.name,
            "source": self.source,
            "present": self.present,
            "sizeBytes": self.size_bytes,
            "sizeLabel": self.size_label,
            "detail": self.detail,
            "fix": self.fix,
        }


def _weight_bytes(root: Path) -> int:
    """Total bytes of checkpoint-sized weight files under ``root``."""
    total = 0
    if not root.is_dir():
        return 0
    for path in root.rglob("*"):
        if path.suffix.lower() not in WEIGHT_SUFFIXES:
            continue
        try:
            # Snapshots are symlinks into blobs/, so resolve before measuring.
            size = path.resolve().stat().st_size
        except OSError:
            continue
        if size >= MIN_CHECKPOINT_BYTES:
            total += size
    return total


def _hf_cache_root() -> Path:
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001
        return Path(
            os.getenv("HF_HUB_CACHE")
            or os.getenv("HUGGINGFACE_HUB_CACHE")
            or Path.home() / ".cache" / "huggingface" / "hub"
        )


def _hf_repo_dir(repo_id: str) -> Path:
    return _hf_cache_root() / f"models--{repo_id.replace('/', '--')}"


def check_hf_repo(key: str, name: str, repo_id: str) -> ModelWeightStatus:
    """Audit one HuggingFace repo in the local hub cache."""
    repo_dir = _hf_repo_dir(repo_id)
    if not repo_dir.is_dir():
        return ModelWeightStatus(
            key=key,
            name=name,
            source=repo_id,
            present=False,
            size_bytes=0,
            detail="not in the HuggingFace cache; never downloaded",
        )
    size = _weight_bytes(repo_dir / "snapshots")
    if size == 0:
        return ModelWeightStatus(
            key=key,
            name=name,
            source=repo_id,
            present=False,
            size_bytes=0,
            detail=(
                "cached metadata only (no weight file) - from_pretrained would "
                "download on first use and fail when offline"
            ),
        )
    return ModelWeightStatus(
        key=key,
        name=name,
        source=repo_id,
        present=True,
        size_bytes=size,
        detail="weights present",
    )


def _coqui_roots() -> List[Path]:
    """
    Candidate Coqui TTS download directories.

    Coqui derives this from ``XDG_DATA_HOME``, which the VS Code snap rewrites
    to a sandboxed path. A backend started from a normal shell therefore looks
    somewhere different than one started from the IDE terminal, so check both.
    """
    roots: List[Path] = []
    try:
        from TTS.utils.manage import ModelManager

        roots.append(Path(ModelManager().output_prefix))
    except Exception as exc:  # noqa: BLE001 - Coqui TTS is optional here
        # Without TTS installed there is no Coqui root to report; the XDG
        # fallbacks below still cover where it would have downloaded to.
        logger.debug("Coqui ModelManager unavailable (%s); using XDG paths only", exc)
    xdg = os.getenv("XDG_DATA_HOME")
    if xdg:
        roots.append(Path(xdg) / "tts")
    roots.append(Path.home() / ".local" / "share" / "tts")
    unique: List[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _coqui_incomplete(folder: Path) -> Optional[str]:
    """
    Why a Coqui download folder cannot be loaded, or None if it looks whole.

    Size alone is not enough. On 30 Sep 2026 the XTTS-v2 download stopped at
    70%: `model.pth` was 1.30 of 1.87 GB and every other file was missing,
    yet the audit said "weights present" because the folder was not empty.

    Two cheap checks, no torch:
    * Coqui's loader needs `config.json`, the download's small companion file;
    * a modern PyTorch checkpoint is a zip whose directory sits at the *end*
      of the file, so a truncated one starts with the zip signature but has no
      directory. `zipfile.is_zipfile` reads only the tail, so a 1.3 GB file
      costs milliseconds. Legacy (non-zip) checkpoints are not judged.
    """
    if not (folder / "config.json").is_file():
        return "config.json is missing, so the download did not finish"
    for checkpoint in sorted(folder.glob("*.pth")):
        with checkpoint.open("rb") as handle:
            looks_like_zip = handle.read(2) == b"PK"
        if looks_like_zip and not zipfile.is_zipfile(checkpoint):
            return (
                f"{checkpoint.name} is truncated ({checkpoint.stat().st_size:,} bytes, "
                "no zip directory), so the download did not finish"
            )
    return None


def check_coqui_model(key: str, name: str, model_name: str) -> ModelWeightStatus:
    """Audit a Coqui TTS model such as XTTS-v2 in its download directory."""
    folder = model_name.replace("/", "--")
    searched: List[str] = []
    broken: Optional[ModelWeightStatus] = None
    for root in _coqui_roots():
        candidate = root / folder
        searched.append(str(candidate))
        size = _weight_bytes(candidate)
        if not size:
            continue
        problem = _coqui_incomplete(candidate)
        if problem is None:
            return ModelWeightStatus(
                key=key,
                name=name,
                source=str(candidate),
                present=True,
                size_bytes=size,
                detail="weights present",
            )
        # Keep looking: another root may hold a complete copy. Remember this
        # one so the report says *why* it was not counted.
        broken = broken or ModelWeightStatus(
            key=key,
            name=name,
            source=str(candidate),
            present=False,
            size_bytes=size,
            detail=f"incomplete download in {candidate}: {problem}",
        )
    if broken is not None:
        return broken
    return ModelWeightStatus(
        key=key,
        name=name,
        source=model_name,
        present=False,
        size_bytes=0,
        detail=(
            "no Coqui download found, so voice cloning cannot run. Searched: "
            + ", ".join(searched)
        ),
    )


def check_mms(key: str = "mms-tts", name: str = "MMS-TTS") -> ModelWeightStatus:
    """
    Audit MMS-TTS, which is one ~145 MB VITS checkpoint per language.

    There is no single repo to check, so report how many are cached. Any one of
    them proves the path works; the router downloads the rest on demand.
    """
    root = _hf_cache_root()
    prefix = "models--facebook--mms-tts-"
    cached: List[str] = []
    total = 0
    if root.is_dir():
        for repo_dir in sorted(root.glob(f"{prefix}*")):
            size = _weight_bytes(repo_dir / "snapshots")
            if size:
                cached.append(repo_dir.name[len(prefix) :])
                total += size
    if not cached:
        return ModelWeightStatus(
            key=key,
            name=name,
            source="facebook/mms-tts-<iso3>",
            present=False,
            size_bytes=0,
            detail="no per-language checkpoint cached yet; each is fetched on demand",
        )
    return ModelWeightStatus(
        key=key,
        name=name,
        source="facebook/mms-tts-<iso3>",
        present=True,
        size_bytes=total,
        detail=f"{len(cached)} language(s) cached: {', '.join(cached)}",
    )


# Routed engines that cannot run on this stack whatever is downloaded,
# established 8 Oct 2026 by loading their real cached configs (T2.6). The
# transformers pin (<4.48) is forced by Coqui TTS; re-check if it ever moves.
UNRUNNABLE: Dict[str, str] = {
    "higgs-tts-2": (
        "cannot run on this stack: its config declares model type "
        "'higgs_audio_v2', which the pinned transformers (<4.48, required by "
        "Coqui TTS) does not recognise, and its 11.6 GB of fp32 weights exceed "
        "the 6 GB card. Downloading it would not help"
    ),
    "dia-1.6b": (
        "cannot run on this stack: its config has no model type the pinned "
        "transformers (<4.48) recognises - it needs the nari-tts package, which "
        "requires numpy 2 while Coqui TTS needs numpy below 2 - and each 6.4 GB "
        "copy of its weights exceeds the 6 GB card. Downloading it would not help"
    ),
}


def _unrunnable(key: str, name: str, repo_id: str) -> ModelWeightStatus:
    return ModelWeightStatus(
        key=key,
        name=name,
        source=repo_id,
        present=False,
        size_bytes=0,
        detail=UNRUNNABLE[key],
        fix="none on this stack (see detail); use another engine",
    )


def audit_model_weights() -> List[ModelWeightStatus]:
    """Audit every model ``VoiceEngineRouter.select_model`` can return."""
    statuses = [
        check_hf_repo("kokoro", "Kokoro v1.0 (82M)", "hexgrad/Kokoro-82M"),
        check_coqui_model(
            "xtts-v2",
            "XTTS-v2 (voice cloning)",
            "tts_models/multilingual/multi-dataset/xtts_v2",
        ),
        _unrunnable("higgs-tts-2", "Higgs TTS 2 (3B)", "bosonai/higgs-tts-2-3b-base"),
        _unrunnable("dia-1.6b", "Dia-1.6B (dialogue)", "nari-labs/Dia-1.6B"),
        check_mms(),
    ]
    # Every other missing model is fixed by fetching it. `replace` builds a
    # new frozen instance with one field changed.
    return [
        s if s.present or s.fix else replace(s, fix=f"{MODEL_FETCH_COMMAND} --only {s.key}")
        for s in statuses
    ]


def check_local_file(
    key: str,
    name: str,
    paths,
    min_bytes: int = 100_000,
    missing_detail: str = "",
) -> ModelWeightStatus:
    """
    Audit a model that is a single file in ``.models/``.

    ``paths`` may be one path or several acceptable alternatives; the first one
    present wins. A file smaller than ``min_bytes`` is reported absent: an
    interrupted download or a saved HTML error page would otherwise pass an
    existence check and fail much later, inside the loader.
    """
    candidates = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    for candidate in candidates:
        try:
            size = candidate.stat().st_size if candidate.is_file() else 0
        except OSError:
            size = 0
        if size >= min_bytes:
            return ModelWeightStatus(
                key=key,
                name=name,
                source=str(candidate),
                present=True,
                size_bytes=size,
                detail="weights present",
            )
    truncated = [c for c in candidates if c.is_file()]
    if truncated:
        detail = (
            f"{truncated[0].name} is only {truncated[0].stat().st_size} bytes - "
            f"a truncated download. Delete it and re-run: "
            f"{VISION_FETCH_COMMAND} --only {key}"
        )
    else:
        detail = missing_detail or (
            f"not downloaded. Fetch with: {VISION_FETCH_COMMAND} --only {key}"
        )
    return ModelWeightStatus(
        key=key,
        name=name,
        source=str(candidates[0]),
        present=False,
        size_bytes=0,
        detail=detail,
    )


def audit_vision_weights() -> List[ModelWeightStatus]:
    """
    Audit every model the vision pipeline can load.

    Kept separate from ``audit_model_weights`` because the two lists answer
    different questions: that one is "which TTS routes are real", this one is
    "which render, metric and studio features are real". Only the face
    landmarker is required for a talking avatar; everything else degrades a
    named feature, and the detail string says which.
    """
    sd = check_hf_repo(
        "avatar-diffusion",
        "Stable Diffusion 1.5 (avatar generator / studio)",
        AVATAR_DIFFUSION_REPO,
    )
    if not sd.present:
        sd = ModelWeightStatus(
            key=sd.key,
            name=sd.name,
            source=sd.source,
            present=False,
            size_bytes=0,
            detail=(
                f"{sd.detail}. Avatar generation, style transfer and hair/clothing "
                f"edits cannot run. Fetch with: {VISION_FETCH_COMMAND} --only avatar-diffusion"
            ),
        )
    return [
        check_local_file(
            "face-landmarker",
            "MediaPipe Face Landmarker (478 points)",
            FACE_LANDMARKER_TASK,
            min_bytes=1_000_000,
        ),
        check_local_file(
            "selfie-segmenter",
            "MediaPipe Selfie Segmenter (background)",
            SELFIE_SEGMENTER_TFLITE,
        ),
        check_local_file(
            "multiclass-segmenter",
            "MediaPipe Multiclass Segmenter (hair / clothes)",
            MULTICLASS_SEGMENTER_TFLITE,
            min_bytes=1_000_000,
        ),
        check_local_file(
            "wav2lip",
            "Wav2Lip (neural lip sync)",
            WAV2LIP_CHECKPOINTS,
            min_bytes=100_000_000,
            missing_detail=(
                "not downloaded, so the 'wav2lip' render engine is unavailable "
                "(the blendshape engine still renders). Research / non-commercial "
                f"licence, a human must accept it: {VISION_FETCH_COMMAND} "
                "--only wav2lip --accept-licence wav2lip"
            ),
        ),
        check_local_file(
            "syncnet",
            "SyncNet v2 (LSE-C / LSE-D lip-sync metric)",
            SYNCNET_MODEL,
            min_bytes=10_000_000,
        ),
        check_local_file(
            "sface",
            "SFace (identity preservation metric)",
            SFACE_MODEL,
            min_bytes=10_000_000,
        ),
        sd,
    ]


def log_weight_audit(statuses: Optional[List[ModelWeightStatus]] = None) -> List[ModelWeightStatus]:
    """
    Log the audit at startup, warning per missing model.

    A missing model is a warning rather than a fatal error: Kokoro alone is
    enough to serve English. Requests that need a missing model are refused
    with the fetch command (VoiceEngineRouter.require_weights) rather than
    downloading it mid-request or quietly substituting another engine.
    """
    statuses = statuses if statuses is not None else audit_model_weights()
    present = [s for s in statuses if s.present]
    missing = [s for s in statuses if not s.present]

    logger.info(
        "Model weight audit: %d/%d available (%s)",
        len(present),
        len(statuses),
        ", ".join(f"{s.key} {s.size_label}" for s in present) or "none",
    )
    for status in missing:
        logger.warning(
            "MODEL UNAVAILABLE - %s [%s]: %s. Requests that need it are "
            "refused. Fix: %s",
            status.name,
            status.key,
            status.detail,
            status.fix,
        )
    return statuses


def log_vision_audit(
    statuses: Optional[List[ModelWeightStatus]] = None,
) -> List[ModelWeightStatus]:
    """Log the vision audit at startup; one warning per missing model."""
    statuses = statuses if statuses is not None else audit_vision_weights()
    present = [s for s in statuses if s.present]
    logger.info(
        "Vision weight audit: %d/%d available (%s)",
        len(present),
        len(statuses),
        ", ".join(f"{s.key} {s.size_label}" for s in present) or "none",
    )
    for status in statuses:
        if not status.present:
            logger.warning(
                "VISION WEIGHTS MISSING - %s [%s]: %s",
                status.name,
                status.key,
                status.detail,
            )
    return statuses


def _summarise(statuses: List[ModelWeightStatus]) -> Dict[str, object]:
    return {
        "available": sum(1 for s in statuses if s.present),
        "total": len(statuses),
        "missing": [s.key for s in statuses if not s.present],
        "models": [s.to_dict() for s in statuses],
    }


def vision_audit_summary() -> Dict[str, object]:
    """The vision audit projected for ``/health``."""
    return _summarise(audit_vision_weights())


def audit_summary() -> Dict[str, object]:
    """The audit projected for ``/health``."""
    statuses = audit_model_weights()
    return {
        "available": sum(1 for s in statuses if s.present),
        "total": len(statuses),
        "missing": [s.key for s in statuses if not s.present],
        "models": [s.to_dict() for s in statuses],
    }
