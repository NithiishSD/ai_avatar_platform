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

Where it sits in the pipeline: nothing here renders or synthesises. ``app.py``
calls ``log_weight_audit`` and ``log_vision_audit`` in its startup hook and
``audit_summary`` / ``vision_audit_summary`` in ``/health``.
``VoiceEngineRouter.require_weights`` re-runs ``audit_model_weights`` before
every synthesis, so a request for a missing model is refused up front. The
vision engines (face, Wav2Lip, SyncNet, SFace, Real-ESRGAN, the avatar
generator) import their file paths from here, so the loader and the audit can
never disagree about where a file lives.

Concepts used throughout, explained once here:

**Model weights** are the trained numbers of a neural network, saved in a
checkpoint file (``.safetensors``, ``.pth``, ``.onnx`` ...). A model's code is
useless without them. The small files next to them (``config.json``, a
tokenizer vocabulary) describe the network's shape but hold no learning.

**The HuggingFace hub cache** is where ``from_pretrained`` stores downloads.
Each repo gets a folder named ``models--<org>--<name>``. Inside it,
``blobs/`` holds the real file contents, named by hash, and
``snapshots/<commit>/`` holds symlinks with the human file names pointing into
``blobs/``. One repo can have several snapshots, one per revision fetched.

**How "present" is decided.** Each check below is a cheap filesystem test, and
each is stricter than "the folder exists", because every weaker test has
already lied once in this project:

* a HuggingFace repo is present when its snapshots hold at least one weight
  file of checkpoint size (``check_hf_repo``), or every named file
  (``check_hf_files``);
* a Coqui model is present when its folder holds weights, its ``config.json``
  and no truncated zip checkpoint (``check_coqui_model``);
* a single local file is present when it is at least ``min_bytes`` long
  (``check_local_file``);
* a model whose package must be importable is present only when
  ``importlib.util.find_spec`` finds that package too.

**The ``fix`` field** carries the command that repairs a missing model, so
every refusal and every log line can tell the reader what to run (golden rule
7: errors say how to fix themselves).

**How to say this in an interview:** "Lazy loading with fallbacks hides missing
models, so we audit weight presence separately, without importing torch, and
we refuse a request for a missing model instead of quietly substituting
another one."
"""

from __future__ import annotations

import logging
import os
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# One logger per module, named after it ("model_registry"), so log lines say
# where they came from and can be filtered by name.
logger = logging.getLogger(__name__)

# Extensions that hold actual trained parameters, as opposed to configs,
# tokenizer vocabularies or READMEs. .safetensors and .bin are HuggingFace
# formats, .pth/.pt/.ckpt are PyTorch checkpoints, .onnx is the portable
# ONNX graph format. A tuple, so it cannot be changed by accident at runtime.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt", ".ckpt", ".onnx")

# A tokenizer vocabulary can be a .bin of a few hundred KB, so require at least
# one weight file big enough to be a real checkpoint shard. 1 MB sits safely
# between the two for the repos audited this way: their checkpoints are far
# bigger and their vocabularies smaller. (Underscores are digit separators.)
MIN_CHECKPOINT_BYTES = 1_000_000

# ---------------------------------------------------------------------------
# Vision-side model files. They live in the project's own ``.models/`` tree
# rather than the HuggingFace cache, so the path *is* the contract: the engine
# that loads a file and the audit that reports on it read the same constant.
# ---------------------------------------------------------------------------
# __file__ is backend/model_registry.py; resolve() makes it absolute and
# follows symlinks, and parents[1] climbs two levels to the repository root.
# Anchoring on this file, not the working directory, means the paths are right
# whether the server is started from the repo root, backend/ or a container.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_DIR = PROJECT_ROOT / ".models"

# MediaPipe ships its models as single files. A ``.task`` file bundles a model
# with its pre/post-processing; ``.tflite`` is a bare TensorFlow Lite model.
MEDIAPIPE_DIR = LOCAL_MODEL_DIR / "mediapipe"
# Finds 478 3D points on a face; the talking avatar cannot render without it.
FACE_LANDMARKER_TASK = MEDIAPIPE_DIR / "face_landmarker.task"
# Person-versus-background mask, used to cut the subject out of a photo.
SELFIE_SEGMENTER_TFLITE = MEDIAPIPE_DIR / "selfie_segmenter.tflite"
# Hair / skin / clothes classes, which the customization studio needs to know
# *what* to repaint. The plain selfie segmenter only separates person from
# background.
MULTICLASS_SEGMENTER_TFLITE = MEDIAPIPE_DIR / "selfie_multiclass_256x256.tflite"

WAV2LIP_DIR = LOCAL_MODEL_DIR / "wav2lip"
# Either checkpoint drives the same network. The GAN one looks sharper, the
# plain one syncs slightly tighter; the GAN file wins when both are present.
WAV2LIP_CHECKPOINTS = (WAV2LIP_DIR / "wav2lip_gan.pth", WAV2LIP_DIR / "wav2lip.pth")

# SyncNet scores whether lips match audio; it gives the LSE-C / LSE-D lip-sync
# metrics (lipsync_metric.py). It measures renders; it never makes them.
SYNCNET_MODEL = LOCAL_MODEL_DIR / "syncnet" / "syncnet_v2.model"
# SFace turns a face into an identity vector; comparing the photo's vector with
# a rendered frame's says how much of the person survived (visual_metrics.py).
SFACE_MODEL = LOCAL_MODEL_DIR / "sface" / "face_recognition_sface_2021dec.onnx"
# Real-ESRGAN compact general model: enlarges a small photo for 1080p renders (super_resolution.py).
SUPER_RESOLUTION_MODEL = LOCAL_MODEL_DIR / "realesrgan" / "realesr-general-x4v3.pth"

# Overridable so a team with more VRAM, or a preferred checkpoint, can swap
# the generator without touching code. Must be a Stable Diffusion 1.x layout.
# Unlike the files above, this one is a HuggingFace repo id, so it lives in the
# hub cache and is audited with ``check_hf_repo``. Read once, at import time.
AVATAR_DIFFUSION_REPO = os.getenv(
    "AVATAR_DIFFUSION_REPO", "stable-diffusion-v1-5/stable-diffusion-v1-5"
)

# The two fetch scripts, quoted in every "fix" message. Defined once so the
# advice printed by the audit, the router and the engines cannot drift apart.
# Vision files go to .models/, speech models to the HuggingFace / Coqui caches.
VISION_FETCH_COMMAND = "python scripts/fetch_vision_models.py"
MODEL_FETCH_COMMAND = "python scripts/fetch_models.py"


@dataclass(frozen=True)
class ModelWeightStatus:
    """
    Whether one routed model's weights are on disk.

    One row of the audit. ``frozen=True`` makes instances immutable: assigning
    to a field raises, so a status cannot be edited after it was measured, and
    a changed copy is made with ``dataclasses.replace`` instead.
    ``to_dict`` gives the camelCase shape served by ``/health``.
    """

    # Stable short id, e.g. "kokoro"; the router and the fetch scripts use it.
    key: str
    # Human label for logs and the UI.
    name: str
    # Where it was looked for or found: a repo id, a folder or a file path.
    source: str
    present: bool
    # Bytes of weights found; 0 when absent. Shown so a half-sized file stands out.
    size_bytes: int
    # One sentence saying what was found, or why the model does not count.
    detail: str
    # How to fix it, when not present. Usually the fetch command; for a model
    # this stack cannot run at all, fetching would not help and this says so.
    fix: str = ""

    # @property makes a method read like an attribute: ``status.size_label``,
    # no parentheses. It is computed from size_bytes, so it can never disagree.
    @property
    def size_label(self) -> str:
        """``size_bytes`` as a short human string ("1.9 GB", "82 MB", "0 B")."""
        # Decimal units (1 GB = 10**9 bytes), the way download pages state sizes.
        if self.size_bytes >= 1_000_000_000:
            return f"{self.size_bytes / 1_000_000_000:.1f} GB"
        if self.size_bytes >= 1_000_000:
            return f"{self.size_bytes / 1_000_000:.0f} MB"
        if self.size_bytes > 0:
            return f"{self.size_bytes / 1_000:.0f} KB"
        return "0 B"

    def to_dict(self) -> Dict[str, object]:
        """
        The JSON shape served by ``/health``.

        Keys are camelCase because the frontend is JavaScript; the Python
        fields stay snake_case. ``sizeLabel`` is included so the UI does not
        have to reimplement the unit formatting.
        """
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
    """
    Total bytes of checkpoint-sized weight files under ``root``.

    Returns 0 when ``root`` does not exist or holds no weight file of at least
    ``MIN_CHECKPOINT_BYTES``. Every caller treats 0 as "not present", so this
    one number is both the presence test and the size shown in the report.
    """
    total = 0
    if not root.is_dir():
        return 0
    # rglob("*") walks every file and folder below root, at any depth, so
    # sharded checkpoints in sub-folders (unet/, text_encoder/ ...) are counted.
    for path in root.rglob("*"):
        # .lower() so "MODEL.PTH" counts too; configs and vocabularies are skipped.
        if path.suffix.lower() not in WEIGHT_SUFFIXES:
            continue
        try:
            # Snapshots are symlinks into blobs/, so resolve before measuring.
            size = path.resolve().stat().st_size
        except OSError:
            # A dangling symlink (its blob was deleted or never finished)
            # raises here. It holds no weights, so it is skipped, not fatal.
            continue
        if size >= MIN_CHECKPOINT_BYTES:
            total += size
    return total


def _hf_cache_root() -> Path:
    """
    The HuggingFace hub cache folder this process would download into.

    Asks ``huggingface_hub`` first, because it applies every override
    (``HF_HOME``, ``HF_HUB_CACHE`` ...) exactly as ``from_pretrained`` will.
    """
    try:
        # Imported here, not at the top, so this module still imports (and
        # /health still answers) when huggingface_hub is not installed.
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001
        # Without the library, rebuild its default: the two environment
        # variables it honours, then ~/.cache/huggingface/hub.
        return Path(
            os.getenv("HF_HUB_CACHE")
            or os.getenv("HUGGINGFACE_HUB_CACHE")
            or Path.home() / ".cache" / "huggingface" / "hub"
        )


def _hf_repo_dir(repo_id: str) -> Path:
    """
    The cache folder for one repo id.

    The hub's naming rule: "hexgrad/Kokoro-82M" becomes
    "models--hexgrad--Kokoro-82M" (a "/" cannot appear in a folder name).
    """
    return _hf_cache_root() / f"models--{repo_id.replace('/', '--')}"


def check_hf_repo(key: str, name: str, repo_id: str) -> ModelWeightStatus:
    """
    Audit one HuggingFace repo in the local hub cache.

    Present when any snapshot holds at least one checkpoint-sized weight file.
    Returns a status in one of three states: never downloaded, metadata only,
    or weights present. Never raises and never touches the network.
    """
    repo_dir = _hf_repo_dir(repo_id)
    # State 1: no folder at all, so nothing was ever fetched.
    if not repo_dir.is_dir():
        return ModelWeightStatus(
            key=key,
            name=name,
            source=repo_id,
            present=False,
            size_bytes=0,
            detail="not in the HuggingFace cache; never downloaded",
        )
    # Measure snapshots/, not blobs/: snapshot names carry the file suffix,
    # while blob names are bare hashes that WEIGHT_SUFFIXES could not match.
    size = _weight_bytes(repo_dir / "snapshots")
    # State 2: the folder exists but holds only configs. This is the case the
    # module docstring describes, where three models looked installed.
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
        # Coqui's own answer comes first: it is the folder its loader reads.
        from TTS.utils.manage import ModelManager

        roots.append(Path(ModelManager().output_prefix))
    except Exception as exc:  # noqa: BLE001 - Coqui TTS is optional here
        # Without TTS installed there is no Coqui root to report; the XDG
        # fallbacks below still cover where it would have downloaded to.
        logger.debug("Coqui ModelManager unavailable (%s); using XDG paths only", exc)
    xdg = os.getenv("XDG_DATA_HOME")
    if xdg:
        roots.append(Path(xdg) / "tts")
    # The XDG default when XDG_DATA_HOME is unset, i.e. a normal shell.
    roots.append(Path.home() / ".local" / "share" / "tts")
    # The same folder often appears twice (Coqui's answer equals an XDG path).
    # Drop repeats but keep order, so the preferred root is still checked first
    # and the "Searched:" message does not list a folder twice.
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
    # sorted() so the same file is reported first on every run.
    for checkpoint in sorted(folder.glob("*.pth")):
        # Every zip file begins with the two bytes "PK" (the format's author,
        # Phil Katz). Reading just those two tells a zip checkpoint from a
        # legacy pickle one without loading anything.
        with checkpoint.open("rb") as handle:
            looks_like_zip = handle.read(2) == b"PK"
        # Starts like a zip but has no readable directory at its end: the
        # file was cut off part-way, which is what an interrupted download does.
        if looks_like_zip and not zipfile.is_zipfile(checkpoint):
            return (
                f"{checkpoint.name} is truncated ({checkpoint.stat().st_size:,} bytes, "
                "no zip directory), so the download did not finish"
            )
    return None


def check_coqui_model(key: str, name: str, model_name: str) -> ModelWeightStatus:
    """
    Audit a Coqui TTS model such as XTTS-v2 in its download directory.

    Coqui does not use the HuggingFace cache; it downloads into its own folder
    (see ``_coqui_roots``). Returns present for the first root holding a whole
    copy; otherwise the first incomplete copy found, with the reason; otherwise
    "not found", listing every folder searched so the reader can check by hand.
    """
    # Coqui's folder naming: "tts_models/multilingual/multi-dataset/xtts_v2"
    # is stored as "tts_models--multilingual--multi-dataset--xtts_v2".
    folder = model_name.replace("/", "--")
    searched: List[str] = []
    broken: Optional[ModelWeightStatus] = None
    for root in _coqui_roots():
        candidate = root / folder
        searched.append(str(candidate))
        size = _weight_bytes(candidate)
        # No weights here at all: try the next root.
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
        # one so the report says *why* it was not counted. ``broken or ...``
        # keeps the first broken copy found rather than the last.
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
    # Every language is its own repo, facebook/mms-tts-<iso3> (e.g. "eng",
    # "hin"), so one glob on the cache-folder prefix finds them all.
    prefix = "models--facebook--mms-tts-"
    cached: List[str] = []
    total = 0
    if root.is_dir():
        for repo_dir in sorted(root.glob(f"{prefix}*")):
            size = _weight_bytes(repo_dir / "snapshots")
            if size:
                # Slice off the prefix to keep only the language code.
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
    """
    A permanent "not present" status for a model listed in ``UNRUNNABLE``.

    The cache is not even inspected: the answer does not depend on it. The
    ``fix`` is set here, so ``audit_model_weights`` will not replace it with a
    fetch command that would download gigabytes and change nothing.
    """
    return ModelWeightStatus(
        key=key,
        name=name,
        source=repo_id,
        present=False,
        size_bytes=0,
        detail=UNRUNNABLE[key],
        fix="none on this stack (see detail); use another engine",
    )


def check_openvoice(key: str = "openvoice-v2", name: str = "OpenVoice V2 (tone-colour cloning)") -> ModelWeightStatus:
    """
    OpenVoice needs both its package and its converter files.

    `find_spec` locates the package without importing it, so this stays a
    cheap filesystem check like the rest of the audit.
    """
    import importlib.util

    # Imported inside the function: openvoice_engine owns these constants, and
    # importing it at the top of this module would load it for every caller of
    # the audit, including ones that never touch OpenVoice.
    from openvoice_engine import CONVERTER_FILES, OPENVOICE_PIP, OPENVOICE_REPO

    # Package first: weights without the code that runs them are no use, and
    # the fix for that case is a pip install, not a download.
    if importlib.util.find_spec("openvoice") is None:
        return ModelWeightStatus(
            key=key, name=name, source=OPENVOICE_REPO, present=False, size_bytes=0,
            detail="the openvoice package is not installed", fix=OPENVOICE_PIP,
        )
    return check_hf_files(key, name, OPENVOICE_REPO, CONVERTER_FILES)


def check_hf_files(key: str, name: str, repo_id: str, required: Sequence[str]) -> ModelWeightStatus:
    """
    Present only if one cached snapshot holds *every* required file.

    Stricter than `check_hf_repo`, which accepts any weight file: a snapshot
    with the checkpoint but not its config would load no better than nothing.
    """
    snapshots = _hf_repo_dir(repo_id) / "snapshots"
    # Each snapshot is judged on its own: required files split across two
    # revisions would not load together.
    for snapshot in sorted(snapshots.glob("*")) if snapshots.is_dir() else []:
        files = [snapshot / rel for rel in required]
        # is_file() follows the symlink, so a link whose blob is missing fails.
        if all(f.is_file() for f in files):
            return ModelWeightStatus(
                key=key, name=name, source=str(snapshot), present=True,
                size_bytes=sum(f.stat().st_size for f in files), detail="weights present",
            )
    return ModelWeightStatus(
        key=key, name=name, source=repo_id, present=False, size_bytes=0,
        detail=f"{', '.join(required)} not all in the HuggingFace cache",
    )


def check_audioseal(key: str = "audioseal", name: str = "AudioSeal (audio watermark)") -> ModelWeightStatus:
    """The watermark needs its package and both checkpoints (about 94 MB, MIT)."""
    import importlib.util

    from watermark_engine import AUDIOSEAL_FILES, AUDIOSEAL_PIP, AUDIOSEAL_REPO

    # AUDIOSEAL_PIP installs audioseal with --no-deps, so its omegaconf
    # dependency is not pulled in automatically; check both are importable.
    missing = [m for m in ("audioseal", "omegaconf") if importlib.util.find_spec(m) is None]
    if missing:
        return ModelWeightStatus(
            key=key, name=name, source=AUDIOSEAL_REPO, present=False, size_bytes=0,
            detail=f"the {', '.join(missing)} package is not installed", fix=AUDIOSEAL_PIP,
        )
    return check_hf_files(key, name, AUDIOSEAL_REPO, list(AUDIOSEAL_FILES))


def check_videoseal(key: str = "videoseal", name: str = "VideoSeal 1.0 (invisible video watermark)") -> ModelWeightStatus:
    """The video watermark needs its packages, the 228 MB checkpoint at its exact size, and its config."""
    import importlib.util

    from video_watermark import ATTENUATION_FILE, CHECKPOINT, CHECKPOINT_BYTES, VIDEOSEAL_PIP

    # Python module names, which differ from pip names: "av" is PyAV and
    # "yaml" is PyYAML. VIDEOSEAL_PIP installs videoseal with --no-deps and
    # these separately, so any one of them can be missing on its own.
    missing = [m for m in ("videoseal", "timm", "av", "yaml") if importlib.util.find_spec(m) is None]
    if missing:
        return ModelWeightStatus(
            key=key, name=name, source=str(CHECKPOINT), present=False, size_bytes=0,
            detail=f"the {', '.join(missing)} package is not installed", fix=VIDEOSEAL_PIP,
        )
    size = CHECKPOINT.stat().st_size if CHECKPOINT.is_file() else 0
    # An exact byte count, not a minimum: the checkpoint size is known, so any
    # other size (shorter or longer) means a damaged or wrong file.
    if size != CHECKPOINT_BYTES or not ATTENUATION_FILE.is_file():
        return ModelWeightStatus(
            key=key, name=name, source=str(CHECKPOINT), present=False, size_bytes=size,
            detail=(f"checkpoint is {size} bytes, expected {CHECKPOINT_BYTES}" if size else "not downloaded")
            + ("" if ATTENUATION_FILE.is_file() else "; the attenuation config is missing"),
            fix=f"{VISION_FETCH_COMMAND} --only videoseal",
        )
    return ModelWeightStatus(key=key, name=name, source=str(CHECKPOINT), present=True, size_bytes=size, detail="weights present")


def check_sadtalker(key: str = "sadtalker", name: str = "SadTalker (head motion and expression)") -> ModelWeightStatus:
    """SadTalker needs its pinned code checkout, four weight files at their exact sizes and five packages."""
    import sadtalker_engine

    problems = sadtalker_engine.missing()
    size = sum(
        (sadtalker_engine.SADTALKER_DIR / relative).stat().st_size
        for relative in sadtalker_engine.SADTALKER_FILES
        if (sadtalker_engine.SADTALKER_DIR / relative).is_file()
    )
    return ModelWeightStatus(
        key=key, name=name, source=str(sadtalker_engine.SADTALKER_DIR), present=not problems, size_bytes=size,
        # Default-off like Wav2Lip: absent means only that the 'sadtalker' engine is unavailable.
        detail="weights present" if not problems else "; ".join(problems)
        + ". The 'sadtalker' render engine is unavailable; the blendshape engine still renders",
        fix="" if not problems else sadtalker_engine.FETCH_COMMAND,
    )


def check_bark(key: str = "bark", name: str = "Bark small (dialogue)") -> ModelWeightStatus:
    """Bark needs its model files and the presets for both dialogue speakers."""
    from bark_engine import BARK_FILES, BARK_REPO, DIALOGUE_VOICES, PRESET_PARTS

    # A Bark speaker preset is several .npy arrays, one per part. Build the
    # cache path of every part of every dialogue voice; a nested comprehension
    # reads like two nested for-loops, outer one first.
    presets = [
        f"speaker_embeddings/{voice}_{part}.npy"
        for voice in DIALOGUE_VOICES.values()
        for part in PRESET_PARTS
    ]
    return check_hf_files(key, name, BARK_REPO, [*BARK_FILES, *presets])


def audit_model_weights() -> List[ModelWeightStatus]:
    """
    Audit every model ``VoiceEngineRouter.select_model`` can return.

    Returns one status per model, in a fixed order, each missing one carrying
    a ``fix``. Filesystem checks only, so the router can afford to call it
    before every synthesis (``require_weights``).
    """
    # The keys here must match the router's model keys: require_weights looks
    # a model up by key, and a key the audit does not know is let through.
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
        check_openvoice(),
        check_bark(),
        check_audioseal(),
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
    # Normalise to a list. A str is itself iterable (character by character),
    # so it has to be caught before the generic "iterable of paths" branch.
    candidates = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    # Order matters: the first acceptable file wins, which is how the GAN
    # Wav2Lip checkpoint is preferred over the plain one.
    for candidate in candidates:
        try:
            size = candidate.stat().st_size if candidate.is_file() else 0
        except OSError:
            # Unreadable (permissions, a broken mount) counts as absent.
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
    # Nothing passed. Tell "exists but too small" apart from "never fetched",
    # because the fix differs: a truncated file must be deleted first.
    truncated = [c for c in candidates if c.is_file()]
    if truncated:
        detail = (
            f"{truncated[0].name} is only {truncated[0].stat().st_size} bytes - "
            f"a truncated download. Delete it and re-run: "
            f"{VISION_FETCH_COMMAND} --only {key}"
        )
    else:
        # A caller-supplied message wins, e.g. Wav2Lip's, which must mention
        # that a human has to accept its licence before it is fetched.
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
    # Rebuild the absent status with a longer detail that names the features
    # lost and the vision fetch command. check_hf_repo's own detail is generic.
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
    # Each min_bytes is a floor well below the real file size and well above an
    # HTML error page, scaled to the model: Wav2Lip's checkpoint is hundreds of
    # MB, the selfie segmenter only a few hundred KB (hence the 100 KB default).
    return [
        check_videoseal(),
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
        check_sadtalker(),
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
        check_local_file(
            "realesrgan",
            "Real-ESRGAN general x4v3 (super-resolution for 1080p renders)",
            SUPER_RESOLUTION_MODEL,
            min_bytes=4_000_000,
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
    # Callers (and tests) may pass statuses they already have; otherwise audit
    # now. "is not None" rather than "or", so an empty list is respected.
    statuses = statuses if statuses is not None else audit_model_weights()
    present = [s for s in statuses if s.present]
    missing = [s for s in statuses if not s.present]

    # %-style arguments, not an f-string: logging only formats the message if
    # the line is actually emitted at this level.
    logger.info(
        "Model weight audit: %d/%d available (%s)",
        len(present),
        len(statuses),
        ", ".join(f"{s.key} {s.size_label}" for s in present) or "none",
    )
    # One WARNING per missing model, in capitals, so it stands out in the
    # startup output (golden rule 1: no silent fallbacks).
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
    """
    Log the vision audit at startup; one warning per missing model.

    Returns the statuses so the caller (``app.py``'s startup hook) can print
    them without auditing a second time.
    """
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
    """
    Counts, missing keys and per-model rows, the shape ``/health`` serves.

    ``missing`` is listed separately so a client can test ``available ==
    total`` or read the gaps without walking every row.
    """
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
    """
    The audit projected for ``/health``.

    Same shape as ``_summarise`` builds, written out inline. The audit runs
    fresh on each call, so weights fetched while the server runs show up
    without a restart.
    """
    statuses = audit_model_weights()
    return {
        "available": sum(1 for s in statuses if s.present),
        "total": len(statuses),
        "missing": [s.key for s in statuses if not s.present],
        "models": [s.to_dict() for s in statuses],
    }
