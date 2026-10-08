"""
Invisible watermark for rendered video (R-33).

Every frame of every rendered video carries a 256-bit message that a viewer
cannot see and that survives the H.264 encoding the video is saved with. Two
halves, each 128 bits:

    [ platform tag (128) | manifest id (128) ]

* The **platform tag** is HMAC(secret key, label). A video counts as ours when
  at least ``MIN_TAG_BITS`` of those 128 bits are read back correctly. A random
  video matches that many bits with probability below 1e-8 (computed in the
  tests), so this is a statement about the key, not a similarity score.
* The **manifest id** names the signed manifest that ships with the video
  (``manifest.py``). It is what lets a stripped or re-encoded copy be traced to
  its record. It is read with bit errors when a copy is heavily compressed, so a
  lookup matches it to the nearest known id, not to an exact one.

The method is Meta's open-source VideoSeal 1.0 (MIT, 57 M parameters), run on
the CPU. It embeds into every 4th frame and propagates to the ones in between,
in windows of 32 frames, so a video of any length is processed with bounded
memory, one window at a time, as it is encoded.

What it proves and what it does not. It says "this video was rendered by a
system holding our key". It does not by itself say what went into it or whether
there was consent; the signed manifest says that, and it binds to the exact file
by hash. Heavy compression (H.264 above about CRF 30), strong cropping or
re-synthesis can remove the mark; the measured limits are in
``docs/12-PROGRESS.md``.

Where it sits: ``render_engine.py`` wraps the frame stream in ``embed_stream`` just before encoding,
then reads the finished file back with ``detect_video`` as a self-check. ``authenticity.py`` calls
``detect_video`` when someone uploads a file to check.

Concepts used here:
  * *Bit accuracy* is the share of message bits read back correctly. The detector outputs one
    *logit* per bit (a score: positive means 1, negative means 0, larger means surer). Averaging
    logits over many frames before deciding makes single noisy frames matter less.
  * *Why a threshold on matching bits, not a perfect match*: compression flips some bits. Each bit of
    an unmarked video agrees with our tag by chance half the time, so the chance of 96 or more of 128
    agreeing is a binomial tail, computed by :func:`chance_of_matching`.
  * *Generators* (``yield``): ``embed_stream`` returns frames one at a time as the caller asks for
    them, so the whole video never has to sit in memory.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

import watermark_engine

logger = logging.getLogger(__name__)

# parents[1] of backend/video_watermark.py is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEOSEAL_DIR = PROJECT_ROOT / ".models" / "videoseal"
CHECKPOINT = VIDEOSEAL_DIR / "y_256b_img.pth"
ATTENUATION_FILE = VIDEOSEAL_DIR / "attenuation.yaml"
CARD_FILE = VIDEOSEAL_DIR / "videoseal_1.0.local.yaml"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth"
# Size and SHA-256 pin the exact file: the size is a cheap "download finished" check, the hash an
# integrity check that a truncated or swapped file cannot pass.
CHECKPOINT_BYTES = 228_351_471
CHECKPOINT_SHA256 = "3d2ff2523d2a89e3532c6dfdcf693098799326e9b3e74c185e815e9baa8340a3"
# --no-deps installs the package without its declared dependency pins; the packages it actually
# needs at runtime are listed after it, each chosen to fit this environment.
VIDEOSEAL_PIP = "pip install --no-deps videoseal==1.0.1 && pip install av lpips pytorch_msssim calflops decord pycocotools PyWavelets timm==0.9.16 'scikit-image<0.22' 'networkx<3'"
# The package's wheel omits this 14-line file (it names the channel counts of the "JND" attenuation
# map); it is written next to the checkpoint when the model is fetched or first loaded.
ATTENUATION_YAML = (
    "jnd_1_1:\n  in_channels: 1\n  out_channels: 1\n\n"
    "jnd_3_3:\n  in_channels: 3\n  out_channels: 3\n\n"
    "jnd_1_3:\n  in_channels: 1\n  out_channels: 3\n\n"
    "jnd_3_1:\n  in_channels: 3\n  out_channels: 1\n"
)

# How strongly the mark is added (VideoSeal's own default is 0.2). Measured 8 Oct on two kinds of clip:
# at 0.2 a textured clip lost the mark at H.264 CRF 23 and was only just readable at the renderer's own
# CRF 20 (117 of 128 tag bits); at 0.3 it reads 128/128 at CRF 20 and 102/128 at CRF 23, while the mark stays
# about 45 dB below the picture (PSNR 44.7-46.2 dB). 0.4 buys little more robustness for 2-3 dB of visibility.
# ``VIDEO_WATERMARK_STRENGTH`` overrides it.
STRENGTH = 0.3

# The checkpoint's capacity is 256 bits, split evenly between the tag and the manifest id.
MESSAGE_BITS = 256
TAG_BITS = 128
ID_BITS = 128
# A random video agrees with our tag on at least this many of 128 bits with probability ~6e-9.
MIN_TAG_BITS = 96
WINDOW = 32          # frames embedded per call; bounds memory for any video length
SAMPLE_FRAMES = 48   # frames read when detecting, spread evenly through the video

METHOD = "videoseal-1.0 (256-bit, MIT): 128-bit keyed platform tag + 128-bit manifest id, per-frame logits averaged over sampled frames"


class VideoWatermarkUnavailable(RuntimeError):
    """The video watermark model cannot run; the message names the fix."""


def platform_bits(key: Optional[bytes] = None) -> List[int]:
    """The 128 tag bits: the first 16 bytes of HMAC(key, label).

    HMAC means only a holder of the key can produce these bits, and the label keeps them different
    from anything else derived from the same key (the manifest signing key uses another label).
    """
    digest = hmac.new(key or watermark_engine.signing_key(), b"avatar-platform/video-watermark/v1", hashlib.sha256).digest()
    # Bit i is in byte i // 8; shifting by (7 - i % 8) reads bits most-significant first.
    return [(digest[i // 8] >> (7 - i % 8)) & 1 for i in range(TAG_BITS)]


def bits_of(data: bytes) -> List[int]:
    """Bytes to a list of 0/1 ints, most significant bit first (8 per byte)."""
    return [(byte >> (7 - i)) & 1 for byte in data for i in range(8)]


def bytes_of(bits: Sequence[int]) -> bytes:
    """The inverse of :func:`bits_of`: groups of 8 bits, most significant first, back to bytes."""
    return bytes(sum((int(bits[i + j]) & 1) << (7 - j) for j in range(8)) for i in range(0, len(bits), 8))


def chance_of_matching(matching: int = MIN_TAG_BITS, of: int = TAG_BITS) -> float:
    """Probability that ``of`` random fair bits agree with a fixed pattern in at least ``matching`` places."""
    # Binomial tail: the number of ways to get exactly k agreements, summed for k >= matching, over
    # all 2**of equally likely bit patterns. math.comb keeps it exact with Python's big integers.
    return sum(math.comb(of, k) for k in range(matching, of + 1)) / 2**of


@dataclass
class VideoWatermarkReport:
    """The result of reading a mark: whether our tag is present, how well, and the id if so."""

    detected: bool                  # the tag matched: a video rendered by us
    tag_bits_matching: int          # of 128
    bit_accuracy: float             # of the tag, 0..1
    manifest_id: Optional[str]      # 32 hex digits read from the mark (may carry bit errors), only when detected
    frames_analysed: int
    method: str = METHOD
    seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """The report as the API's camelCase JSON, with the threshold it was judged against."""
        return {
            "detected": self.detected,
            "tagBitsMatching": self.tag_bits_matching,
            "tagBitsRequired": MIN_TAG_BITS,
            "bitAccuracy": round(self.bit_accuracy, 4),
            "manifestId": self.manifest_id,
            "framesAnalysed": self.frames_analysed,
            "method": self.method,
            "seconds": round(self.seconds, 2),
            "warnings": self.warnings,
        }


class VideoWatermarker:
    """Lazy wrapper around VideoSeal: embeds into a frame stream and reads a mark back.

    A failed load is cached in ``_failure``, so later calls fail fast with the same fix-it message
    instead of retrying an expensive import on every render.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        # GPU when there is one (AVATAR_DEVICE overrides): the mark runs on every frame, and the
        # owner's RTX 4050 run showed it on the CPU cost 28 s of a 32 s render (8.5x the render itself).
        self.device = device or _default_device()
        self._model: Any = None
        self._failure: Optional[str] = None
        self._lock = threading.Lock()

    @staticmethod
    def files_present() -> bool:
        """The checkpoint is on disk at its full size and the attenuation file exists."""
        try:
            return CHECKPOINT.is_file() and CHECKPOINT.stat().st_size == CHECKPOINT_BYTES and ATTENUATION_FILE.is_file()
        except OSError:
            return False

    def _load(self) -> Any:
        """Load VideoSeal once; raises VideoWatermarkUnavailable (with the fetch command) if it cannot."""
        if self._model is not None:
            return self._model
        if self._failure is not None:
            raise VideoWatermarkUnavailable(self._failure)
        try:
            import torch
            import videoseal
            import yaml
            from videoseal.utils.cfg import setup_model_from_model_card

            if not self.files_present():
                raise FileNotFoundError(f"{CHECKPOINT} is missing or incomplete")
            if _sha256(CHECKPOINT) != CHECKPOINT_SHA256:  # presence is not integrity: checked once per process
                raise ValueError(f"{CHECKPOINT.name} does not match its pinned SHA-256")
            # The packaged card points at a URL and at repo-relative config paths; a local copy points at ours.
            card = yaml.safe_load((Path(videoseal.__file__).parent / "cards" / "videoseal_1.0.yaml").read_text())
            card["checkpoint_path"] = str(CHECKPOINT)
            card["args"]["attenuation_config"] = str(ATTENUATION_FILE)
            CARD_FILE.write_text(yaml.safe_dump(card))
            model = setup_model_from_model_card(CARD_FILE)
            # scaling_w is how strongly the mark is added to each frame (see STRENGTH above).
            model.blender.scaling_w = float(os.getenv("VIDEO_WATERMARK_STRENGTH", STRENGTH))
            model.eval()
            model.to(torch.device(self.device))
            self._model = model
            logger.info("Loaded VideoSeal 1.0 (256-bit) on %s", self.device)
            return model
        except Exception as exc:  # noqa: BLE001 - cached; the message names the fix
            self._failure = (
                f"The video watermark model could not be loaded ({type(exc).__name__}: {exc}). "
                f"Fetch it with: python scripts/fetch_vision_models.py --only videoseal (needs: {VIDEOSEAL_PIP})"
            )
            raise VideoWatermarkUnavailable(self._failure) from exc

    def release(self) -> None:
        """Drop the model reference so its memory can be reclaimed; the next use reloads it."""
        self._model = None

    # -- embedding -----------------------------------------------------------

    def message(self, manifest_id: bytes) -> "Any":
        """The 256-bit message for one video: tag then manifest id, as a (1, 256) float tensor."""
        import torch

        # The id must fill its 128-bit half exactly, or the tag and id would shift against each other.
        if len(manifest_id) != ID_BITS // 8:
            raise ValueError(f"the manifest id must be {ID_BITS // 8} bytes")
        return torch.tensor([platform_bits() + bits_of(manifest_id)], dtype=torch.float32, device=self.device)

    def embed_stream(self, frames: Iterable[np.ndarray], manifest_id: bytes) -> Iterator[np.ndarray]:
        """
        Mark RGB uint8 frames as they stream through, in windows of ``WINDOW``.

        Yields the marked frames in order, each the same shape and dtype as its input. At most one window
        is held in memory, so a long or high-resolution video costs time, not RAM.
        """
        window: List[np.ndarray] = []
        # Built on the first full window, so a stream that yields nothing never loads torch.
        message = None
        for frame in frames:
            window.append(frame)
            if len(window) == WINDOW:
                message = message if message is not None else self.message(manifest_id)
                yield from self._embed_window(window, message)
                window = []
        # The last, shorter window (a video rarely divides evenly by WINDOW).
        if window:
            message = message if message is not None else self.message(manifest_id)
            yield from self._embed_window(window, message)

    def _embed_window(self, window: List[np.ndarray], message: Any) -> Iterator[np.ndarray]:
        """Mark one window of frames in a single model call and yield them back as uint8 RGB."""
        import torch

        # (frames, H, W, 3) uint8 -> (frames, 3, H, W) float 0..1, the layout the model takes.
        batch = torch.from_numpy(np.stack(window)).permute(0, 3, 1, 2).float().div(255.0).to(self.device)
        with self._lock:
            model = self._load()
            # inference_mode: like no_grad, and a little faster, for code that never needs gradients.
            # is_video=True embeds into every 4th frame and propagates the mark to the frames between.
            with torch.inference_mode():
                marked = model.embed(batch, message, is_video=True, lowres_attenuation=True)["imgs_w"]
        out = (marked.clamp(0, 1) * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for original, frame in zip(window, out, strict=True):
            yield np.ascontiguousarray(frame.reshape(original.shape))

    # -- detection -----------------------------------------------------------

    def detect_frames(self, frames: Sequence[np.ndarray]) -> VideoWatermarkReport:
        """Read the mark from RGB uint8 frames (a sample of a video)."""
        import torch

        started = time.perf_counter()
        warnings: List[str] = []
        if len(frames) == 0:
            return VideoWatermarkReport(False, 0, 0.0, None, 0, warnings=["no frames to analyse"])
        # Fewer frames means fewer logits to average, so a noisy frame weighs more.
        if len(frames) < 8:
            warnings.append("fewer than 8 frames were available, so the result is less reliable")
        batch = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255.0).to(self.device)
        with self._lock:
            model = self._load()
            with torch.inference_mode():
                preds = model.detect(batch, is_video=True)["preds"]
        # Column 0 of preds is VideoSeal's detection bit (unused here); columns 1.. are the 256 message bits.
        logits = preds[:, 1:].float().mean(dim=0)              # one logit per message bit, averaged over frames
        bits = [int(v > 0) for v in logits.cpu().tolist()]
        tag_matching = sum(1 for got, want in zip(bits[:TAG_BITS], platform_bits(), strict=True) if got == want)
        detected = tag_matching >= MIN_TAG_BITS
        return VideoWatermarkReport(
            detected=detected,
            tag_bits_matching=tag_matching,
            bit_accuracy=tag_matching / TAG_BITS,
            # The id is only read when the tag says the mark is ours; otherwise its bits are noise.
            manifest_id=bytes_of(bits[TAG_BITS:]).hex() if detected else None,
            frames_analysed=len(frames),
            seconds=time.perf_counter() - started,
            warnings=warnings,
        )

    def detect_video(self, path: Path | str, sample_frames: int = SAMPLE_FRAMES) -> VideoWatermarkReport:
        """Sample frames evenly through an MP4 (or any ffmpeg-readable video) and read the mark."""
        import video_io

        info = video_io.probe(path)
        total = info.frame_count or 0
        # Read every step-th frame so the sample spans the whole video. When the frame count is unknown
        # step is 1, so the first ``sample_frames`` frames are taken instead.
        step = max(1, total // sample_frames) if total else 1
        sampled: List[np.ndarray] = []
        for index, frame in enumerate(video_io.read_frames(path)):
            if index % step == 0:
                # copy(): read_frames yields read-only views of ffmpeg's bytes; a copy is a normal array.
                sampled.append(frame.copy())
            if len(sampled) >= sample_frames:
                break
        return self.detect_frames(sampled)


def _sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in 4 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


_shared: Optional[VideoWatermarker] = None
_shared_lock = threading.Lock()


def _default_device() -> str:
    """``AVATAR_DEVICE`` if set, else CUDA when PyTorch sees it, else the CPU (the rule every other model here follows)."""
    import torch

    import gpu_utils

    return gpu_utils.preferred_device() or ("cuda" if torch.cuda.is_available() else "cpu")


def shared_video_watermarker() -> VideoWatermarker:
    """The process-wide watermarker, created on first use; locked so two threads cannot both create it."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = VideoWatermarker()
        return _shared
