"""
Inaudible watermark for generated speech (R-32).

Every clip this platform synthesises carries a short watermark that a listener
cannot hear and that survives ordinary handling (re-encoding to AAC inside an
MP4, resampling, mild noise, trimming). A detector can tell whether an audio
file, or the soundtrack of a video, came from here.

The method is Meta's open-source AudioSeal (MIT code and weights): a neural
generator adds a low-level signal shaped by the speech it is hidden in, and a
detector reports, for the clip, the probability that a mark is present plus 16
message bits. We embed a 16-bit **platform tag** derived from a secret key
(HMAC of a fixed label), so a mark only counts if its bits match that tag: a
clip carrying some *other* AudioSeal user's mark is detected as watermarked but
reported as "not ours".

What a watermark proves and what it does not. It says "this audio came out of a
system that holds our key". It does not say which clip, who asked for it or
under what consent; that is the signed manifest's job (``manifest.py``), and the
manifest binds itself to the audio's hash. A determined attacker can degrade a
watermark (heavy re-synthesis, strong noise, time-stretching); the measured
limits are in ``docs/12-PROGRESS.md``, not claimed here.

Nothing downloads at run time: the two checkpoints (~94 MB) are fetched once by
``scripts/fetch_models.py --only audioseal`` and loaded from the local cache.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIOSEAL_REPO = "facebook/audioseal"
GENERATOR_FILE = "generator_base.pth"
DETECTOR_FILE = "detector_base.pth"
AUDIOSEAL_FILES = (GENERATOR_FILE, DETECTOR_FILE)
AUDIOSEAL_PIP = "pip install --no-deps audioseal==0.2.0 omegaconf"
MESSAGE_BITS = 16
MODEL_RATE = 16000  # AudioSeal's working rate; other rates are resampled around it

# A clip counts as ours when AudioSeal is at least this sure a mark is present AND at least this many
# of the 16 bits equal our tag. 14/16 tolerates one or two flipped bits from re-encoding while a random
# 16-bit message matches that well with probability (C(16,14)+C(16,15)+1)/2^16 = 137/65536 = 0.2%.
MIN_PROBABILITY = 0.5
MIN_BITS_MATCHING = 14
# The strength the generator's output is added at. 1.0 is the model's own calibration.
DEFAULT_ALPHA = 1.0

METHOD = "audioseal-16bit (facebook/audioseal, MIT): generator_base + detector_base, 16 kHz internal rate"


class WatermarkUnavailable(RuntimeError):
    """The watermark model cannot run; the message names the fix."""


# --------------------------------------------------------------------------- keys
_key_lock = threading.Lock()
_cached_key: Optional[bytes] = None


def signing_key() -> bytes:
    """
    The platform's secret (watermark tag and manifest signing).

    ``WATERMARK_KEY`` from the environment when set (what a deployment does).
    Otherwise a random key is created once and kept in ``outputs/.provenance-key``
    (mode 0600) and the log says so loudly: it works for local use, and anything
    marked with it can only be verified on a machine that has the file.
    """
    global _cached_key
    with _key_lock:
        if _cached_key is not None:
            return _cached_key
        configured = os.getenv("WATERMARK_KEY", "").strip()
        if configured:
            _cached_key = configured.encode("utf-8")
            return _cached_key
        path = Path(os.getenv("PROVENANCE_KEY_FILE", str(PROJECT_ROOT / "outputs" / ".provenance-key")))
        if path.is_file():
            _cached_key = path.read_bytes().strip()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            fresh = secrets.token_hex(32).encode("ascii")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(fresh)
            logger.warning(
                "WATERMARK_KEY is not set: created a local provenance key at %s. Marks and manifests made "
                "with it can only be verified where this file exists; set WATERMARK_KEY for a deployment.",
                path,
            )
            _cached_key = fresh
        return _cached_key


def reset_key_cache() -> None:
    """Forget the cached key (tests, and after rotating ``WATERMARK_KEY``)."""
    global _cached_key
    with _key_lock:
        _cached_key = None


def platform_tag(key: Optional[bytes] = None) -> List[int]:
    """The 16 message bits every clip of ours carries: the first 2 bytes of HMAC(key, label)."""
    digest = hmac.new(key or signing_key(), b"avatar-platform/audio-watermark/v1", hashlib.sha256).digest()
    return [(digest[i // 8] >> (7 - i % 8)) & 1 for i in range(MESSAGE_BITS)]


# --------------------------------------------------------------------------- report
@dataclass
class WatermarkReport:
    """What the detector found in one clip."""

    detected: bool            # a mark of ours: probability high enough AND the tag bits match
    probability: float        # AudioSeal's confidence that *some* mark is present (0..1)
    bits_matching: int        # of 16, how many equal our tag
    method: str = METHOD
    seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "probability": round(self.probability, 4),
            "bitsMatching": self.bits_matching,
            "bitsRequired": MIN_BITS_MATCHING,
            "method": self.method,
            "seconds": round(self.seconds, 2),
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- engine
class AudioWatermarker:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or "cpu"  # the models are small and fast enough on a CPU; the GPU is for the big ones
        self._generator: Any = None
        self._detector: Any = None
        self._failure: Optional[str] = None
        self._lock = threading.Lock()

    @staticmethod
    def weights_dir() -> Path:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(AUDIOSEAL_REPO, allow_patterns=list(AUDIOSEAL_FILES), local_files_only=True))

    def _load(self) -> None:
        if self._generator is not None and self._detector is not None:
            return
        if self._failure is not None:
            raise WatermarkUnavailable(self._failure)
        try:
            import torch

            # AudioSeal's convolution blocks are wrapped in torch.compile, which recompiled for
            # every new clip length: a 5 s clip took 42 s to mark the first time. Eager mode costs
            # a fraction of a second per clip and has no warm-up.
            torch._dynamo.config.disable = True
            from audioseal import AudioSeal

            snapshot = self.weights_dir()
            generator = AudioSeal.load_generator(str(snapshot / GENERATOR_FILE), nbits=MESSAGE_BITS, device=torch.device(self.device))
            detector = AudioSeal.load_detector(str(snapshot / DETECTOR_FILE), nbits=MESSAGE_BITS, device=torch.device(self.device))
            generator.eval()
            detector.eval()
            self._generator, self._detector = generator, detector
            logger.info("Loaded AudioSeal watermark generator and detector on %s", self.device)
        except Exception as exc:  # noqa: BLE001 - cached; the message names the fix
            self._failure = (
                f"The audio watermark model could not be loaded ({type(exc).__name__}: {exc}). "
                f"Fetch it with: python scripts/fetch_models.py --only audioseal (needs: {AUDIOSEAL_PIP})"
            )
            raise WatermarkUnavailable(self._failure) from exc

    def release(self) -> None:
        self._generator = self._detector = None

    # -- embedding -----------------------------------------------------------

    def embed(self, audio: np.ndarray, sample_rate: int, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
        """
        Return ``audio`` with the platform tag hidden in it (same length, same rate, float32).

        The watermark is made at 16 kHz and added at the clip's own rate. If the sum would pass full
        scale the whole clip is scaled down to fit (a fraction of a dB), rather than clipped.
        """
        import torch

        signal = _mono(audio)
        if signal.size == 0:
            raise ValueError("cannot watermark an empty clip")
        # AudioSeal 0.2+ does not resample for you: it must be fed 16 kHz. So the mark is made at 16 kHz
        # from a 16 kHz copy, brought back to the clip's rate, and added to the original untouched samples.
        model_input = _resample(signal, sample_rate, MODEL_RATE)
        with self._lock:
            self._load()
            tag = torch.tensor([platform_tag()], dtype=torch.int32, device=self.device)
            wave = torch.from_numpy(model_input).to(self.device)[None, None, :]
            with torch.inference_mode():
                mark = self._generator.get_watermark(wave, sample_rate=MODEL_RATE, message=tag)
        mark = _resample(mark.squeeze().detach().cpu().numpy().astype(np.float32), MODEL_RATE, sample_rate)
        mark = _fit(mark, signal.shape[0])
        out = signal + alpha * mark
        peak = float(np.max(np.abs(out)))
        if peak > 1.0:
            out = out / peak
        return out.astype(np.float32)

    # -- detection -----------------------------------------------------------

    def detect(self, audio: np.ndarray, sample_rate: int) -> WatermarkReport:
        """Is this clip marked by us? Never raises for a clip too short or silent: it says so."""
        import time

        import torch

        started = time.perf_counter()
        signal = _mono(audio)
        warnings: List[str] = []
        if signal.size < sample_rate * 0.5:
            warnings.append("the clip is shorter than half a second, so the result is unreliable")
        if signal.size == 0 or float(np.max(np.abs(signal))) < 1e-4:
            return WatermarkReport(False, 0.0, 0, seconds=time.perf_counter() - started, warnings=warnings + ["the clip is silent"])
        model_input = _resample(signal, sample_rate, MODEL_RATE)
        with self._lock:
            self._load()
            wave = torch.from_numpy(model_input).to(self.device)[None, None, :]
            with torch.inference_mode():
                probability, message = self._detector.detect_watermark(wave, sample_rate=MODEL_RATE)
        probability = float(probability if not hasattr(probability, "item") else probability.item())
        bits = [int(round(float(b))) for b in message.squeeze().detach().cpu().tolist()]
        matching = sum(1 for got, want in zip(bits, platform_tag(), strict=False) if got == want)
        detected = probability >= MIN_PROBABILITY and matching >= MIN_BITS_MATCHING
        return WatermarkReport(detected, probability, matching, seconds=time.perf_counter() - started, warnings=warnings)


def _resample(signal: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Polyphase resampling (anti-aliased); a no-op when the rates already match."""
    if from_rate == to_rate:
        return signal.astype(np.float32, copy=False)
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(int(from_rate), int(to_rate))
    return resample_poly(signal, int(to_rate) // divisor, int(from_rate) // divisor).astype(np.float32)


def _fit(signal: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad to exactly ``length`` samples (resampling can be off by one)."""
    if signal.shape[0] >= length:
        return signal[:length]
    return np.pad(signal, (0, length - signal.shape[0]))


def _mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    return array.mean(axis=1) if array.ndim > 1 else array


_shared: Optional[AudioWatermarker] = None
_shared_lock = threading.Lock()


def shared_watermarker() -> AudioWatermarker:
    """The process-wide watermarker (one pair of small models, loaded once)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = AudioWatermarker()
        return _shared
