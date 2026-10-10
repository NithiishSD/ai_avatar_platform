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

Concepts used here, explained once:

**Keyed tag.** AudioSeal can hide any 16-bit message. Anyone can run AudioSeal,
so "a mark is present" alone proves little. The message we hide is computed from
our secret key, so only a holder of the key can produce, or check for, our bits.

**Resampling to 16 kHz.** The sample rate is how many numbers per second describe
the sound (Kokoro speaks at 24 kHz, for example). AudioSeal was trained on 16 kHz
audio only, so every clip is converted to 16 kHz before it goes into the model,
and the watermark the model returns is converted back to the clip's own rate.

**Bit matching.** The detector returns its best guess for each of the 16 bits.
Re-encoding can flip one or two, so instead of demanding an exact match we count
how many bits agree with our tag and accept 14 or more (see the arithmetic at
``MIN_BITS_MATCHING``).
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
# The Hugging Face repository the two checkpoints come from, and their file names inside it.
AUDIOSEAL_REPO = "facebook/audioseal"
GENERATOR_FILE = "generator_base.pth"
DETECTOR_FILE = "detector_base.pth"
AUDIOSEAL_FILES = (GENERATOR_FILE, DETECTOR_FILE)
# Quoted in the "could not be loaded" error, so the message carries its own fix (golden rule 7).
AUDIOSEAL_PIP = "pip install --no-deps audioseal==0.2.0 omegaconf antlr4-python3-runtime==4.9.3"
MESSAGE_BITS = 16
MODEL_RATE = 16000  # AudioSeal's working rate; other rates are resampled around it

# A clip counts as ours when AudioSeal is at least this sure a mark is present AND at least this many
# of the 16 bits equal our tag. 14/16 tolerates one or two flipped bits from re-encoding while a random
# 16-bit message matches that well with probability (C(16,14)+C(16,15)+1)/2^16 = 137/65536 = 0.2%.
MIN_PROBABILITY = 0.5
MIN_BITS_MATCHING = 14
# The strength the generator's output is added at. 1.0 is the model's own calibration.
DEFAULT_ALPHA = 1.0

# Copied into every report, so each detection result records how it was obtained (golden rule 2).
METHOD = "audioseal-16bit (facebook/audioseal, MIT): generator_base + detector_base, 16 kHz internal rate"


def enabled() -> bool:
    """One switch for both marks (audio and video): ``WATERMARK_ENABLED=false`` turns them off, visibly."""
    # Read on every call rather than once at import, so changing the variable takes effect at once.
    # Anything not in the "off" list, the default included, means on.
    return os.getenv("WATERMARK_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")


class WatermarkUnavailable(RuntimeError):
    """The watermark model cannot run; the message names the fix."""


# --------------------------------------------------------------------------- keys
# The key is created at most once per process. The lock stops two threads that both find no key
# from each generating a different one at the same moment.
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
    # "global" lets this function assign the module-level cache instead of a new local variable.
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
            # secrets (not random) is the standard library's source for unguessable values.
            # token_hex(32) gives 32 random bytes as 64 hex characters.
            fresh = secrets.token_hex(32).encode("ascii")
            # O_EXCL fails if the file already exists, and 0o600 makes it readable by the owner only;
            # os.open sets that mode at creation, so the key is never briefly world-readable.
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
    # HMAC mixes the secret key into a hash, so the result cannot be computed without the key.
    # The fixed label (with a version) keeps this use of the key separate from manifest signing.
    digest = hmac.new(key or signing_key(), b"avatar-platform/audio-watermark/v1", hashlib.sha256).digest()
    # Unpack bytes into bits, most significant bit first: bit i lives in byte i // 8, and
    # shifting right by (7 - i % 8) then masking with & 1 isolates it.
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
        """The camelCase JSON shape returned by the API, with the threshold that was applied."""
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
    """
    Loads AudioSeal lazily and hides or reads the platform tag in a clip.

    The generator and detector load on first use, not at construction, so importing this module or
    creating the object costs nothing (golden rule 5: lazy load). One lock serialises loading and
    model calls, so two threads sharing the process-wide instance never load the models twice.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        # GPU when there is one (AVATAR_DEVICE overrides). It was CPU-only on the theory that a small model
        # is cheap anywhere; on the owner's GPU host it still cost 2.2 s of a 2.7 s synthesis (4x Kokoro).
        self.device = device or _default_device()
        import gpu_utils

        # Registering a releaser lets gpu_utils unload these models when a bigger one needs the card.
        gpu_utils.register_releaser("audio-watermark", self.release)
        self._generator: Any = None
        self._detector: Any = None
        # A load failure is remembered, so later calls fail fast with the same message instead of
        # retrying a slow load that will fail again.
        self._failure: Optional[str] = None
        self._lock = threading.Lock()

    @staticmethod
    def weights_dir() -> Path:
        """The local folder holding both checkpoints; raises if they were never fetched."""
        from huggingface_hub import snapshot_download

        # local_files_only=True: look in the cache and never download (models are fetched by script).
        return Path(snapshot_download(AUDIOSEAL_REPO, allow_patterns=list(AUDIOSEAL_FILES), local_files_only=True))

    def _load(self) -> None:
        """Load both models once; raise ``WatermarkUnavailable`` with the fix when that fails."""
        if self._generator is not None and self._detector is not None:
            return
        if self._failure is not None:
            raise WatermarkUnavailable(self._failure)
        if self.device == "cuda":
            import gpu_utils

            # Room first; a full card is a passing condition, so it is not cached as a permanent failure.
            try:
                gpu_utils.ensure_vram(VRAM_MB, "AudioSeal audio watermark", keep="audio-watermark")
            except gpu_utils.InsufficientVRAM as err:
                raise WatermarkUnavailable(str(err)) from err
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
            # eval() switches layers such as dropout to inference behaviour; nothing is trained here.
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
        """Drop both models so other models can use the GPU; the next use reloads them."""
        import gpu_utils

        # Dropping the references lets Python free the tensors; empty_cache then hands the memory back.
        self._generator = self._detector = None
        gpu_utils.empty_cache()

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
        import gpu_utils

        def make() -> Any:
            """Run the generator on the 16 kHz copy; returns the watermark signal alone."""
            with self._lock:
                self._load()
                # Shape (1, 16): a batch of one message.
                tag = torch.tensor([platform_tag()], dtype=torch.int32, device=self.device)
                # [None, None, :] adds two axes: AudioSeal expects (batch, channels, samples).
                wave = torch.from_numpy(model_input).to(self.device)[None, None, :]
                # inference_mode turns off gradient tracking, which saves memory and time.
                with torch.inference_mode():
                    return self._generator.get_watermark(wave, sample_rate=MODEL_RATE, message=tag)

        # A big model still on the GPU can leave too little room: free it and retry once.
        mark = gpu_utils.retry_after_freeing(make, keep="audio-watermark")
        mark = _resample(mark.squeeze().detach().cpu().numpy().astype(np.float32), MODEL_RATE, sample_rate)
        # The round trip through 16 kHz can change the length by a sample, so match it exactly.
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
        # signal.size is a sample count, so sample_rate * 0.5 is half a second of audio.
        if signal.size < sample_rate * 0.5:
            warnings.append("the clip is shorter than half a second, so the result is unreliable")
        # A peak below 1e-4 (about -80 dBFS) is treated as silence: there is nothing to read.
        if signal.size == 0 or float(np.max(np.abs(signal))) < 1e-4:
            return WatermarkReport(False, 0.0, 0, seconds=time.perf_counter() - started, warnings=warnings + ["the clip is silent"])
        model_input = _resample(signal, sample_rate, MODEL_RATE)
        import gpu_utils

        def read() -> Any:
            """Run the detector on the 16 kHz copy; returns (probability, message bits)."""
            with self._lock:
                self._load()
                wave = torch.from_numpy(model_input).to(self.device)[None, None, :]
                with torch.inference_mode():
                    return self._detector.detect_watermark(wave, sample_rate=MODEL_RATE)

        probability, message = gpu_utils.retry_after_freeing(read, keep="audio-watermark")
        # The probability may come back as a tensor or a plain number; .item() unwraps a tensor.
        probability = float(probability if not hasattr(probability, "item") else probability.item())
        # Round each bit estimate to 0 or 1, then count how many agree with our tag.
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

    # resample_poly upsamples by `up` then downsamples by `down`. Dividing both rates by their greatest
    # common divisor gives the smallest such pair, e.g. 24000 -> 16000 becomes up=2, down=3.
    divisor = gcd(int(from_rate), int(to_rate))
    return resample_poly(signal, int(to_rate) // divisor, int(from_rate) // divisor).astype(np.float32)


def _fit(signal: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad to exactly ``length`` samples (resampling can be off by one)."""
    if signal.shape[0] >= length:
        return signal[:length]
    return np.pad(signal, (0, length - signal.shape[0]))


def _mono(audio: np.ndarray) -> np.ndarray:
    """Float32 mono: a (samples, channels) array is averaged across its channels."""
    array = np.asarray(audio, dtype=np.float32)
    return array.mean(axis=1) if array.ndim > 1 else array


# One watermarker for the whole process, created on first use by shared_watermarker().
_shared: Optional[AudioWatermarker] = None
_shared_lock = threading.Lock()


# GPU memory to free before loading AudioSeal's generator and detector (93 MB of weights) with headroom.
VRAM_MB = 400


def _default_device() -> str:
    """``AVATAR_DEVICE`` if set, else CUDA when PyTorch sees it, else the CPU (the rule every other model here follows)."""
    import torch

    import gpu_utils

    return gpu_utils.preferred_device() or ("cuda" if torch.cuda.is_available() else "cpu")


def shared_watermarker() -> AudioWatermarker:
    """The process-wide watermarker (one pair of small models, loaded once)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = AudioWatermarker()
        return _shared
