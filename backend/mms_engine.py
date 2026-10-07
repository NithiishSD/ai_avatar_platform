"""
Facebook MMS-TTS engine (Phase 3 — 1000+ language synthesis).

Each MMS-TTS language is a separate VITS checkpoint (``facebook/mms-tts-<iso3>``)
of roughly 145 MB. Loading one per request would thrash a 6 GB GPU, so this
module keeps a small LRU cache of loaded checkpoints and evicts the least
recently used one when the cache is full.

Checkpoints for non-Latin scripts are "uroman" models: their tokenizer expects
romanized input. ``MMSTTSEngine`` detects that via ``tokenizer.is_uroman`` and
romanizes with ``uroman`` when it is installed, otherwise it raises a message
that names the missing package rather than emitting garbled audio.
"""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from language_registry import LanguageInfo, resolve
from romanizer import as_text, get_romanizer

logger = logging.getLogger(__name__)

# How many VITS checkpoints stay resident. Each is ~145 MB of weights, so 3
# fits comfortably alongside Kokoro/XTTS on a 6 GB card.
DEFAULT_CACHE_SIZE = 3


class MMSLanguageNotSupported(RuntimeError):
    """Raised when MMS-TTS has no checkpoint for the requested language."""


class MMSRomanizationRequired(RuntimeError):
    """Raised when a uroman checkpoint is selected but no romanizer is available."""


@dataclass(frozen=True)
class MMSSynthesisResult:
    output_path: str
    sample_rate: int
    duration_seconds: float
    model_id: str
    language: LanguageInfo
    romanized: bool


class MMSTTSEngine:
    """Lazy, LRU-cached multilingual VITS synthesizer."""

    def __init__(self, device: Optional[str] = None, cache_size: int = DEFAULT_CACHE_SIZE):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_size = max(1, cache_size)
        # iso3 -> (model, tokenizer)
        # Any, not object: these are a transformers VitsModel and tokenizer,
        # whose attributes (config, speaking_rate, ...) we use directly.
        self._cache: "OrderedDict[str, Tuple[Any, Any]]" = OrderedDict()
        self._failed: dict[str, str] = {}
        self._lock = threading.Lock()
        self._uroman = None
        self._uroman_checked = False

    # ------------------------------------------------------------------
    # Capability probes
    # ------------------------------------------------------------------

    def supports(self, language: str) -> bool:
        return resolve(language).mms_supported

    def loaded_languages(self) -> list[str]:
        return list(self._cache.keys())

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self, info: LanguageInfo):
        """Return (model, tokenizer) for a language, loading and caching on miss."""
        iso3 = info.iso3
        with self._lock:
            if iso3 in self._cache:
                self._cache.move_to_end(iso3)
                return self._cache[iso3]
            if iso3 in self._failed:
                raise MMSLanguageNotSupported(
                    f"MMS-TTS checkpoint for '{info.name}' ({iso3}) previously failed to "
                    f"load: {self._failed[iso3]}"
                )

        from transformers import VitsModel, AutoTokenizer

        model_id = info.mms_model
        logger.info("Loading MMS-TTS %s (%s) on %s", model_id, info.name, self.device)
        try:
            # transformers types its lazily imported model classes as possibly
            # None; at runtime this is always the class.
            model = VitsModel.from_pretrained(model_id)  # pyrefly: ignore[not-callable]
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = model.to(self.device).eval()
        except Exception as exc:  # noqa: BLE001 - reported verbatim to the caller
            with self._lock:
                self._failed[iso3] = str(exc)
            raise MMSLanguageNotSupported(
                f"Could not load MMS-TTS checkpoint '{model_id}' for {info.name}: {exc}"
            ) from exc

        with self._lock:
            self._cache[iso3] = (model, tokenizer)
            self._cache.move_to_end(iso3)
            while len(self._cache) > self.cache_size:
                evicted_iso3, (evicted_model, _) = self._cache.popitem(last=False)
                logger.info("Evicting MMS-TTS checkpoint %s from cache", evicted_iso3)
                del evicted_model
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return self._cache[iso3]

    # ------------------------------------------------------------------
    # Romanization for uroman checkpoints
    # ------------------------------------------------------------------

    def _get_romanizer(self):
        """The shared process-wide romanizer; ``None`` when uroman is missing."""
        if self._uroman_checked:
            return self._uroman
        self._uroman_checked = True
        self._uroman = get_romanizer()
        return self._uroman

    def _romanize(self, text: str, info: LanguageInfo) -> str:
        romanizer = self._get_romanizer()
        if romanizer is None:
            # ASCII text needs no romanization even on a uroman checkpoint.
            if all(ord(ch) < 128 for ch in text):
                return text
            raise MMSRomanizationRequired(
                f"The MMS-TTS checkpoint for {info.name} ({info.iso3}) expects romanized "
                "input. Install the romanizer with `pip install uroman`, or send "
                "already-romanized text."
            )
        # The check sits outside the try: its own TypeError must not be taken
        # for the old-uroman signature fallback.
        try:
            result = romanizer.romanize_string(text, lcode=info.iso3)
        except TypeError:
            # Older uroman builds take no lcode keyword.
            result = romanizer.romanize_string(text)
        return as_text(result)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        language: str,
        output_path: Path,
        speaking_rate: Optional[float] = None,
        noise_scale: Optional[float] = None,
    ) -> MMSSynthesisResult:
        """
        Synthesize ``text`` in ``language`` and write a WAV to ``output_path``.

        ``speaking_rate`` and ``noise_scale`` map onto the VITS generator's own
        controls, so rate changes here are re-synthesized rather than
        time-stretched.
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        info = resolve(language)
        if not info.mms_supported:
            raise MMSLanguageNotSupported(
                f"MMS-TTS has no checkpoint for '{language}'"
                + (f" (resolved to ISO-639-3 '{info.iso3}')" if info.iso3 else "")
                + ". Use mode='high_quality' (Higgs) or mode='clone' (XTTS-v2) instead."
            )

        model, tokenizer = self._load(info)

        prepared = unicodedata.normalize("NFC", text.strip())
        romanized = False
        if getattr(tokenizer, "is_uroman", False):
            converted = self._romanize(prepared, info)
            romanized = converted != prepared
            prepared = converted

        if speaking_rate is not None:
            model.speaking_rate = float(speaking_rate)
        if noise_scale is not None:
            model.noise_scale = float(noise_scale)

        inputs = tokenizer(prepared, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}

        started = time.time()
        with torch.inference_mode():
            waveform = model(**inputs).waveform

        audio = waveform.squeeze().detach().float().cpu().numpy().astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        sample_rate = int(model.config.sampling_rate)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sample_rate)
        duration = len(audio) / sample_rate if sample_rate else 0.0

        logger.info(
            "MMS-TTS %s: %.2fs audio in %.0f ms (romanized=%s)",
            info.iso3,
            duration,
            (time.time() - started) * 1000,
            romanized,
        )

        return MMSSynthesisResult(
            output_path=str(output_path),
            sample_rate=sample_rate,
            duration_seconds=duration,
            model_id=info.mms_model or "",
            language=info,
            romanized=romanized,
        )

    def unload_all(self) -> None:
        """Drop every cached checkpoint and free GPU memory."""
        with self._lock:
            self._cache.clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()
