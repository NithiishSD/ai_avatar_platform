"""
Voice-to-avatar (R-40, T8.2): the caller brings the audio, the platform brings the face.

A recording someone uploads is a person's voice, so it needs a stated consent basis, like a clone
reference. To move the mouth the renderer needs phoneme timing, which needs the words: if the
caller does not send a transcript, it is produced by open speech recognition (Whisper base, MIT
licence, through ``transformers``), and then force-aligned like any synthesised clip. What came
from where is written down: the speech record (and so the video's manifest) says the audio was
**supplied**, under which basis, and whether the words came from the caller or from ASR; the
audit trail gets an ``audio_supplied`` entry. The supplied audio is not watermarked: the mark
says "made here", and this voice was not; the rendered video carries the video mark.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import audit_log
import manifest
import provenance

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPLIED_DIR = PROJECT_ROOT / "outputs" / "supplied"
ASR_MODEL = "openai/whisper-base"
ASR_RATE = 16000
OUTPUT_RATE = 24000
MAX_SECONDS = 120.0      # a longer upload is refused, not truncated
MIN_SECONDS = 0.5


class VoiceToAvatarError(ValueError):
    """The upload cannot be used; the message says why and what to send instead."""


@dataclass
class SuppliedSpeech:
    audio_path: Path
    duration_seconds: float
    transcript: str
    transcript_source: str          # "client" or "asr"
    language: str
    phoneme_timestamps: List[Dict[str, Any]]
    alignment_method: Optional[str]
    sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audioUrl": f"/outputs/supplied/{self.audio_path.name}", "durationSeconds": round(self.duration_seconds, 3),
            "transcript": self.transcript, "transcriptSource": self.transcript_source, "language": self.language,
            "alignmentMethod": self.alignment_method, "phonemeCount": len(self.phoneme_timestamps), "sha256": self.sha256,
        }


class Transcriber:
    """Whisper base, loaded on first use and kept; speech recognition for uploads without a transcript."""

    def __init__(self, model_name: str = ASR_MODEL) -> None:
        self.model_name = model_name
        self._lock = threading.Lock()
        self._processor: Any = None
        self._model: Any = None

    def _load(self) -> Tuple[Any, Any]:
        if self._model is None:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            self._processor = WhisperProcessor.from_pretrained(self.model_name)
            # Same suppression as mms_engine / bark_engine: transformers' lazy-import stubs type the class as possibly None.
            self._model = WhisperForConditionalGeneration.from_pretrained(self.model_name)  # pyrefly: ignore[not-callable]
            self._model.eval()
            logger.info("Loaded %s for speech recognition", self.model_name)
        return self._processor, self._model

    def transcribe(self, audio16k: np.ndarray, language: Optional[str] = None) -> Tuple[str, str]:
        """``(text, language)``. Whisper reads 30 s at a time, so longer audio is transcribed window by window."""
        import torch

        with self._lock:
            processor, model = self._load()
            windows = [audio16k[i : i + 30 * ASR_RATE] for i in range(0, len(audio16k), 30 * ASR_RATE)]
            features = [processor(w, sampling_rate=ASR_RATE, return_tensors="pt").input_features for w in windows]
            if language is None:
                with torch.no_grad():
                    token = model.detect_language(features[0])
                found = re.search(r"<\|([a-z]{2,3})\|>", processor.tokenizer.decode(token[0]))
                language = found.group(1) if found else "en"
            pieces = []
            for feats in features:
                with torch.no_grad():
                    ids = model.generate(feats, language=language, task="transcribe", max_new_tokens=440)
                pieces.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
        return " ".join(p for p in pieces if p).strip(), language


_transcriber: Optional[Transcriber] = None


def shared_transcriber() -> Transcriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber


def prepare(
    upload: Path,
    consent_basis: str,
    transcript: Optional[str] = None,
    language: Optional[str] = None,
    transcriber: Optional[Transcriber] = None,
    aligner: Any = None,
) -> SuppliedSpeech:
    """
    Turn an uploaded recording into audio the renderer can use: decoded to 24 kHz mono WAV under
    ``outputs/supplied/``, transcribed if needed, force-aligned, with a speech record saying it was
    supplied, and an ``audio_supplied`` audit entry. Raises ``VoiceToAvatarError`` with the reason.
    """
    import soundfile as sf

    import video_io

    if consent_basis not in provenance.CONSENT_BASES:
        raise VoiceToAvatarError(f"consentBasis must be one of {sorted(provenance.CONSENT_BASES)}: this is a person's voice")
    try:
        audio = video_io.read_audio(upload, OUTPUT_RATE)
    except Exception as err:  # noqa: BLE001 - any decode failure is the caller's file
        raise VoiceToAvatarError("could not read the audio: send a WAV, MP3, M4A, OGG or FLAC recording") from err
    seconds = len(audio) / OUTPUT_RATE
    if seconds < MIN_SECONDS:
        raise VoiceToAvatarError(f"the recording is {seconds:.2f} s long; send at least {MIN_SECONDS} s of speech")
    if seconds > MAX_SECONDS:
        raise VoiceToAvatarError(f"the recording is {seconds:.0f} s long; the limit is {MAX_SECONDS:.0f} s, so split it")
    if float(np.sqrt(np.mean(audio**2))) < 1e-4:
        raise VoiceToAvatarError("the recording is silent; there is no speech to move the mouth")

    SUPPLIED_DIR.mkdir(parents=True, exist_ok=True)
    path = SUPPLIED_DIR / f"supplied-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.wav"
    sf.write(path, audio, OUTPUT_RATE, subtype="PCM_16")

    source = "client"
    words = (transcript or "").strip()
    if not words:
        from scipy.signal import resample_poly

        words, language = (transcriber or shared_transcriber()).transcribe(
            resample_poly(audio, ASR_RATE, OUTPUT_RATE).astype(np.float32), language)
        source = "asr"
        if not words:
            path.unlink(missing_ok=True)
            raise VoiceToAvatarError("speech recognition found no words; send the transcript with the audio")
    language = language or "en"

    if aligner is None:
        from alignment_engine import ForcedAligner

        aligner = ForcedAligner()
    stamps = aligner.align(audio_path_or_tensor=str(path), transcript=words, sample_rate=OUTPUT_RATE, language=language)
    timestamps = [t.model_dump(by_alias=True) for t in stamps]
    timestamps = [t for t in timestamps if t["endMs"] <= seconds * 1000]
    if not timestamps:
        path.unlink(missing_ok=True)
        raise VoiceToAvatarError("the words could not be aligned to the audio; check the transcript and the language")

    digest = manifest.sha256_file(path)
    origin = {"type": "supplied", "consentBasis": consent_basis, "transcriptSource": source,
              **({"asrModel": ASR_MODEL} if source == "asr" else {})}
    manifest.write_speech_record(
        path, model="supplied", mode="supplied", language=language, speaker_wav=None, clone_engine=None, emotion=None,
        alignment_method=getattr(aligner, "last_method", None), duration_seconds=seconds,
        watermark={"applied": False, "reason": "supplied audio is not marked as made here; the rendered video carries the video mark"},
        origin=origin,
    )
    audit_log.shared_audit().record("audio_supplied", subject=digest, basis=consent_basis, transcript=source,
                                    language=language, seconds=round(seconds, 2))
    return SuppliedSpeech(path, seconds, words, source, language, timestamps, getattr(aligner, "last_method", None), digest)
