"""
VoiceEngineRouter — Multi-Model TTS Orchestration (Phases 1-3)
==============================================================
Routing matrix, highest priority first:
  dialogue | style=dialogue | [S1]/[S2] → Dia-1.6B      (multi-speaker)
  clone                                 → XTTS-v2       (zero-shot cloning)
  high_quality | quality=high           → Higgs TTS 2   (MOS >4.0)
  multilingual, or non-English with an
    MMS-TTS checkpoint                  → MMS-TTS       (1077 languages)
  non-English without MMS coverage      → Higgs TTS 2
  fast + English                        → Kokoro-82M    (sub-second)

Fallbacks: Dia → Kokoro, Higgs → XTTS-v2, MMS → Higgs → XTTS-v2.

Phase 3 post-processing runs after whichever backend produced the audio:
emotion prosody (blended with the caller's speed/pitch in a single transform),
forced alignment, then the optional MOS/PESQ quality audit.
"""

import os
import threading
import time
import math
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf
import numpy as np

from audio_utils import validate_and_convert_for_cloning
import alignment_engine
import watermark_engine
from alignment_engine import ForcedAligner
import audit_log
import gpu_utils
import manifest
import language_registry
import model_registry
import provenance
from emotion_engine import EmotionProsodyEngine
from mms_engine import MMSTTSEngine, MMSRomanizationRequired
from openvoice_engine import OpenVoiceEngine
from bark_engine import BarkEngine, BarkUnavailable
from quality_auditor import SpeechQualityAuditor

logger = logging.getLogger(__name__)


class ModelWeightsMissing(RuntimeError):
    """
    The engine a request needs has no weights on disk.

    Raised *before* any loader runs. Without it, ``from_pretrained`` would
    have started a multi-gigabyte download in the middle of a request - the
    first time someone picked high quality or dialogue - while the startup log
    claimed such requests "fall back to another model". The message names the
    command that fixes it (golden rule 7).
    """

    def __init__(self, model_key: str, detail: str, fix: str = "") -> None:
        self.model_key = model_key
        fix = fix or f"{model_registry.MODEL_FETCH_COMMAND} --only {model_key}"
        # "Fetch it with" only when fetching is the fix: for an engine this
        # stack cannot run, that advice would cost gigabytes and change nothing.
        remedy = f"Fetch it with: {fix}" if fix.startswith(model_registry.MODEL_FETCH_COMMAND) else f"Fix: {fix}"
        super().__init__(f"{model_key} is not available: {detail}. {remedy}")


# Engines that condition on a reference recording - that is, clone a voice.
# Only for these does a supplied speaker_wav actually get used.
REFERENCE_ENGINES = frozenset({"xtts-v2", "higgs-tts-2", "openvoice-v2"})
# Peak process memory each engine adds on a CPU host, measured 8 Oct 2026 by
# loading it alone and synthesising once (ru_maxrss minus the bare router):
# XTTS-v2 4157 MiB, Bark 1840, OpenVoice V2 1637, Kokoro 1101, SD 1.5 6610.
# Rounded up; ``gpu_utils.ensure_host_memory`` unloads the others when short.
XTTS_RAM_MB = 4200
# Engines a caller may ask for in mode="clone"; the first is the default.
CLONE_ENGINES = ("xtts-v2", "openvoice-v2")


class VoiceConsentRequired(PermissionError):
    """
    The reference voice has no consent on record, so it may not be cloned.

    A PermissionError because it is exactly that: the request is well formed,
    but this recording is not ours to use. The API turns it into a 403.
    """


def require_voice_consent(speaker_wav: str) -> None:
    """
    Refuse to clone a recording unless its provenance permits it.

    Same rule as faces (`provenance.usability`): synthetic speech depicts no
    real person and is always usable; a human recording needs a recorded
    consent basis; a recording with no sidecar is refused - silence is not
    consent. Before this check nothing on the cloning path asked at all.
    """
    path = Path(speaker_wav)
    if not path.is_file():
        # Checked first so a typo is reported as a missing file, not as a
        # missing consent record for a file that does not exist.
        raise FileNotFoundError(
            f"Reference audio not found: '{speaker_wav}'. "
            "Pick one from GET /api/v1/audio/samples."
        )
    record = provenance.load(path)
    if record is None:
        _refuse_voice(path, "no provenance record", VoiceConsentRequired(
            f"'{path.name}' has no provenance record, so it may not be cloned. "
            "Register it with scripts/make_reference.py --human FILE --speaker NAME "
            "--licence LICENCE --consent <basis>, or use --smoke for a synthetic test voice."
        ))
    usable, reason = record.usability()  # type: ignore[union-attr]
    if not usable:
        _refuse_voice(path, reason, VoiceConsentRequired(f"'{path.name}' may not be cloned: {reason}."))


def _refuse_voice(path: Path, reason: str, error: Exception) -> None:
    """Write the refusal to the audit trail (by file hash, never by name), then raise it."""
    audit_log.shared_audit().record("voice_refused", subject=manifest.sha256_file(path), basis=None, reason=reason, file=path.name)
    raise error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
INPUT_DIR = PROJECT_ROOT / "inputs"
OUTPUT_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Supported English language codes for Kokoro routing
# ---------------------------------------------------------------------------
_KOKORO_ENGLISH_CODES = {"en", "en-us", "en-gb", "en-au", "en-ca"}

# ---------------------------------------------------------------------------
# Higgs TTS 2 default voice preset (no reference audio required)
# ---------------------------------------------------------------------------
_HIGGS_DEFAULT_VOICE = "default"

# ---------------------------------------------------------------------------
# Dia speaker tag pattern — if text contains [S1] or [S2] triggers Dia
# ---------------------------------------------------------------------------
_DIA_SPEAKER_TAGS = {"[S1]", "[S2]", "[s1]", "[s2]"}
_TAG_PATTERN = re.compile(r"\[(?:S|s)[12]\]")


@dataclass(frozen=True)
class SynthesisResult:
    output_path: str
    sample_rate: int
    duration_seconds: float
    latency_ms: float
    model: str
    mode: str
    phoneme_timestamps: Optional[list] = None
    # "mms_fa" when the timestamps were measured from the audio,
    # "acoustic-fallback" when they were only estimated from the text.
    alignment_method: Optional[str] = None
    emotion: Optional[dict] = None
    quality_report: Optional[dict] = None
    language: Optional[dict] = None
    # What the audio watermark step did: {"applied": bool, ...}. Always present, so an
    # unmarked clip can never be mistaken for a marked one (golden rule 1).
    watermark: Optional[dict] = None



class VoiceEngineRouter:
    """
    Intelligent Multi-Model TTS Orchestration Router.

    Lazily loads each model on first use to keep startup fast.
    All models are cached in instance attributes after first load.
    """

    def __init__(self, device: Optional[str] = None):
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Router Initialized] Processing Device: {self.device.upper()}")

        # Lazy-loaded model handles
        self.kokoro_pipeline = None   # Kokoro-82M
        self.xtts_model = None        # XTTS-v2
        self._higgs_pipe = None       # Higgs TTS 2 (3B)

        # Phase 3 engines. All three are cheap to construct and load lazily.
        self._mms = MMSTTSEngine(device=self.device)
        self._openvoice = OpenVoiceEngine(device=self.device)
        self._bark = BarkEngine(device=self.device)
        self._emotion = EmotionProsodyEngine()
        self._auditor: Optional[SpeechQualityAuditor] = None

        # Re-entrant: synthesize() can call itself through a fallback.
        self._synthesis_lock = threading.RLock()

        # Track load failures to avoid retrying broken models
        self._higgs_failed = False
        # Set when Bark's weights are present but it fails to load (e.g. out
        # of memory): dialogue then degrades visibly to Kokoro (D-26).
        self._bark_failed = False
        self._mms_failed = False

        # Lets the vision side make room on the 6 GB card before it loads a
        # model (golden rule 5). Everything reloads lazily on the next call.
        gpu_utils.register_releaser("tts-router", self.release)

    def release(self) -> None:
        """Drop the heavy TTS models from memory; they reload on next use."""
        self.kokoro_pipeline = None
        self.xtts_model = None
        self._higgs_pipe = None
        self._openvoice.release()
        self._bark.release()
        alignment_engine.release_shared_models()
        watermark_engine.shared_watermarker().release()

    @property
    def auditor(self) -> SpeechQualityAuditor:
        """Quality auditor, constructed on first use (SQUIM loads lazily inside)."""
        if self._auditor is None:
            self._auditor = SpeechQualityAuditor(device=self.device)
        return self._auditor

    # ------------------------------------------------------------------
    # Model Selection — Routing Decision Matrix
    # ------------------------------------------------------------------

    def select_model(
        self,
        mode: str = "fast",
        language: str = "en",
        quality: str = "balanced",
        style: Optional[str] = None,
        text: str = "",
        clone_engine: Optional[str] = None,
    ) -> str:
        """
        Returns the canonical model key for the given request parameters.

        Priority order:
          1. Explicit dialogue mode / style or [S1]/[S2] tags → dia-1.6b
          2. Explicit clone mode → xtts-v2
          3. Explicit high_quality mode OR quality=high → higgs-tts-2
          4. mode='multilingual' → mms-tts (errors if the language has no checkpoint)
          5. Non-English with an MMS-TTS checkpoint → mms-tts
          6. Non-English without one → higgs-tts-2
          7. fast mode English → kokoro
          8. Unknown mode → raise ValueError
        """
        normalized_lang = language.lower().replace("_", "-")
        info = language_registry.resolve(language)

        # 1. Dialogue detection
        is_dialogue_mode = mode == "dialogue"
        is_dialogue_style = (style or "").lower() in {"dialogue", "multi-speaker"}
        has_speaker_tags = any(tag in text for tag in _DIA_SPEAKER_TAGS)

        if is_dialogue_mode or is_dialogue_style or has_speaker_tags:
            # Bark, not Dia: Dia cannot run on this stack (model_registry.UNRUNNABLE).
            if self._bark_failed:
                logger.warning("Bark failed to load — falling back to Kokoro for dialogue request")
                return "kokoro"
            return "bark"

        # 2. Voice cloning, with the engine the caller chose (XTTS-v2 by default)
        if mode == "clone":
            engine = clone_engine or CLONE_ENGINES[0]
            if engine not in CLONE_ENGINES:
                raise ValueError(f"cloneEngine must be one of {list(CLONE_ENGINES)}, got {engine!r}")
            return engine

        # 3. Explicit high quality mode
        if mode == "high_quality" or quality == "high":
            if self._higgs_failed:
                logger.warning("Higgs unavailable — falling back to XTTS-v2 for high_quality request")
                return "xtts-v2"
            return "higgs-tts-2"

        # 4. Explicit multilingual mode — the caller wants MMS-TTS specifically.
        if mode == "multilingual":
            if not info.mms_supported:
                raise ValueError(
                    f"mode='multilingual' requires an MMS-TTS checkpoint, but "
                    f"'{language}'"
                    + (f" (ISO-639-3 '{info.iso3}')" if info.iso3 else "")
                    + " is not among the "
                    f"{language_registry.supported_count()} supported languages. "
                    "Use mode='high_quality' (Higgs) or mode='clone' (XTTS-v2)."
                )
            if self._mms_failed:
                logger.warning("MMS-TTS unavailable — falling back to Higgs for %s", language)
                return "xtts-v2" if self._higgs_failed else "higgs-tts-2"
            return "mms-tts"

        # 5/6. Non-English: prefer MMS-TTS coverage, else Higgs.
        if normalized_lang not in _KOKORO_ENGLISH_CODES:
            if info.mms_supported and not self._mms_failed:
                return "mms-tts"
            if self._higgs_failed:
                raise ValueError(
                    f"Language '{language}' has no available backend: MMS-TTS "
                    f"{'failed to load' if info.mms_supported else 'has no checkpoint'} "
                    "and Higgs TTS 2 failed to load. "
                    "Use mode='clone' with XTTS-v2 for multilingual synthesis."
                )
            return "higgs-tts-2"

        # 7. Fast English
        if mode == "fast":
            return "kokoro"

        # 8. Unknown
        raise ValueError(
            f"Unsupported mode='{mode}'. Valid modes: 'fast', 'clone', "
            "'high_quality', 'dialogue', 'multilingual'."
        )

    @staticmethod
    def xtts_language(language: str) -> str:
        """XTTS-v2's code for ``language`` (``spa`` -> ``es``), or a refusal naming the alternative."""
        code = language_registry.xtts_code(language)
        if code is None:
            raise ValueError(
                f"XTTS-v2 cannot speak '{language}'. It covers "
                f"{', '.join(sorted(language_registry.XTTS_LANGUAGES))}; use "
                "cloneEngine='openvoice-v2' to clone into any of the 1077 MMS-TTS languages."
            )
        return code

    def openvoice_base(self, language: str) -> str:
        """
        The engine that speaks the words OpenVoice then re-timbres.

        Kokoro for English, MMS-TTS for any language it has a checkpoint for -
        which is what makes OpenVoice a cross-lingual cloner here.
        """
        if language.lower().replace("_", "-") in _KOKORO_ENGLISH_CODES:
            return "kokoro"
        if language_registry.resolve(language).mms_supported:
            return "mms-tts"
        raise ValueError(
            f"OpenVoice clones over a base voice, and no base engine speaks '{language}': "
            "Kokoro covers English and MMS-TTS its 1077 languages. Use cloneEngine='xtts-v2'."
        )

    def preflight(
        self, model_key: str, speaker_wav: Optional[str] = None, language: str = "en"
    ) -> None:
        """
        Everything that must hold before a loader runs, cheapest first.

        Consent before weights: a recording we may not use is refused whatever
        is installed. Shared by the API route (so the refusal happens before
        anything is queued) and synthesize() (so no other caller skips it).
        """
        if speaker_wav and model_key in REFERENCE_ENGINES:
            require_voice_consent(speaker_wav)
        self.require_weights(model_key)
        if self.watermark_enabled():
            self.require_weights("audioseal")  # refused up front, not after minutes of synthesis
        if model_key == "xtts-v2" and speaker_wav:
            # A language XTTS cannot speak is a 400 now, not a failure minutes into the job.
            self.xtts_language(language)
        if model_key == "openvoice-v2":
            # Its base voice must be installed too, or the job would fail late.
            self.require_weights(self.openvoice_base(language))

    def require_weights(self, model_key: str) -> None:
        """
        Raise ModelWeightsMissing unless ``model_key``'s weights are on disk.

        Audited fresh on every call (about 5 ms; filesystem checks only, no
        torch), so weights fetched while the server runs are picked up without
        a restart. A key the audit does not know is let through.
        """
        for status in model_registry.audit_model_weights():
            if status.key == model_key and not status.present:
                raise ModelWeightsMissing(model_key, status.detail, status.fix)

    # ------------------------------------------------------------------
    # Model Loaders (lazy, cached)
    # ------------------------------------------------------------------

    def load_kokoro_realtime(self) -> None:
        """Loads Kokoro-82M for real-time sub-100ms streaming generation."""
        if self.kokoro_pipeline is not None:
            return
        print(f"\n[Loading Model] Kokoro v1.0 (82M) on {self.device.upper()}...")
        from kokoro import KPipeline
        self.kokoro_pipeline = KPipeline(lang_code="a", device=self.device)
        print(" -> Kokoro v1.0 loaded.")

    def load_xtts_cloning(self) -> None:
        """Loads XTTS-v2 for zero-shot voice cloning."""
        if self.xtts_model is not None:
            return
        if self.device == "cpu":
            gpu_utils.ensure_host_memory(XTTS_RAM_MB, "XTTS-v2")
        print(f"\n[Loading Model] XTTS-v2 on {self.device.upper()}...")
        from TTS.api import TTS
        self.xtts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        if self.device == "cuda" and torch.cuda.is_available():
            self.xtts_model = self.xtts_model.to("cuda")
        else:
            self.xtts_model = self.xtts_model.to("cpu")
        print(" -> XTTS-v2 loaded.")

    def load_higgs(self) -> bool:
        """
        Loads Higgs TTS 2 (3B, bosonai/higgs-tts-2-3b-base) via transformers pipeline.
        Returns True on success, False on failure (OOM / missing).
        Caches failure in self._higgs_failed to skip future attempts.
        """
        if self._higgs_pipe is not None:
            return True
        if self._higgs_failed:
            return False

        print(f"\n[Loading Model] Higgs TTS 2 (3B) on {self.device.upper()}...")
        try:
            from transformers import pipeline as hf_pipeline

            device_arg = 0 if self.device == "cuda" else -1
            self._higgs_pipe = hf_pipeline(
                "text-to-speech",
                model="bosonai/higgs-tts-2-3b-base",
                device=device_arg,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            print(" -> Higgs TTS 2 (3B) loaded.")
            return True
        except Exception as exc:  # noqa: BLE001 - cached failure, logged and reported
            logger.error("Higgs TTS 2 failed to load: %s", exc)
            self._higgs_failed = True
            print(f" -> [WARNING] Higgs TTS 2 load failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Synthesis Backends
    # ------------------------------------------------------------------

    def _synthesize_kokoro(self, text: str, output_path: Path) -> tuple[int, float]:
        """Run Kokoro-82M synthesis. Returns (sample_rate, duration_seconds)."""
        self.load_kokoro_realtime()
        assert self.kokoro_pipeline is not None
        print(f"\n[Kokoro] Synthesizing: '{text[:80]}...'")
        generator = self.kokoro_pipeline(text, voice="af_heart", speed=1.0)
        chunks = [audio for _, _, audio in generator]
        if not chunks:
            raise RuntimeError("Kokoro returned no audio chunks")
        audio = np.concatenate(chunks)
        sample_rate = 24000
        sf.write(output_path, audio, sample_rate)
        return sample_rate, len(audio) / sample_rate

    def _synthesize_xtts(
        self,
        text: str,
        output_path: Path,
        speaker_wav: Optional[str],
        language: str,
    ) -> tuple[int, float]:
        """Run XTTS-v2 voice cloning. Returns (sample_rate, duration_seconds)."""
        if not speaker_wav:
            raise FileNotFoundError(
                "Voice cloning requires a reference audio file.\n"
                "Place a WAV/MP3/FLAC recording (30–60 s) into the inputs/ folder, "
                "then select it in the UI."
            )

        speaker_path = Path(speaker_wav)
        if not speaker_path.exists():
            raise FileNotFoundError(
                f"Reference audio not found: '{speaker_wav}'\n"
                "Ensure the file is in the inputs/ folder and select it again."
            )

        # Validate + auto-convert to WAV 24 kHz mono (raises AudioValidationError on bad files)
        converted_dir = speaker_path.parent / ".converted"
        ready_path = validate_and_convert_for_cloning(speaker_path, converted_dir=converted_dir)

        self.load_xtts_cloning()
        assert self.xtts_model is not None
        print(f"\n[XTTS-v2] Cloning from '{ready_path.name}', text: '{text[:80]}...'")
        lang_code = self.xtts_language(language)
        self.xtts_model.tts_to_file(
            text=text,
            speaker_wav=str(ready_path),
            language=lang_code,
            file_path=str(output_path),
        )
        sample_rate = 24000
        duration = float(sf.info(output_path).duration)
        return sample_rate, duration

    def _synthesize_higgs(
        self,
        text: str,
        output_path: Path,
        speaker_wav: Optional[str] = None,
    ) -> tuple[int, float]:
        """
        Run Higgs TTS 2 (3B) synthesis.
        Supports optional speaker_wav for voice cloning.
        Falls back to XTTS-v2 on load failure.
        Returns (sample_rate, duration_seconds).
        """
        if not self.load_higgs():
            print("[WARN] Higgs unavailable — falling back to XTTS-v2")
            # Higgs fallback: use XTTS-v2 if speaker_wav provided, else Kokoro
            if speaker_wav and os.path.exists(speaker_wav):
                return self._synthesize_xtts(text, output_path, speaker_wav, "en")
            return self._synthesize_kokoro(text, output_path)

        print(f"\n[Higgs TTS 2] Synthesizing: '{text[:80]}...'")
        result = self._higgs_pipe(text)  # type: ignore[call-arg]
        audio_array = result["audio"]
        sample_rate = result.get("sampling_rate", 24000)

        # Ensure 1D float32 numpy array
        if isinstance(audio_array, (list, tuple)):
            audio_array = np.array(audio_array, dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze()

        sf.write(output_path, audio_array, sample_rate)
        duration = len(audio_array) / sample_rate
        return int(sample_rate), duration

    def _synthesize_mms(
        self,
        text: str,
        output_path: Path,
        language: str,
        speaker_wav: Optional[str] = None,
    ) -> tuple[int, float, dict]:
        """
        Run MMS-TTS for one of its 1077 languages.

        Returns (sample_rate, duration_seconds, language_info). Falls back to
        Higgs, then XTTS-v2, when the checkpoint cannot be fetched or loaded —
        the caller still gets audio, and ``language_info`` records what happened.
        """
        info = language_registry.resolve(language)
        try:
            result = self._mms.synthesize(
                text=text, language=language, output_path=output_path
            )
            payload = info.to_dict()
            payload.update({"backend": "mms-tts", "romanized": result.romanized})
            return result.sample_rate, result.duration_seconds, payload

        except MMSRomanizationRequired:
            # A missing romanizer is a configuration problem, not a model
            # failure; surface it instead of silently degrading quality.
            raise

        except Exception as exc:  # noqa: BLE001 - re-raised with the language named
            # This used to fall back to Higgs, then XTTS-v2. That path ran
            # *after* preflight, so it bypassed the weights check: Higgs's
            # loader would have started an 11.6 GB download of a model this
            # stack cannot run. One language failing (say, its checkpoint not
            # cached while offline) also set _mms_failed and disabled MMS for
            # every other language. Now the failure is this request's alone.
            logger.error("MMS-TTS failed for %s (%s)", language, exc)
            raise RuntimeError(
                f"MMS-TTS could not synthesise {info.name} ({info.iso3}): "
                f"{str(exc).splitlines()[0]}. A language it has not cached needs network "
                "on first use, or: python scripts/fetch_models.py --only mms-tts"
            ) from exc

    def _synthesize_bark(self, text: str, output_path: Path) -> tuple[int, float, str]:
        """
        Two-speaker dialogue with Bark; Kokoro if Bark fails to load.

        Returns (sample_rate, duration_seconds, engine_that_spoke).

        A load failure with weights present (missing weights never get here:
        preflight refuses them) degrades to Kokoro with the speaker tags
        removed, and is cached so later requests route straight to Kokoro.
        `model_used` then names Kokoro, so the fallback is never silent.
        """
        try:
            sample_rate, duration, turns = self._bark.synthesize_dialogue(text, output_path)
            print(f"[Bark] {turns} turn(s), {duration:.2f}s")
            return sample_rate, duration, "bark"
        except BarkUnavailable as exc:
            logger.error("Bark unavailable (%s); dialogue falls back to Kokoro", exc)
            self._bark_failed = True
            clean = " ".join(_TAG_PATTERN.sub(" ", text).split())
            sample_rate, duration = self._synthesize_kokoro(clean, output_path)
            return sample_rate, duration, "kokoro"

    def _synthesize_openvoice(
        self,
        text: str,
        output_path: Path,
        speaker_wav: Optional[str],
        language: str,
    ) -> tuple[int, float, str]:
        """
        Speak with the base engine, then re-timbre it as the reference speaker.

        Returns (sample_rate, duration_seconds, base_engine_key).
        """
        import tempfile

        if not speaker_wav:
            raise FileNotFoundError(
                "Voice cloning requires a reference recording (30-60 s). "
                "Pick one from GET /api/v1/audio/samples."
            )
        speaker_path = Path(speaker_wav)
        # Same reference preparation as XTTS-v2: validated, 24 kHz mono.
        ready = validate_and_convert_for_cloning(speaker_path, converted_dir=speaker_path.parent / ".converted")
        base_key = self.openvoice_base(language)
        # The base clip is an intermediate, not an output: a temporary file
        # that is removed whatever happens.
        with tempfile.TemporaryDirectory() as scratch:
            base_wav = Path(scratch) / "base.wav"
            if base_key == "kokoro":
                self._synthesize_kokoro(text, base_wav)
            else:
                self._mms.synthesize(text=text, language=language, output_path=base_wav)
            print(f"\n[OpenVoice V2] Re-timbring {base_key} speech as '{ready.name}'")
            sample_rate, duration = self._openvoice.convert(base_wav, ready, output_path)
        return sample_rate, duration, base_key

    # ------------------------------------------------------------------
    # Main Public Interface
    # ------------------------------------------------------------------

    def synthesize(self, *args, **kwargs) -> SynthesisResult:
        """
        Synthesize speech (see ``_synthesize`` for the parameters), one call at a time.

        The router holds lazily-loaded models and per-model state, and one heavy
        model should be resident at a time (golden rule 5), so two simultaneous
        calls (two REST requests, or a live session beside a REST request) must
        not load or run models at once. The lock makes them take turns.
        """
        with self._synthesis_lock:
            return self._synthesize(*args, **kwargs)

    def _synthesize(
        self,
        text: str,
        mode: str = "fast",
        speaker_wav: Optional[str] = None,
        output_filename: str = "speech.wav",
        language: str = "en",
        quality: str = "balanced",
        style: Optional[str] = None,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        return_alignment: bool = False,
        emotion: Optional[str] = None,
        emotion_intensity: float = 1.0,
        emotion_vector: Optional[dict] = None,
        audit_quality: bool = False,
        clone_engine: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Synthesize speech using the automatically selected model.

        Parameters
        ----------
        text              : Text to synthesize. Use [S1]/[S2] tags for Dia dialogue.
        mode              : 'fast' | 'clone' | 'high_quality' | 'dialogue' | 'multilingual'
        speaker_wav       : Path to reference WAV (required for 'clone', optional for 'high_quality')
        output_filename   : Output filename under outputs/
        language          : BCP-47 or ISO-639-3 code ('en', 'es', 'hin', 'swh', ...)
        quality           : 'fast' | 'balanced' | 'high'
        style             : Optional style hint ('dialogue', 'expressive', 'narration')
        speed             : Speed/rhythm multiplier (0.5x to 2.0x)
        pitch             : Pitch shift multiplier (0.5x to 2.0x)
        return_alignment  : Generate millisecond phoneme/viseme timestamps
        emotion           : Named emotion preset ('joy', 'anger', 'sorrow', 'authority', ...)
        emotion_intensity : Strength of that preset; the remainder stays neutral
        emotion_vector    : Blend of emotions, e.g. {'joy': 0.6, 'authority': 0.4}.
                            Takes precedence over ``emotion``.
        audit_quality     : Run the MOS/PESQ auditor on the result
        """
        if not text.strip():
            raise ValueError("text must not be empty")

        model_key = self.select_model(
            mode=mode,
            language=language,
            quality=quality,
            style=style,
            text=text,
            clone_engine=clone_engine,
        )
        # The one place every synthesis path passes through before a loader
        # runs: no request can trigger a download or clone an unconsented voice.
        self.preflight(model_key, speaker_wav, language)

        start_time = time.time()
        output_path = OUTPUT_DIR / output_filename
        language_info = language_registry.resolve(language).to_dict()
        language_info["backend"] = model_key

        print(
            f"\n{'='*60}\n"
            f"[VoiceEngine] mode={mode!r} lang={language!r} quality={quality!r} "
            f"style={style!r} speed={speed} pitch={pitch} align={return_alignment} → model={model_key!r}\n"
            f"{'='*60}"
        )

        if model_key == "kokoro":
            sample_rate, duration = self._synthesize_kokoro(text, output_path)

        elif model_key == "xtts-v2":
            sample_rate, duration = self._synthesize_xtts(
                text, output_path, speaker_wav, language
            )

        elif model_key == "higgs-tts-2":
            sample_rate, duration = self._synthesize_higgs(text, output_path, speaker_wav)

        elif model_key == "bark":
            sample_rate, duration, spoke = self._synthesize_bark(text, output_path)
            if spoke != "bark":
                # Bark failed to load and Kokoro spoke instead: say so, so the
                # response's model never names an engine that did not run.
                model_key = spoke
                language_info["backend"] = f"{spoke} (Bark unavailable)"

        elif model_key == "mms-tts":
            sample_rate, duration, language_info = self._synthesize_mms(
                text, output_path, language, speaker_wav
            )

        elif model_key == "openvoice-v2":
            sample_rate, duration, base_key = self._synthesize_openvoice(
                text, output_path, speaker_wav, language
            )
            # Two engines made this clip; both are named (golden rule 1).
            language_info["backend"] = f"openvoice-v2 (base: {base_key})"
            language_info["baseEngine"] = base_key

        else:
            raise ValueError(f"Internal error: unknown model key '{model_key}'")

        # ---- Phase 3: emotion prosody, blended with explicit speed/pitch ----
        # Both are time-stretch + pitch-shift operations, so they are combined
        # into one transform rather than applied in sequence.
        extra_rate = float(speed) if speed is not None else 1.0
        extra_semitones = 12.0 * math.log2(float(pitch)) if pitch is not None and pitch > 0 else 0.0

        emotion_report = None
        try:
            application = self._emotion.apply_to_file(
                output_path,
                vector=emotion_vector,
                emotion=emotion,
                intensity=emotion_intensity,
                extra_rate=extra_rate,
                extra_semitones=extra_semitones,
            )
            emotion_report = application.to_dict()
            if application.applied:
                duration = application.duration_seconds
                sample_rate = application.sample_rate
                print(
                    f"[Prosody] emotion={application.dominant} "
                    f"({application.intensity:.2f}) speed={extra_rate} "
                    f"pitch={extra_semitones:+.2f}st -> duration={duration:.2f}s"
                )
        except Exception as prosody_err:  # noqa: BLE001 - keep the raw audio
            logger.warning("Prosody / emotion post-processing failed: %s", prosody_err)

        # ---- Phase 5: inaudible watermark on every generated clip (R-32) -----
        # After prosody (the mark must survive nothing we do ourselves) and before alignment and
        # the quality audit, so both see the audio that is actually delivered.
        watermark_result = self._watermark_output(output_path)

        # Generate millisecond phoneme/viseme timestamps if requested
        phoneme_timestamps = None
        alignment_method = None
        if return_alignment:
            try:
                aligner = ForcedAligner(device=self.device)
                timestamps = aligner.align(
                    audio_path_or_tensor=str(output_path),
                    transcript=text,
                    sample_rate=sample_rate,
                    language=language,
                )
                phoneme_timestamps = [t.model_dump(by_alias=True) for t in timestamps]
                alignment_method = aligner.last_method
                print(
                    f"[Aligner] Extracted {len(phoneme_timestamps)} phoneme/viseme timestamps "
                    f"({alignment_method})."
                )
                if alignment_method == "acoustic-fallback":
                    print(f"[Aligner] WARNING: timing is estimated, not measured: {aligner.last_fallback_reason}")
            except Exception as align_err:  # noqa: BLE001 - speech is still returned
                # The audio is good without timestamps; the response simply
                # carries none, which callers can see.
                logger.warning("Forced alignment failed: %s", align_err)

        # ---- Phase 3: automated speech quality audit ------------------------
        quality_report = None
        if audit_quality:
            try:
                quality_report = self.auditor.audit(
                    output_path, reference_path=speaker_wav
                ).to_dict()
                print(
                    f"[Auditor] MOS={quality_report.get('mos')} "
                    f"PESQ={quality_report.get('pesq')} "
                    f"({quality_report.get('method')})"
                )
            except Exception as audit_err:  # noqa: BLE001 - auditing is advisory
                logger.warning("Quality audit failed: %s", audit_err)

        # A reference recording was used: that goes on the consent trail with the basis it was used under.
        if model_key in REFERENCE_ENGINES and speaker_wav:
            reference = manifest.reference_summary(speaker_wav)
            if reference:
                audit_log.shared_audit().record(
                    "voice_use", subject=reference["sha256"], basis=reference["consentBasis"] or reference["source"],
                    source=reference["source"], model=model_key, engine=clone_engine, language=language,
                    audio_sha256=manifest.sha256_file(output_path),
                )

        # How this clip was made, tied to its bytes: the render step copies it into the video's manifest.
        try:
            manifest.write_speech_record(
                output_path, model=model_key, mode=mode, language=language, speaker_wav=speaker_wav,
                clone_engine=clone_engine if model_key in REFERENCE_ENGINES else None,
                emotion=emotion_report, alignment_method=alignment_method, duration_seconds=duration,
                watermark=watermark_result,
            )
        except Exception as record_err:  # noqa: BLE001 - the audio is fine; the manifest will say "unrecorded"
            logger.warning("Could not write the speech record for %s: %s", output_path, record_err)

        latency = (time.time() - start_time) * 1000
        print(f"\n✅ Audio saved → {os.path.abspath(output_path)}")
        print(f"⏱  Latency: {latency:.0f} ms  |  Duration: {duration:.2f}s  |  Model: {model_key}")
        print("=" * 60)

        return SynthesisResult(
            output_path=str(output_path),
            sample_rate=sample_rate,
            duration_seconds=duration,
            latency_ms=latency,
            model=model_key,
            mode=mode,
            phoneme_timestamps=phoneme_timestamps,
            alignment_method=alignment_method,
            emotion=emotion_report,
            quality_report=quality_report,
            language=language_info,
            watermark=watermark_result,
        )

    @staticmethod
    def watermark_enabled() -> bool:
        return watermark_engine.enabled()

    def _watermark_output(self, output_path: Path) -> dict:
        """
        Hide the platform tag in the saved clip and check it can be read back.

        Marking is on by default and a failure to mark is a failure of the request, not a quiet
        fall-back to unmarked audio (golden rule 1). ``WATERMARK_ENABLED=false`` is the explicit
        opt-out, and the result then says ``applied: false`` and why.
        """
        if not self.watermark_enabled():
            return {"applied": False, "reason": "disabled by WATERMARK_ENABLED=false"}
        import soundfile as sf

        from watermark_engine import shared_watermarker

        marker = shared_watermarker()
        audio, rate = sf.read(str(output_path), dtype="float32", always_2d=False)
        sf.write(str(output_path), marker.embed(audio, rate), rate)
        # Read the file back and detect: proof the mark survived the write, with its own method string.
        check, _ = sf.read(str(output_path), dtype="float32", always_2d=False)
        report = marker.detect(check, rate)
        if not report.detected:
            raise RuntimeError(
                f"the watermark was embedded but could not be read back from {output_path.name} "
                f"({report.to_dict()}); refusing to deliver audio that claims a mark it does not carry"
            )
        return {"applied": True, "verified": True, **report.to_dict()}



# ------------------------------------------------------------------
# CLI Benchmark / Smoke Test
# ------------------------------------------------------------------
if __name__ == "__main__":
    router = VoiceEngineRouter()

    print("\n--- TEST 1: Kokoro fast (English) ---")
    router.synthesize(
        text="Hello! I am your AI avatar running real-time speech synthesis.",
        mode="fast",
        output_filename="test_kokoro.wav",
    )

    print("\n--- TEST 2: Higgs TTS 2 high_quality ---")
    router.synthesize(
        text="This is a high quality neural synthesis test using Higgs TTS 2.",
        mode="high_quality",
        output_filename="test_higgs.wav",
    )

    print("\n--- TEST 3: Dia-1.6B dialogue ---")
    router.synthesize(
        text="[S1] Good morning! How are you today? [S2] I am doing great, thank you!",
        mode="dialogue",
        output_filename="test_dia.wav",
    )

    print("\n--- TEST 4: Higgs multilingual (Spanish) ---")
    router.synthesize(
        text="Hola, soy tu avatar de inteligencia artificial.",
        mode="fast",
        language="es",
        output_filename="test_higgs_spanish.wav",
    )

    reference = str(INPUT_DIR / "voice_sample.wav")
    if os.path.exists(reference):
        print("\n--- TEST 5: XTTS-v2 voice clone ---")
        router.synthesize(
            text="This audio was generated by cloning the target voice using zero-shot deep learning.",
            mode="clone",
            speaker_wav=reference,
            output_filename="test_xtts_clone.wav",
        )
    else:
        print(f"\n--- TEST 5: SKIPPED (no reference audio at {reference}) ---")