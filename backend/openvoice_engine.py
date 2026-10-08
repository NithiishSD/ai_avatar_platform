"""
OpenVoice V2 tone-colour cloning: any base voice, re-timbred as a reference.

OpenVoice does not speak text itself. It is a *voice converter*: given speech
in one voice (the "base") and a short recording of another (the reference),
it keeps the words, rhythm and intonation of the base and swaps the timbre -
the quality that makes a voice recognisably someone's - for the reference's.
The router produces the base with an engine that already works here (Kokoro
for English, MMS-TTS for 1000+ other languages), so one converter clones in
every language either of those speaks.

The model compresses timbre into a *speaker embedding* ("SE"): a short vector
the reference encoder computes from a spectrogram. Conversion runs the base
through a flow with the base's embedding swapped for the target's.

Measured (T2.6, 8 Oct 2026, CPU): converting 3 s of Kokoro speech toward the
LJSpeech reference takes ~2 s, and ECAPA similarity to held-out LJSpeech
speech is 41.2% (5 seeds, sd 2.4) against 31% for the unconverted base and
92% for the speaker's own real speech. It moves the voice; it does not reach
the 85% target. XTTS-v2 is the engine expected to.

Licence: MIT (myshell-ai/OpenVoice, myshell-ai/OpenVoiceV2). The package is
installed without its pinned dependencies (numpy 1.22, librosa 0.9) - see
backend/requirements.txt.

**How to say this in an interview:** "Cloning is split into synthesis and
voice conversion: a TTS model we already trust produces the words, and a
small converter transfers the speaker embedding, so cloning works in every
language the base TTS covers."
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

import gpu_utils

logger = logging.getLogger(__name__)

OPENVOICE_REPO = "myshell-ai/OpenVoiceV2"
# Measured 8 Oct 2026: converter plus its base voice add 1637 MiB on a CPU host.
RAM_MB = 1700
# The package, pinned to the commit whose code was read before installing it,
# and installed without its own requirements (they would downgrade numpy).
OPENVOICE_PIP = (
    'pip install --no-deps "git+https://github.com/myshell-ai/OpenVoice.git'
    '@74a1d147b17a8c3092dd5430504bd83ef6c7eb23"'
)
# Only the converter is needed: the repo's base speakers belong to MeloTTS,
# which we do not use (its pinned transformers 4.27 would break Coqui).
CONVERTER_FILES = ("converter/config.json", "converter/checkpoint.pth")
OUTPUT_SAMPLE_RATE = 24000  # the pipeline's standard; the converter runs at 22.05 kHz
# Flow temperature. 0.3 is upstream's default and measured best of 0.3/0.5 over
# five seeds (41.2% vs 38.4%); single runs vary by ~5 points, so the sweep
# showed no reason to move it.
DEFAULT_TAU = 0.3


class OpenVoiceUnavailable(RuntimeError):
    """The converter or its weights cannot be loaded; the message names the fix."""


class OpenVoiceEngine:
    """
    Lazily loaded tone-colour converter with a per-reference embedding cache.

    Loading is deferred to first use and a failure is cached, the same pattern
    as the other engines: a missing model costs one clear error, not a retry
    (and a stack trace) on every request.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device
        self._converter: Any = None
        self._failure: Optional[str] = None
        # Reference embeddings, keyed by (path, mtime, size) so a re-recorded
        # file is never matched to its old embedding.
        self._target_se: Dict[Tuple[str, int, int], Any] = {}

    def _load(self) -> Any:
        if self._converter is not None:
            return self._converter
        if self._failure is not None:
            raise OpenVoiceUnavailable(self._failure)
        # Before the try: running short of RAM is not a broken install, so it
        # must not be cached as a permanent load failure.
        if self.device in (None, "cpu"):
            gpu_utils.ensure_host_memory(RAM_MB, "OpenVoice V2")
        try:
            import torch
            from huggingface_hub import snapshot_download
            from openvoice.api import OpenVoiceBaseClass, ToneColorConverter

            # local_files_only: never download at request time; a missing
            # file must fail here with the fetch command.
            snapshot = Path(
                snapshot_download(OPENVOICE_REPO, allow_patterns=["converter/*"], local_files_only=True)
            )
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

            class _Converter(ToneColorConverter):
                """
                Upstream ToneColorConverter without its watermark.

                Upstream forwards ``enable_watermark`` to a base class that
                does not accept it, so the documented way to turn the
                watermark off raises TypeError; the default (on) needs the
                `wavmark` package and stamps OpenVoice's own mark. This skips
                straight to the base initialiser instead.
                """

                def __init__(self, config_path: str, device: str) -> None:
                    OpenVoiceBaseClass.__init__(self, config_path, device=device)
                    self.watermark_model = None
                    self.version = getattr(self.hps, "_version_", "v1")

            converter = _Converter(str(snapshot / CONVERTER_FILES[0]), device=device)
            converter.load_ckpt(str(snapshot / CONVERTER_FILES[1]))
            self.device = device
            self._converter = converter
            logger.info("Loaded OpenVoice V2 converter on %s", device)
            return converter
        except Exception as exc:  # noqa: BLE001 - cached and raised with the fix
            self._failure = (
                f"OpenVoice V2 could not be loaded ({str(exc).splitlines()[0]}). "
                "Fetch it with: python scripts/fetch_models.py --only openvoice-v2"
            )
            logger.error(self._failure)
            raise OpenVoiceUnavailable(self._failure) from exc

    def _embedding_for(self, reference: Path) -> Any:
        stat = reference.stat()
        key = (str(reference.resolve()), stat.st_mtime_ns, stat.st_size)
        if key not in self._target_se:
            self._target_se[key] = self._load().extract_se([str(reference)])
        return self._target_se[key]

    def convert(
        self,
        base_wav: Path,
        reference_wav: Path,
        output_path: Path,
        seed: int = 0,
        tau: float = DEFAULT_TAU,
    ) -> Tuple[int, float]:
        """
        Re-timbre ``base_wav`` as the speaker of ``reference_wav``.

        Writes 24 kHz mono to ``output_path`` and returns
        ``(sample_rate, duration_seconds)``. Seeded: the flow samples noise,
        and an unseeded conversion of the same input varied by ~5 similarity
        points between runs.
        """
        import librosa
        import soundfile as sf
        import torch

        converter = self._load()
        source_se = converter.extract_se([str(base_wav)])
        target_se = self._embedding_for(reference_wav)
        # fork_rng scopes the seed to this block, so seeding here cannot change
        # the random state any other model in the process relies on.
        devices = [torch.cuda.current_device()] if str(self.device).startswith("cuda") else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            audio = converter.convert(str(base_wav), source_se, target_se, output_path=None, tau=tau)
        native = int(converter.hps.data.sampling_rate)
        audio = np.asarray(audio, dtype=np.float32)
        if native != OUTPUT_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=native, target_sr=OUTPUT_SAMPLE_RATE)
        sf.write(output_path, audio, OUTPUT_SAMPLE_RATE)
        return OUTPUT_SAMPLE_RATE, len(audio) / OUTPUT_SAMPLE_RATE

    def release(self) -> None:
        """Drop the converter and cached embeddings; both rebuild on next use."""
        self._converter = None
        self._target_se.clear()
