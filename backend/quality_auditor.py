"""
Speech quality auditor (Phase 3).

Two measurements the acceptance matrix asks for, both automated:

1. **Perceptual quality** - torchaudio's SQUIM models predict quality metrics
   from a single degraded waveform, with no clean reference needed:
     * ``SQUIM_OBJECTIVE``  -> STOI, wideband PESQ, SI-SDR
     * ``SQUIM_SUBJECTIVE`` -> MOS, against any non-matching reference clip
   Target: MOS > 3.5.

2. **Speaker similarity** - cosine similarity between speaker embeddings of the
   cloning reference and the cloned output. SpeechBrain's ECAPA-TDNN is used
   when installed (the method the roadmap names); otherwise the auditor falls
   back to a spectral-envelope embedding and *says so* in ``method``, because a
   fallback number must never be reported as an ECAPA score.
   Target: similarity > 0.85.

Every report carries the method that produced it and whether the model was
real or a fallback, so benchmark evidence is never silently overstated.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# SQUIM operates on 16 kHz audio.
SQUIM_SAMPLE_RATE = 16000

# Acceptance thresholds from the assignment.
MOS_TARGET = 3.5
SIMILARITY_TARGET = 0.85


@dataclass
class QualityReport:
    """Result of auditing one synthesized clip."""

    mos: Optional[float] = None
    pesq: Optional[float] = None
    stoi: Optional[float] = None
    si_sdr: Optional[float] = None
    method: str = "unavailable"
    duration_seconds: float = 0.0
    sample_rate: int = 0
    clipping_ratio: float = 0.0
    silence_ratio: float = 0.0
    passes_mos_target: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        def r(value: Optional[float], digits: int = 3) -> Optional[float]:
            return None if value is None else round(float(value), digits)

        return {
            "mos": r(self.mos),
            "pesq": r(self.pesq),
            "stoi": r(self.stoi),
            "siSdr": r(self.si_sdr),
            "method": self.method,
            "durationSeconds": r(self.duration_seconds),
            "sampleRate": self.sample_rate,
            "clippingRatio": r(self.clipping_ratio, 5),
            "silenceRatio": r(self.silence_ratio, 5),
            "mosTarget": MOS_TARGET,
            "passesMosTarget": self.passes_mos_target,
            "warnings": list(self.warnings),
        }


@dataclass
class SimilarityReport:
    """Result of comparing a cloned voice against its reference."""

    similarity: float = 0.0
    method: str = "unavailable"
    reference_seconds: float = 0.0
    generated_seconds: float = 0.0
    passes_target: bool = False
    is_ecapa: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "similarity": round(float(self.similarity), 4),
            "similarityPercent": round(float(self.similarity) * 100.0, 2),
            "method": self.method,
            "isEcapa": self.is_ecapa,
            "referenceSeconds": round(self.reference_seconds, 3),
            "generatedSeconds": round(self.generated_seconds, 3),
            "similarityTarget": SIMILARITY_TARGET,
            "passesTarget": self.passes_target,
            "warnings": list(self.warnings),
        }


def _load_mono(path: Path | str, target_sr: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Read any audio file as a mono float32 array, optionally resampled."""
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if target_sr and sample_rate != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


class SpeechQualityAuditor:
    """Lazily-loaded MOS/PESQ predictor and speaker-similarity scorer."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._objective = None
        self._subjective = None
        self._squim_failed = False
        self._ecapa = None
        self._ecapa_failed = False

    # ------------------------------------------------------------------
    # SQUIM loading
    # ------------------------------------------------------------------

    def _load_squim(self):
        """Return (objective_model, subjective_model); either may be None."""
        if self._squim_failed:
            return None, None
        if self._objective is not None or self._subjective is not None:
            return self._objective, self._subjective

        try:
            import torchaudio.pipelines as pipelines

            self._objective = pipelines.SQUIM_OBJECTIVE.get_model().to(self.device).eval()
            self._subjective = pipelines.SQUIM_SUBJECTIVE.get_model().to(self.device).eval()
            logger.info("Loaded torchaudio SQUIM objective + subjective models on %s", self.device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SQUIM unavailable (%s); using DSP quality estimator", exc)
            self._squim_failed = True
            self._objective = None
            self._subjective = None
        return self._objective, self._subjective

    # ------------------------------------------------------------------
    # Quality audit
    # ------------------------------------------------------------------

    def audit(
        self,
        audio_path: Path | str,
        reference_path: Optional[Path | str] = None,
    ) -> QualityReport:
        """
        Predict perceptual quality for one synthesized clip.

        ``reference_path`` is the non-matching reference SQUIM_SUBJECTIVE needs:
        any clean speech clip, *not* required to be the same utterance. When it
        is omitted the clip is scored against itself, which SQUIM accepts but
        which biases MOS upward, so that case is flagged in ``warnings``.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        audio, sample_rate = _load_mono(path)
        report = QualityReport(
            duration_seconds=len(audio) / sample_rate if sample_rate else 0.0,
            sample_rate=sample_rate,
        )
        if audio.size == 0:
            report.warnings.append("audio file is empty")
            report.method = "empty"
            return report

        report.clipping_ratio = float(np.mean(np.abs(audio) >= 0.999))
        report.silence_ratio = float(np.mean(np.abs(audio) < 1e-4))
        if report.clipping_ratio > 0.01:
            report.warnings.append(
                f"{report.clipping_ratio * 100:.1f}% of samples are clipped"
            )
        if report.silence_ratio > 0.6:
            report.warnings.append(
                f"{report.silence_ratio * 100:.1f}% of the clip is silence"
            )

        resampled, _ = _load_mono(path, target_sr=SQUIM_SAMPLE_RATE)
        objective, subjective = self._load_squim()

        # Both, not just objective: _load_squim may return either as None, and
        # calling a missing subjective model used to raise inside the try below,
        # leaving a report labelled dsp-estimate that still held SQUIM's PESQ.
        if objective is None or subjective is None:
            report.method = "dsp-estimate"
            report.mos = self._dsp_mos_estimate(resampled)
            report.warnings.append(
                "SQUIM models unavailable; MOS is a DSP estimate, not a model prediction"
            )
            report.passes_mos_target = bool(report.mos and report.mos > MOS_TARGET)
            return report

        waveform = torch.from_numpy(resampled).unsqueeze(0).to(self.device)
        try:
            with torch.inference_mode():
                stoi, pesq, si_sdr = objective(waveform)
                report.stoi = float(stoi[0])
                report.pesq = float(pesq[0])
                report.si_sdr = float(si_sdr[0])

                if reference_path is not None and Path(reference_path).exists():
                    ref, _ = _load_mono(reference_path, target_sr=SQUIM_SAMPLE_RATE)
                    ref_tensor = torch.from_numpy(ref).unsqueeze(0).to(self.device)
                else:
                    ref_tensor = waveform
                    report.warnings.append(
                        "no non-matching reference supplied; MOS is self-referenced "
                        "and biased upward"
                    )
                ref_tensor = self._match_length(ref_tensor, waveform)
                mos = subjective(waveform, ref_tensor)
                report.mos = float(mos[0])
            report.method = "torchaudio-squim"
        except Exception as exc:  # noqa: BLE001
            logger.warning("SQUIM inference failed (%s); using DSP estimator", exc)
            report.method = "dsp-estimate"
            report.mos = self._dsp_mos_estimate(resampled)
            report.warnings.append(f"SQUIM inference failed: {exc}")

        report.passes_mos_target = bool(report.mos and report.mos > MOS_TARGET)
        return report

    @staticmethod
    def _match_length(reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Tile or trim the reference so both tensors have the same length."""
        need = target.shape[-1]
        have = reference.shape[-1]
        if have == need:
            return reference
        if have > need:
            return reference[..., :need]
        repeats = int(math.ceil(need / max(have, 1)))
        return reference.repeat(1, repeats)[..., :need]

    @staticmethod
    def _dsp_mos_estimate(audio: np.ndarray) -> float:
        """
        Crude MOS proxy for when SQUIM cannot load.

        Combines three cheap proxies for naturalness: harmonic-to-noise ratio,
        spectral flatness (buzzy synthetic speech is flatter than natural
        speech), and how much of the clip is actually voiced. Reported only as
        ``method="dsp-estimate"`` and never presented as a model prediction.
        """
        try:
            import librosa

            if audio.size < 512:
                return 1.0
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_energy = float(np.sum(harmonic**2)) + 1e-9
            noise_energy = float(np.sum(percussive**2)) + 1e-9
            hnr_db = 10.0 * math.log10(harmonic_energy / noise_energy)

            flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
            rms = librosa.feature.rms(y=audio)[0]
            voiced_ratio = float(np.mean(rms > (0.1 * (np.max(rms) + 1e-9))))

            score = 2.4
            score += min(1.2, max(-1.2, hnr_db / 12.0))
            score += 0.7 * (1.0 - min(1.0, flatness * 12.0))
            score += 0.6 * voiced_ratio
            return float(max(1.0, min(5.0, score)))
        except Exception:  # noqa: BLE001
            return 2.5

    # ------------------------------------------------------------------
    # Speaker similarity
    # ------------------------------------------------------------------

    def _load_ecapa(self):
        if self._ecapa is not None or self._ecapa_failed:
            return self._ecapa
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            self._ecapa = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(Path(__file__).resolve().parents[1] / ".models" / "ecapa"),
                run_opts={"device": self.device},
            )
            logger.info("Loaded SpeechBrain ECAPA-TDNN speaker encoder")
        except Exception as exc:  # noqa: BLE001
            logger.info("ECAPA-TDNN unavailable (%s); using spectral fallback", exc)
            self._ecapa_failed = True
            self._ecapa = None
        return self._ecapa

    def speaker_similarity(
        self,
        reference_path: Path | str,
        generated_path: Path | str,
    ) -> SimilarityReport:
        """Cosine similarity between the speaker embeddings of two clips."""
        ref_path, gen_path = Path(reference_path), Path(generated_path)
        for candidate in (ref_path, gen_path):
            if not candidate.exists():
                raise FileNotFoundError(f"Audio file not found: {candidate}")

        ref, _ = _load_mono(ref_path, target_sr=SQUIM_SAMPLE_RATE)
        gen, _ = _load_mono(gen_path, target_sr=SQUIM_SAMPLE_RATE)
        report = SimilarityReport(
            reference_seconds=len(ref) / SQUIM_SAMPLE_RATE,
            generated_seconds=len(gen) / SQUIM_SAMPLE_RATE,
        )
        if ref.size == 0 or gen.size == 0:
            report.warnings.append("one of the clips is empty")
            return report
        if report.reference_seconds < 3.0:
            report.warnings.append(
                f"reference is only {report.reference_seconds:.1f}s; "
                "the assignment asks for a 30-60s reference"
            )

        encoder = self._load_ecapa()
        if encoder is not None:
            try:
                with torch.inference_mode():
                    ref_emb = encoder.encode_batch(
                        torch.from_numpy(ref).unsqueeze(0).to(self.device)
                    ).squeeze()
                    gen_emb = encoder.encode_batch(
                        torch.from_numpy(gen).unsqueeze(0).to(self.device)
                    ).squeeze()
                similarity = torch.nn.functional.cosine_similarity(
                    ref_emb.flatten().unsqueeze(0), gen_emb.flatten().unsqueeze(0)
                ).item()
                report.similarity = float(similarity)
                report.method = "ecapa-tdnn (speechbrain/spkrec-ecapa-voxceleb)"
                report.is_ecapa = True
                report.passes_target = report.similarity > SIMILARITY_TARGET
                return report
            except Exception as exc:  # noqa: BLE001
                logger.warning("ECAPA inference failed (%s); using spectral fallback", exc)
                report.warnings.append(f"ECAPA inference failed: {exc}")

        report.similarity = self._spectral_similarity(ref, gen)
        report.method = "spectral-envelope-fallback (NOT ECAPA-TDNN)"
        report.is_ecapa = False
        report.passes_target = report.similarity > SIMILARITY_TARGET
        report.warnings.append(
            "ECAPA-TDNN not installed; this score is a spectral-envelope proxy and "
            "is not valid evidence for the >85% cloning-similarity threshold. "
            "Install with `pip install speechbrain`."
        )
        return report

    @staticmethod
    def _spectral_similarity(reference: np.ndarray, generated: np.ndarray) -> float:
        """
        Fallback similarity: cosine distance between long-term MFCC statistics.

        Captures vocal-tract shape well enough to separate different speakers,
        but it is not a verification-grade embedding.
        """
        try:
            import librosa

            def embed(signal: np.ndarray) -> np.ndarray:
                mfcc = librosa.feature.mfcc(
                    y=signal, sr=SQUIM_SAMPLE_RATE, n_mfcc=24
                )
                delta = librosa.feature.delta(mfcc)
                stats = np.concatenate(
                    [
                        mfcc.mean(axis=1), mfcc.std(axis=1),
                        delta.mean(axis=1), delta.std(axis=1),
                    ]
                )
                norm = np.linalg.norm(stats) + 1e-9
                return stats / norm

            a, b = embed(reference), embed(generated)
            return float(np.clip(np.dot(a, b), -1.0, 1.0))
        except Exception:  # noqa: BLE001
            return 0.0
