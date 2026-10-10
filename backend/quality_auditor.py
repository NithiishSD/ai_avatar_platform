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

Where it sits in the pipeline: after speech synthesis and before any number is
reported. ``app.py`` exposes ``audit`` and ``speaker_similarity`` as the
``/api/v1/audio/quality-audit`` and ``/api/v1/audio/voice-similarity``
endpoints, ``voice_engine`` audits clips it synthesises, ``protected_voices``
uses ``embed_speaker`` to block cloning of registered voices, and the
``scripts/measure_*.py`` benchmarks use it to produce recorded evidence.

Concepts used throughout, explained once here:

**MOS** (Mean Opinion Score) is the average rating, 1 (bad) to 5 (excellent),
that human listeners give a clip. Collecting real ratings is slow, so SQUIM
*predicts* it with a neural network trained on rated speech.

**PESQ** (Perceptual Evaluation of Speech Quality, ITU-T P.862) scores how
degraded speech sounds; higher is better, up to about 4.5. **STOI** (Short-Time Objective
Intelligibility) is 0 to 1 and estimates how much of the speech a listener
would understand. **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio) is
in dB and measures how much of the signal is speech rather than noise or
artefacts. Classically all three need the clean original to compare against;
SQUIM's trick is to predict them from the degraded clip alone.

**Speaker embedding**: a speaker encoder such as ECAPA-TDNN maps a recording
of any length to a fixed-size vector (192 numbers here) that describes *who*
is speaking, not *what* is said. Two clips of the same voice give vectors
pointing in nearly the same direction.

**Cosine similarity** compares two vectors by the angle between them:
``dot(a, b) / (|a| * |b|)``. 1.0 means same direction (same voice), 0 means
unrelated. Loudness changes a vector's length but not its direction, which is
why cosine rather than plain distance is used for embeddings.

**How to say this in an interview:** "Every quality number we report is
produced by a named model and carries that name, so a heuristic fallback can
never be mistaken for real evidence."
"""

# ``from __future__ import annotations`` stores type hints as strings, so the
# ``Path | str`` union syntax works on Python 3.10 in every annotation position.
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
# Its models were trained at that rate, so every clip is resampled to it before
# inference. The speaker encoder used below (spkrec-ecapa-voxceleb) is also a
# 16 kHz model, which is why the similarity code reuses this constant.
SQUIM_SAMPLE_RATE = 16000

# Acceptance thresholds from the assignment.
# Both are strict "greater than" checks below: exactly 3.5 or 0.85 does not pass.
MOS_TARGET = 3.5
SIMILARITY_TARGET = 0.85


# A *dataclass* generates ``__init__``, ``__repr__`` and ``__eq__`` from the
# annotated fields, so a plain result container needs no boilerplate. Every
# field has a default, so a report can be built empty and filled in step by step.
@dataclass
class QualityReport:
    """
    Result of auditing one synthesized clip.

    ``mos``, ``pesq``, ``stoi`` and ``si_sdr`` stay ``None`` when no model
    produced them; ``method`` names what did ("torchaudio-squim",
    "dsp-estimate" or "empty"). ``warnings`` collects plain-language problems
    the caller should see next to the numbers.
    """

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
    # ``field(default_factory=list)`` gives each report its own new list. A bare
    # ``= []`` default would be one list shared by every instance, so a warning
    # added to one report would appear in all of them; dataclasses refuse it.
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        The report as a JSON-ready dict with camelCase keys for the API.

        Floats are rounded for display; ``None`` stays ``None`` so the client
        can tell "not measured" apart from a real 0.
        """
        # Small local helper: round a float but pass None through untouched.
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
            # Five digits for the ratios: a 0.01% clipping rate would round to 0.0 at three.
            "clippingRatio": r(self.clipping_ratio, 5),
            "silenceRatio": r(self.silence_ratio, 5),
            # The target travels with the result so the UI does not hard-code it.
            "mosTarget": MOS_TARGET,
            "passesMosTarget": self.passes_mos_target,
            # A copy, so a caller mutating the dict cannot change the report.
            "warnings": list(self.warnings),
        }


@dataclass
class SimilarityReport:
    """
    Result of comparing a cloned voice against its reference.

    ``is_ecapa`` is the flag that matters for evidence: only a True value means
    ``similarity`` came from the ECAPA-TDNN encoder and may be counted against
    the 0.85 target (golden rule 2).
    """

    similarity: float = 0.0
    method: str = "unavailable"
    reference_seconds: float = 0.0
    generated_seconds: float = 0.0
    passes_target: bool = False
    is_ecapa: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """The report as a JSON-ready dict with camelCase keys for the API."""
        return {
            "similarity": round(float(self.similarity), 4),
            # The same number as a percentage, because the assignment states the target as ">85%".
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
    """
    Read any audio file as a mono float32 array, optionally resampled.

    Returns ``(samples, sample_rate)``. Samples are floats in roughly [-1, 1].
    """
    # Imported inside the function so importing this module stays cheap; the
    # audio libraries load only when a clip is actually read.
    import soundfile as sf

    # always_2d=False returns a 1-D array for mono files and (frames, channels)
    # for multi-channel ones.
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        # Down-mix stereo (or more) to mono by averaging the channels; the
        # models below expect a single channel.
        audio = audio.mean(axis=1)
    if target_sr and sample_rate != target_sr:
        import librosa

        # Resampling changes the number of samples per second without changing
        # pitch or duration. Feeding 24 kHz audio to a 16 kHz model unconverted
        # would make the model hear it as slowed down and lower.
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr
    # ascontiguousarray: torch.from_numpy needs one unbroken block of memory,
    # and averaging or resampling can return a strided view.
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


class SpeechQualityAuditor:
    """
    Lazily-loaded MOS/PESQ predictor and speaker-similarity scorer.

    *Lazy loading* means the models are not loaded in ``__init__`` but on first
    use, so constructing an auditor at app start-up costs nothing and a GPU
    with 6 GB is not filled by models nobody has asked for yet (golden rule 5).
    One instance is shared by the app; it caches whatever it has loaded.
    """

    def __init__(self, device: Optional[str] = None):
        # "cuda" when a GPU is visible, else "cpu"; a caller may force either.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._objective = None
        self._subjective = None
        # The *_failed flags remember a failed load so it is not retried (and
        # re-logged) on every request; the fallback path is taken straight away.
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
            # torchaudio.pipelines bundles a model definition with the URL of its
            # pretrained weights; get_model() downloads them on first use.
            import torchaudio.pipelines as pipelines

            # .eval() switches layers such as dropout to inference behaviour, so
            # the same clip always gets the same score.
            self._objective = pipelines.SQUIM_OBJECTIVE.get_model().to(self.device).eval()
            self._subjective = pipelines.SQUIM_SUBJECTIVE.get_model().to(self.device).eval()
            logger.info("Loaded torchaudio SQUIM objective + subjective models on %s", self.device)
        except Exception as exc:  # noqa: BLE001
            # Broad catch on purpose: a missing package, a failed download and a
            # CUDA error all mean the same thing here, "use the labelled fallback".
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

        Returns a ``QualityReport`` whose ``method`` says which scorer ran.
        Raises ``FileNotFoundError`` when ``audio_path`` does not exist.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # First read at the file's own rate: duration and the clipping/silence
        # checks describe the file as delivered, before any resampling.
        audio, sample_rate = _load_mono(path)
        report = QualityReport(
            duration_seconds=len(audio) / sample_rate if sample_rate else 0.0,
            sample_rate=sample_rate,
        )
        if audio.size == 0:
            report.warnings.append("audio file is empty")
            report.method = "empty"
            return report

        # Clipping: samples at full scale (|x| >= 0.999) mean the waveform hit
        # the ceiling and was flattened, which sounds like crackle. Silence:
        # samples below 1e-4 (about -80 dBFS) are effectively no sound. np.mean
        # of a boolean array is the fraction of True values.
        report.clipping_ratio = float(np.mean(np.abs(audio) >= 0.999))
        report.silence_ratio = float(np.mean(np.abs(audio) < 1e-4))
        # Warn above 1% clipped samples or 60% silence; these limits are not
        # explained elsewhere in the code, they flag clips worth listening to.
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

        # Models take a batch: unsqueeze(0) turns shape (samples,) into
        # (1, samples), a batch of one clip.
        waveform = torch.from_numpy(resampled).unsqueeze(0).to(self.device)
        try:
            # inference_mode turns off gradient tracking: less memory, faster,
            # and nothing here trains.
            with torch.inference_mode():
                # The objective model returns three batched tensors; [0] picks
                # the score for our single clip.
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
                # The subjective model is given both clips at the same length.
                ref_tensor = self._match_length(ref_tensor, waveform)
                mos = subjective(waveform, ref_tensor)
                report.mos = float(mos[0])
            # Set only after every score succeeded, so a mid-way failure is not
            # labelled as a SQUIM result.
            report.method = "torchaudio-squim"
        except Exception as exc:  # noqa: BLE001
            logger.warning("SQUIM inference failed (%s); using DSP estimator", exc)
            report.method = "dsp-estimate"
            report.mos = self._dsp_mos_estimate(resampled)
            report.warnings.append(f"SQUIM inference failed: {exc}")

        # ``report.mos and ...`` guards against None (and treats 0.0 as a fail).
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
            # ``...`` (Ellipsis) means "all leading dimensions"; slice only time.
            return reference[..., :need]
        # Too short: repeat it end to end enough times, then cut to size.
        # max(have, 1) avoids dividing by zero on an empty reference.
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

        Returns a score clamped to the MOS range 1.0 to 5.0.
        """
        try:
            import librosa

            # Under 512 samples (32 ms at 16 kHz) there is too little audio to
            # analyse; score it as the worst possible.
            if audio.size < 512:
                return 1.0
            # HPSS (harmonic-percussive source separation) splits the signal into
            # a tonal part (voiced speech) and a noisy, transient part. Their
            # energy ratio in dB is a rough harmonic-to-noise ratio. The 1e-9
            # keeps log10 and the division defined for silent input.
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_energy = float(np.sum(harmonic**2)) + 1e-9
            noise_energy = float(np.sum(percussive**2)) + 1e-9
            hnr_db = 10.0 * math.log10(harmonic_energy / noise_energy)

            # Spectral flatness: near 1 for white noise, near 0 for a clean tone.
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
            # RMS is the loudness of each short frame; a frame counts as voiced
            # when it is louder than 10% of the loudest frame.
            rms = librosa.feature.rms(y=audio)[0]
            voiced_ratio = float(np.mean(rms > (0.1 * (np.max(rms) + 1e-9))))

            # Hand-picked weights around a mid-scale base of 2.4; the code does
            # not record how they were chosen, which is why this is labelled a
            # crude proxy and never counts as evidence.
            score = 2.4
            score += min(1.2, max(-1.2, hnr_db / 12.0))
            score += 0.7 * (1.0 - min(1.0, flatness * 12.0))
            score += 0.6 * voiced_ratio
            return float(max(1.0, min(5.0, score)))
        except Exception:  # noqa: BLE001
            # Mid-scale neutral value if even the DSP path fails (e.g. librosa missing).
            return 2.5

    # ------------------------------------------------------------------
    # Speaker similarity
    # ------------------------------------------------------------------

    def _load_ecapa(self):
        """
        Return the SpeechBrain ECAPA-TDNN speaker encoder, or None if it cannot load.

        ECAPA-TDNN is a neural network trained on VoxCeleb (thousands of
        speakers) to output speaker embeddings. The result is cached, and a
        failure is remembered so it is not retried on every call.
        """
        if self._ecapa is not None or self._ecapa_failed:
            return self._ecapa
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            # from_hparams downloads the model from the Hugging Face hub id in
            # ``source`` on first use and caches it in ``savedir``: the repo's
            # own .models/ecapa folder (parents[1] is the repo root, one level
            # above backend/).
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

    def embed_speaker(self, audio_path: Path | str) -> np.ndarray:
        """
        The ECAPA-TDNN speaker embedding of a recording (192 numbers), for comparing voices.

        There is no fallback: an abuse check built on a spectral proxy would look like it protected
        someone and not do it, so a missing encoder is an error that says how to fix it.

        Raises ``FileNotFoundError`` for a missing file, ``RuntimeError`` when the encoder
        cannot load, and ``ValueError`` for an empty recording. Used by ``protected_voices``.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        encoder = self._load_ecapa()
        if encoder is None:
            raise RuntimeError("ECAPA-TDNN is not available; install speechbrain (pip install speechbrain) to compare voices")
        wave, _ = _load_mono(path, target_sr=SQUIM_SAMPLE_RATE)
        if wave.size == 0:
            raise ValueError("the recording is empty")
        with torch.inference_mode():
            # encode_batch returns shape (batch, 1, 192); squeeze drops the size-1 axes.
            embedding = encoder.encode_batch(torch.from_numpy(wave).unsqueeze(0).to(self.device)).squeeze()
        # Back to a flat CPU numpy array so the caller can store it as JSON.
        return embedding.flatten().cpu().numpy().astype(np.float32)

    def speaker_similarity(
        self,
        reference_path: Path | str,
        generated_path: Path | str,
    ) -> SimilarityReport:
        """
        Cosine similarity between the speaker embeddings of two clips.

        Uses ECAPA-TDNN when it loads and runs; otherwise the MFCC fallback,
        labelled as such in ``method`` and ``warnings``. Raises
        ``FileNotFoundError`` when either file is missing.
        """
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
        # A short reference gives the encoder little voice to describe, so the
        # score is less trustworthy; warn below 3 s rather than refuse.
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
                # cosine_similarity compares row by row, so each embedding is
                # flattened and given a batch axis of 1; .item() unwraps the
                # single result to a Python float.
                similarity = torch.nn.functional.cosine_similarity(
                    ref_emb.flatten().unsqueeze(0), gen_emb.flatten().unsqueeze(0)
                ).item()
                report.similarity = float(similarity)
                report.method = "ecapa-tdnn (speechbrain/spkrec-ecapa-voxceleb)"
                report.is_ecapa = True
                report.passes_target = report.similarity > SIMILARITY_TARGET
                return report
            except Exception as exc:  # noqa: BLE001
                # Fall through to the spectral path below, with the failure on record.
                logger.warning("ECAPA inference failed (%s); using spectral fallback", exc)
                report.warnings.append(f"ECAPA inference failed: {exc}")

        report.similarity = self._spectral_similarity(ref, gen)
        # The method string itself says "NOT ECAPA-TDNN", so the label survives
        # even if a reader ignores ``isEcapa`` and the warnings.
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

        **MFCCs** (mel-frequency cepstral coefficients) describe the shape of
        the spectrum in each short frame on a pitch scale close to human
        hearing; that shape follows the speaker's vocal tract. Averaging them
        over the whole clip gives a crude voice fingerprint.
        """
        try:
            import librosa

            def embed(signal: np.ndarray) -> np.ndarray:
                # 24 coefficients per frame: shape (24, frames).
                mfcc = librosa.feature.mfcc(
                    y=signal, sr=SQUIM_SAMPLE_RATE, n_mfcc=24
                )
                # Deltas are the frame-to-frame change of each coefficient, a
                # little information about how the voice moves, not just its shape.
                delta = librosa.feature.delta(mfcc)
                # Mean and spread over time make a fixed-length vector (4 x 24 = 96)
                # whatever the clip length, so two clips can be compared.
                stats = np.concatenate(
                    [
                        mfcc.mean(axis=1), mfcc.std(axis=1),
                        delta.mean(axis=1), delta.std(axis=1),
                    ]
                )
                # Normalise to unit length; the dot product of two unit vectors
                # is their cosine similarity.
                norm = np.linalg.norm(stats) + 1e-9
                return stats / norm

            a, b = embed(reference), embed(generated)
            # Clip guards against tiny float error pushing the value past +/-1.
            return float(np.clip(np.dot(a, b), -1.0, 1.0))
        except Exception:  # noqa: BLE001
            # 0.0 means "no evidence of similarity", the safe answer if this fails.
            return 0.0
