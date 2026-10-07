"""
Tests for the Phase 3 speech quality auditor.

SQUIM and ECAPA are stubbed so these run offline. The point of these tests is
the *reporting contract*: a fallback score must never be presented as a model
prediction, because the acceptance matrix depends on that distinction.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
import torch

from quality_auditor import (
    MOS_TARGET,
    SIMILARITY_TARGET,
    QualityReport,
    SimilarityReport,
    SpeechQualityAuditor,
)


def write_tone(path: Path, freq: float = 150.0, duration: float = 1.5, sr: int = 24000):
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = 0.3 * (np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(2 * np.pi * freq * 2 * t))
    sf.write(path, signal.astype(np.float32), sr)
    return path


class ReportSerializationTests(unittest.TestCase):
    def test_quality_report_exposes_the_target_and_verdict(self):
        payload = QualityReport(mos=4.1, method="torchaudio-squim").to_dict()
        self.assertEqual(payload["mosTarget"], MOS_TARGET)
        self.assertEqual(payload["method"], "torchaudio-squim")

    def test_similarity_report_reports_percent_and_method(self):
        payload = SimilarityReport(
            similarity=0.91, method="ecapa-tdnn", is_ecapa=True, passes_target=True
        ).to_dict()
        self.assertAlmostEqual(payload["similarityPercent"], 91.0)
        self.assertEqual(payload["similarityTarget"], SIMILARITY_TARGET)
        self.assertTrue(payload["isEcapa"])


class QualityAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio = write_tone(Path(self.tmp.name) / "clip.wav")
        self.auditor = SpeechQualityAuditor(device="cpu")

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.auditor.audit(Path(self.tmp.name) / "nope.wav")

    def test_squim_unavailable_falls_back_and_says_so(self):
        with mock.patch.object(self.auditor, "_load_squim", return_value=(None, None)):
            report = self.auditor.audit(self.audio)
        self.assertEqual(report.method, "dsp-estimate")
        self.assertTrue(any("SQUIM" in w for w in report.warnings))
        self.assertGreaterEqual(report.mos, 1.0)
        self.assertLessEqual(report.mos, 5.0)

    def test_missing_subjective_model_is_a_dsp_estimate_not_mixed_metrics(self):
        # _load_squim may return either model as None. With only the objective
        # model present, the old code computed SQUIM's PESQ/STOI and then
        # crashed calling None, leaving a report labelled dsp-estimate that
        # still held model numbers. Either missing must mean the honest fallback.
        objective = mock.Mock()
        with mock.patch.object(self.auditor, "_load_squim", return_value=(objective, None)):
            report = self.auditor.audit(self.audio)
        self.assertEqual(report.method, "dsp-estimate")
        objective.assert_not_called()
        self.assertIsNone(report.pesq)
        self.assertTrue(any("SQUIM models unavailable" in w for w in report.warnings))

    def test_squim_results_are_reported_with_all_four_metrics(self):
        objective = mock.Mock(
            return_value=(torch.tensor([0.93]), torch.tensor([3.8]), torch.tensor([17.2]))
        )
        subjective = mock.Mock(return_value=torch.tensor([4.21]))
        with mock.patch.object(
            self.auditor, "_load_squim", return_value=(objective, subjective)
        ):
            report = self.auditor.audit(self.audio)

        self.assertEqual(report.method, "torchaudio-squim")
        self.assertAlmostEqual(report.stoi, 0.93, places=3)
        self.assertAlmostEqual(report.pesq, 3.8, places=3)
        self.assertAlmostEqual(report.si_sdr, 17.2, places=3)
        self.assertAlmostEqual(report.mos, 4.21, places=3)
        self.assertTrue(report.passes_mos_target)

    def test_mos_below_target_does_not_pass(self):
        objective = mock.Mock(
            return_value=(torch.tensor([0.5]), torch.tensor([1.5]), torch.tensor([2.0]))
        )
        subjective = mock.Mock(return_value=torch.tensor([2.9]))
        with mock.patch.object(
            self.auditor, "_load_squim", return_value=(objective, subjective)
        ):
            report = self.auditor.audit(self.audio)
        self.assertFalse(report.passes_mos_target)

    def test_self_referenced_mos_is_flagged(self):
        objective = mock.Mock(
            return_value=(torch.tensor([0.9]), torch.tensor([3.0]), torch.tensor([10.0]))
        )
        subjective = mock.Mock(return_value=torch.tensor([4.0]))
        with mock.patch.object(
            self.auditor, "_load_squim", return_value=(objective, subjective)
        ):
            report = self.auditor.audit(self.audio)
        self.assertTrue(any("non-matching reference" in w for w in report.warnings))

    def test_supplying_a_reference_removes_the_bias_warning(self):
        reference = write_tone(Path(self.tmp.name) / "ref.wav", freq=200.0)
        objective = mock.Mock(
            return_value=(torch.tensor([0.9]), torch.tensor([3.0]), torch.tensor([10.0]))
        )
        subjective = mock.Mock(return_value=torch.tensor([4.0]))
        with mock.patch.object(
            self.auditor, "_load_squim", return_value=(objective, subjective)
        ):
            report = self.auditor.audit(self.audio, reference_path=reference)
        self.assertFalse(any("non-matching reference" in w for w in report.warnings))

    def test_squim_inference_failure_degrades_to_the_dsp_estimate(self):
        objective = mock.Mock(side_effect=RuntimeError("cuda oom"))
        subjective = mock.Mock()
        with mock.patch.object(
            self.auditor, "_load_squim", return_value=(objective, subjective)
        ):
            report = self.auditor.audit(self.audio)
        self.assertEqual(report.method, "dsp-estimate")
        self.assertTrue(any("cuda oom" in w for w in report.warnings))

    def test_clipping_is_detected(self):
        clipped = Path(self.tmp.name) / "clipped.wav"
        sf.write(clipped, np.ones(24000, dtype=np.float32), 24000)
        with mock.patch.object(self.auditor, "_load_squim", return_value=(None, None)):
            report = self.auditor.audit(clipped)
        self.assertGreater(report.clipping_ratio, 0.9)
        self.assertTrue(any("clipped" in w for w in report.warnings))

    def test_silence_is_detected(self):
        silent = Path(self.tmp.name) / "silent.wav"
        sf.write(silent, np.zeros(24000, dtype=np.float32), 24000)
        with mock.patch.object(self.auditor, "_load_squim", return_value=(None, None)):
            report = self.auditor.audit(silent)
        self.assertGreater(report.silence_ratio, 0.9)
        self.assertTrue(any("silence" in w for w in report.warnings))

    def test_reference_length_is_matched_before_scoring(self):
        short = torch.zeros(1, 100)
        target = torch.zeros(1, 350)
        matched = SpeechQualityAuditor._match_length(short, target)
        self.assertEqual(matched.shape[-1], 350)

        long = torch.zeros(1, 900)
        matched = SpeechQualityAuditor._match_length(long, target)
        self.assertEqual(matched.shape[-1], 350)


class SpeakerSimilarityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.reference = write_tone(root / "ref.wav", freq=150.0, duration=4.0)
        self.same = write_tone(root / "same.wav", freq=150.0, duration=4.0)
        self.different = write_tone(root / "diff.wav", freq=400.0, duration=4.0)
        self.auditor = SpeechQualityAuditor(device="cpu")

    def tearDown(self):
        self.tmp.cleanup()

    def test_ecapa_result_is_labelled_as_ecapa(self):
        encoder = mock.Mock()
        encoder.encode_batch.side_effect = [
            torch.ones(1, 1, 192),
            torch.ones(1, 1, 192),
        ]
        with mock.patch.object(self.auditor, "_load_ecapa", return_value=encoder):
            report = self.auditor.speaker_similarity(self.reference, self.same)
        self.assertTrue(report.is_ecapa)
        self.assertIn("ecapa", report.method)
        self.assertAlmostEqual(report.similarity, 1.0, places=4)
        self.assertTrue(report.passes_target)

    def test_fallback_is_clearly_marked_as_not_ecapa(self):
        """A fallback number must never be mistaken for verification evidence."""
        with mock.patch.object(self.auditor, "_load_ecapa", return_value=None):
            report = self.auditor.speaker_similarity(self.reference, self.same)
        self.assertFalse(report.is_ecapa)
        self.assertIn("NOT ECAPA", report.method)
        self.assertTrue(any("not valid evidence" in w for w in report.warnings))

    def test_fallback_separates_a_different_voice_from_the_same_one(self):
        with mock.patch.object(self.auditor, "_load_ecapa", return_value=None):
            same = self.auditor.speaker_similarity(self.reference, self.same)
            other = self.auditor.speaker_similarity(self.reference, self.different)
        self.assertGreater(same.similarity, other.similarity)

    def test_short_reference_is_flagged_against_the_assignment(self):
        short = write_tone(Path(self.tmp.name) / "short.wav", duration=1.0)
        with mock.patch.object(self.auditor, "_load_ecapa", return_value=None):
            report = self.auditor.speaker_similarity(short, self.same)
        self.assertTrue(any("30-60s reference" in w for w in report.warnings))

    def test_ecapa_inference_failure_falls_back(self):
        encoder = mock.Mock()
        encoder.encode_batch.side_effect = RuntimeError("bad shape")
        with mock.patch.object(self.auditor, "_load_ecapa", return_value=encoder):
            report = self.auditor.speaker_similarity(self.reference, self.same)
        self.assertFalse(report.is_ecapa)
        self.assertTrue(any("bad shape" in w for w in report.warnings))

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.auditor.speaker_similarity(self.reference, Path(self.tmp.name) / "no.wav")


if __name__ == "__main__":
    unittest.main()
