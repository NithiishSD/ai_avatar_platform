"""
The aligner must say how it produced its timestamps.

On 30 Sep 2026 another process held the GPU, MMS_FA failed to load, and every
render silently used the acoustic guess: SyncNet LSE-C fell from 2.8 to 1.6
and nothing in any result said why. These tests pin the fix.
"""

import unittest
from unittest import mock

import torch

from alignment_engine import ForcedAligner

WAVE = torch.zeros(1, 24000)
SPANS = ["measured"]


class AlignmentMethodTests(unittest.TestCase):
    def test_measured_alignment_is_labelled_mms_fa(self):
        aligner = ForcedAligner(device="cpu")
        aligner._get_mms_pipeline = mock.Mock(return_value=(mock.Mock(), mock.Mock()))
        aligner._align_mms = mock.Mock(return_value=SPANS)
        self.assertIs(aligner.align(WAVE, "hello there"), SPANS)
        self.assertEqual(aligner.last_method, "mms_fa")
        self.assertIsNone(aligner.last_fallback_reason)

    def test_fallback_is_labelled_and_loud(self):
        aligner = ForcedAligner(device="cpu")
        aligner._mms_failed = True
        aligner.last_fallback_reason = "MMS_FA could not be loaded: no weights"
        with self.assertLogs("alignment_engine", level="WARNING") as logs:
            timestamps = aligner.align(WAVE, "hello there")
        self.assertTrue(timestamps)
        self.assertEqual(aligner.last_method, "acoustic-fallback")
        self.assertTrue(any("FELL BACK" in line and "no weights" in line for line in logs.output))

    def test_inference_failure_is_a_warning_with_the_reason(self):
        aligner = ForcedAligner(device="cpu")
        aligner._get_mms_pipeline = mock.Mock(return_value=(mock.Mock(), mock.Mock()))
        aligner._align_mms = mock.Mock(side_effect=ValueError("token not in dictionary"))
        with self.assertLogs("alignment_engine", level="WARNING") as logs:
            aligner.align(WAVE, "hello there")
        self.assertEqual(aligner.last_method, "acoustic-fallback")
        self.assertIn("token not in dictionary", aligner.last_fallback_reason)
        self.assertTrue(any("token not in dictionary" in line for line in logs.output))

    def test_gpu_out_of_memory_retries_on_cpu_instead_of_guessing(self):
        aligner = ForcedAligner(device="cuda")
        model = mock.Mock()
        model.to.return_value = model
        aligner._mms_aligner = model
        aligner._get_mms_pipeline = mock.Mock(return_value=(model, mock.Mock()))
        aligner._align_mms = mock.Mock(side_effect=[RuntimeError("CUDA out of memory. Tried ..."), SPANS])
        with mock.patch("alignment_engine.torch.cuda.empty_cache"):
            with self.assertLogs("alignment_engine", level="WARNING"):
                result = aligner.align(WAVE, "hello there")
        self.assertIs(result, SPANS)
        self.assertEqual(aligner.last_method, "mms_fa")
        self.assertEqual(aligner.device, "cpu")
        model.to.assert_called_with("cpu")

    def test_model_that_will_not_fit_on_the_gpu_is_loaded_on_cpu(self):
        aligner = ForcedAligner(device="cuda")
        model = mock.Mock()
        model.to.side_effect = [RuntimeError("CUDA out of memory"), model]
        bundle = mock.Mock()
        bundle.get_model.return_value = model
        with mock.patch("torchaudio.pipelines.MMS_FA", bundle), \
             mock.patch("alignment_engine.torch.cuda.empty_cache"):
            with self.assertLogs("alignment_engine", level="WARNING"):
                loaded, _ = aligner._get_mms_pipeline()
        self.assertIs(loaded, model)
        self.assertEqual(aligner.device, "cpu")
        self.assertFalse(aligner._mms_failed)

    def test_empty_transcript_is_silence_not_a_fallback(self):
        aligner = ForcedAligner(device="cpu")
        aligner.align(WAVE, "   ")
        self.assertEqual(aligner.last_method, "silence")


if __name__ == "__main__":
    unittest.main()


class SharedModelTests(unittest.TestCase):
    """The wav2vec2 aligner loads once per process, not once per synthesis."""

    def setUp(self):
        import alignment_engine

        self.module = alignment_engine
        alignment_engine.release_shared_models()
        self.addCleanup(alignment_engine.release_shared_models)

    def fake_bundle(self):
        from unittest import mock

        model = mock.MagicMock()
        model.to.return_value = model
        bundle = mock.MagicMock()
        bundle.get_model.return_value = model
        bundle.get_tokenizer.return_value = "tokenizer"
        return bundle

    def test_a_second_aligner_reuses_the_loaded_model(self):
        from unittest import mock

        bundle = self.fake_bundle()
        with mock.patch("torchaudio.pipelines.MMS_FA", bundle):
            first = ForcedAligner(device="cpu")._get_mms_pipeline()
            second = ForcedAligner(device="cpu")._get_mms_pipeline()
        self.assertIs(first[0], second[0])
        bundle.get_model.assert_called_once()

    def test_each_aligner_keeps_its_own_per_call_state(self):
        a, b = ForcedAligner(device="cpu"), ForcedAligner(device="cpu")
        a.last_method = "mms_fa"
        self.assertIsNone(b.last_method)  # sharing the model must not share this

    def test_release_makes_the_next_aligner_load_again(self):
        from unittest import mock

        bundle = self.fake_bundle()
        with mock.patch("torchaudio.pipelines.MMS_FA", bundle):
            ForcedAligner(device="cpu")._get_mms_pipeline()
            self.module.release_shared_models()
            ForcedAligner(device="cpu")._get_mms_pipeline()
        self.assertEqual(bundle.get_model.call_count, 2)

    def test_the_routers_release_drops_the_shared_model(self):
        from unittest import mock

        from voice_engine import VoiceEngineRouter

        with mock.patch("torchaudio.pipelines.MMS_FA", self.fake_bundle()):
            ForcedAligner(device="cpu")._get_mms_pipeline()
        self.assertTrue(self.module._SHARED_MMS)
        VoiceEngineRouter(device="cpu").release()
        self.assertFalse(self.module._SHARED_MMS)
