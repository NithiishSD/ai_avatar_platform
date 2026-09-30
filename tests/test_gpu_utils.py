"""VRAM guard for the 6 GB card."""

import unittest
from unittest import mock

import gpu_utils


class GpuUtilsTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(gpu_utils._releasers, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch("gpu_utils.empty_cache")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_no_cuda_passes(self):
        with mock.patch("gpu_utils.free_vram_mb", return_value=None):
            gpu_utils.ensure_vram(99999, "anything")

    def test_enough_memory_releases_nothing(self):
        release = mock.Mock()
        gpu_utils.register_releaser("tts", release)
        with mock.patch("gpu_utils.free_vram_mb", return_value=4000):
            gpu_utils.ensure_vram(1200, "Wav2Lip")
        release.assert_not_called()

    def test_frees_other_models_but_keeps_the_caller(self):
        other, own = mock.Mock(), mock.Mock()
        gpu_utils.register_releaser("tts", other)
        gpu_utils.register_releaser("wav2lip", own)
        with mock.patch("gpu_utils.free_vram_mb", side_effect=[500, 3000]):
            gpu_utils.ensure_vram(1200, "Wav2Lip", keep="wav2lip")
        other.assert_called_once()
        own.assert_not_called()

    def test_still_short_raises_with_the_numbers(self):
        with mock.patch("gpu_utils.free_vram_mb", side_effect=[500, 600]):
            with self.assertRaises(gpu_utils.InsufficientVRAM) as ctx:
                gpu_utils.ensure_vram(1200, "Wav2Lip")
        message = str(ctx.exception)
        for fragment in ("1200", "600", "nvidia-smi"):
            self.assertIn(fragment, message)

    def test_one_failing_releaser_does_not_stop_the_others(self):
        good = mock.Mock()
        gpu_utils.register_releaser("broken", mock.Mock(side_effect=RuntimeError("x")))
        gpu_utils.register_releaser("good", good)
        with self.assertLogs("gpu_utils", level="WARNING"):
            released = gpu_utils.release_others()
        self.assertEqual(released, ["good"])
        good.assert_called_once()


if __name__ == "__main__":
    unittest.main()
