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
        torch = mock.MagicMock()
        torch.cuda.memory_allocated.return_value = 0
        with mock.patch.dict("sys.modules", {"torch": torch}):
            with mock.patch("gpu_utils.free_vram_mb", side_effect=[500, 600]):
                with self.assertRaises(gpu_utils.InsufficientVRAM) as ctx:
                    gpu_utils.ensure_vram(1200, "Wav2Lip")
        message = str(ctx.exception)
        for fragment in ("1200", "600", "nvidia-smi"):
            self.assertIn(fragment, message)

    def test_blames_this_process_when_it_is_the_one_holding_memory(self):
        torch = mock.MagicMock()
        torch.cuda.memory_allocated.return_value = 3000 * 1024 * 1024
        with mock.patch.dict("sys.modules", {"torch": torch}):
            with mock.patch("gpu_utils.free_vram_mb", side_effect=[500, 600]):
                with self.assertRaises(gpu_utils.InsufficientVRAM) as ctx:
                    gpu_utils.ensure_vram(1200, "Wav2Lip")
        self.assertIn("This process still holds 3000 MiB", str(ctx.exception))
        self.assertNotIn("Another process", str(ctx.exception))

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


class HostMemoryTests(unittest.TestCase):
    """The RAM guard for CPU hosts: unload others when short, refuse rather than be OOM-killed."""

    def setUp(self):
        saved = dict(gpu_utils._releasers)
        self.addCleanup(lambda: (gpu_utils._releasers.clear(), gpu_utils._releasers.update(saved)))
        gpu_utils._releasers.clear()
        self.released = []
        gpu_utils.register_releaser("other", lambda: self.released.append("other"))
        gpu_utils.register_releaser("self", lambda: self.released.append("self"))
        patch = mock.patch("gpu_utils.cuda_available", return_value=False)
        patch.start()
        self.addCleanup(patch.stop)

    def free(self, *values):
        patch = mock.patch("gpu_utils.free_ram_mb", side_effect=list(values))
        patch.start()
        self.addCleanup(patch.stop)

    def test_plenty_of_ram_unloads_nothing(self):
        self.free(20000)
        gpu_utils.ensure_host_memory(4200, "XTTS-v2")
        self.assertEqual(self.released, [])

    def test_short_of_ram_unloads_the_others_and_then_proceeds(self):
        self.free(3000, 9000)  # before, after unloading
        gpu_utils.ensure_host_memory(4200, "XTTS-v2", keep="self")
        self.assertEqual(self.released, ["other"])

    def test_still_short_after_unloading_refuses_with_the_numbers_and_the_fix(self):
        self.free(1000, 2000)
        with self.assertRaises(gpu_utils.InsufficientRAM) as caught:
            gpu_utils.ensure_host_memory(4200, "XTTS-v2")
        message = str(caught.exception)
        self.assertIn("XTTS-v2", message)
        self.assertIn("4200", message)
        self.assertIn("2000", message)
        self.assertIn("other", message)  # names what it already unloaded
        self.assertIn("Close other programs", message)

    def test_headroom_is_required_on_top_of_the_model_size(self):
        # Exactly the model size free is not enough: the request still needs room.
        self.free(4200, 9000)
        gpu_utils.ensure_host_memory(4200, "XTTS-v2")
        self.assertEqual(sorted(self.released), ["other", "self"])

    def test_unknown_free_memory_does_not_interfere(self):
        self.free(None)
        gpu_utils.ensure_host_memory(4200, "XTTS-v2")
        self.assertEqual(self.released, [])

    def test_a_gpu_run_skips_the_ram_check(self):
        with mock.patch("gpu_utils.cuda_available", return_value=True), mock.patch("gpu_utils.free_ram_mb") as free:
            gpu_utils.ensure_host_memory(4200, "XTTS-v2")
        free.assert_not_called()

    def test_free_ram_reads_the_kernels_figure(self):
        value = gpu_utils.free_ram_mb()
        self.assertTrue(value is None or value > 0)  # None off Linux


class HeapTrimTests(unittest.TestCase):
    def test_empty_cache_asks_glibc_to_return_freed_pages(self):
        with mock.patch("ctypes.CDLL") as cdll:
            gpu_utils.empty_cache()
        cdll.assert_called_with("libc.so.6")
        cdll.return_value.malloc_trim.assert_called_once_with(0)

    def test_a_system_without_glibc_is_not_an_error(self):
        with mock.patch("ctypes.CDLL", side_effect=OSError("no libc.so.6")):
            gpu_utils.empty_cache()  # must not raise
