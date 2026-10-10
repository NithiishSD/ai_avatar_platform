"""
Every heavy CPU loader asks for RAM first, and running short is not a broken install.

The failure being prevented: engines loaded one after another into the same
process until the operating system killed the server (exit 137, no message).
What is checked per engine: the guard is asked for that engine's measured
size, a refusal propagates, and it is NOT cached as a permanent load failure
(which would make the engine unusable until a restart even after memory freed).
"""

import unittest
from unittest import mock

import avatar_generator
import bark_engine
import gpu_utils
import openvoice_engine
import voice_engine
from avatar_generator import AvatarGenerator
from bark_engine import BarkEngine
from openvoice_engine import OpenVoiceEngine
from voice_engine import VoiceEngineRouter

SHORT = gpu_utils.InsufficientRAM("not enough RAM")


class LoaderGuardTests(unittest.TestCase):
    def guard(self, **kwargs):
        patcher = mock.patch("gpu_utils.ensure_host_memory", **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_openvoice_asks_for_its_size_and_a_refusal_is_not_cached(self):
        guard = self.guard(side_effect=SHORT)
        engine = OpenVoiceEngine(device="cpu")
        with self.assertRaises(gpu_utils.InsufficientRAM):
            engine._load()
        guard.assert_called_once_with(openvoice_engine.RAM_MB, "OpenVoice V2")
        self.assertIsNone(engine._failure)

    def test_bark_asks_for_its_size_and_a_refusal_is_not_cached(self):
        guard = self.guard(side_effect=SHORT)
        engine = BarkEngine(device="cpu")
        with self.assertRaises(gpu_utils.InsufficientRAM):
            engine._load()
        guard.assert_called_once_with(bark_engine.RAM_MB, "Bark")
        self.assertIsNone(engine._failure)

    def test_xtts_asks_for_its_size_and_stays_unloaded_on_refusal(self):
        guard = self.guard(side_effect=SHORT)
        router = VoiceEngineRouter(device="cpu")
        with self.assertRaises(gpu_utils.InsufficientRAM):
            router.load_xtts_cloning()
        guard.assert_called_once_with(voice_engine.XTTS_RAM_MB, "XTTS-v2")
        self.assertIsNone(router.xtts_model)

    def test_stable_diffusion_asks_for_its_size_and_keeps_itself_loaded(self):
        guard = self.guard(side_effect=SHORT)
        generator = AvatarGenerator(device="cpu")
        with self.assertRaises(gpu_utils.InsufficientRAM):
            generator._load()
        guard.assert_called_once_with(avatar_generator.RAM_MB, "Stable Diffusion 1.5", keep="avatar-diffusion")
        self.assertIsNone(generator._load_error)

    def test_a_gpu_engine_does_not_ask_for_system_ram(self):
        guard = self.guard()
        engine = OpenVoiceEngine(device="cuda")
        with mock.patch("huggingface_hub.snapshot_download", side_effect=OSError("no snapshot")):
            with self.assertRaises(openvoice_engine.OpenVoiceUnavailable):
                engine._load()
        guard.assert_not_called()

    def test_the_sizes_are_the_measured_ones(self):
        # 8 Oct 2026, loaded alone on the CPU: XTTS 4157, Bark 1840, OpenVoice 1637, SD 6610 MiB.
        self.assertGreaterEqual(voice_engine.XTTS_RAM_MB, 4157)
        self.assertGreaterEqual(bark_engine.RAM_MB, 1840)
        self.assertGreaterEqual(openvoice_engine.RAM_MB, 1637)
        self.assertGreaterEqual(avatar_generator.RAM_MB, 6610)


if __name__ == "__main__":
    unittest.main()
