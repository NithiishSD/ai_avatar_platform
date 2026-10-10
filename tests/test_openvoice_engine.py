"""
Tests for the OpenVoice V2 tone-colour converter wrapper.

The converter is replaced by a fake, so nothing is downloaded or loaded; what
is tested is our side: output format, the per-reference embedding cache, the
seed not leaking into the process, and a load failure that names its fix.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import soundfile as sf
import torch

from openvoice_engine import OUTPUT_SAMPLE_RATE, OpenVoiceEngine, OpenVoiceUnavailable


class FakeConverter:
    """Stands in for ToneColorConverter: 22.05 kHz output, counts calls."""

    def __init__(self):
        self.hps = SimpleNamespace(data=SimpleNamespace(sampling_rate=22050))
        self.extracted = []

    def extract_se(self, paths):
        self.extracted.append(paths[0])
        return torch.zeros(1, 256, 1)

    def convert(self, src, src_se, tgt_se, output_path=None, tau=0.3):
        # One second at the converter's native rate, and it draws randomness
        # the way the real flow does.
        torch.rand(1)
        return np.zeros(22050, dtype=np.float32)


def wav(path, seconds=1.0, sr=24000):
    sf.write(path, np.zeros(int(sr * seconds), dtype=np.float32), sr)
    return path


class ConvertTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.engine = OpenVoiceEngine(device="cpu")
        self.fake = FakeConverter()
        self.engine._converter = self.fake
        self.base = wav(self.dir / "base.wav")
        self.reference = wav(self.dir / "reference.wav")

    def test_output_is_written_at_the_pipeline_rate(self):
        out = self.dir / "out.wav"
        rate, duration = self.engine.convert(self.base, self.reference, out)
        self.assertEqual(rate, OUTPUT_SAMPLE_RATE)
        info = sf.info(str(out))
        self.assertEqual(info.samplerate, 24000)  # resampled from 22.05 kHz
        self.assertAlmostEqual(duration, 1.0, places=2)

    def test_the_reference_embedding_is_computed_once_per_recording(self):
        self.engine.convert(self.base, self.reference, self.dir / "a.wav")
        self.engine.convert(self.base, self.reference, self.dir / "b.wav")
        self.assertEqual(self.fake.extracted.count(str(self.reference)), 1)

    def test_a_changed_reference_is_not_matched_to_its_old_embedding(self):
        self.engine.convert(self.base, self.reference, self.dir / "a.wav")
        wav(self.reference, seconds=2.0)  # re-recorded: new size and mtime
        self.engine.convert(self.base, self.reference, self.dir / "b.wav")
        self.assertEqual(self.fake.extracted.count(str(self.reference)), 2)

    def test_seeding_does_not_change_anyone_elses_random_state(self):
        torch.manual_seed(1234)
        expected = torch.rand(3)
        torch.manual_seed(1234)
        self.engine.convert(self.base, self.reference, self.dir / "out.wav", seed=7)
        np.testing.assert_array_equal(torch.rand(3).numpy(), expected.numpy())


class LoadTests(unittest.TestCase):
    def test_a_load_failure_names_the_fix_and_is_not_retried(self):
        engine = OpenVoiceEngine(device="cpu")
        with mock.patch("huggingface_hub.snapshot_download", side_effect=FileNotFoundError("no snapshot")) as fetch:
            with self.assertRaises(OpenVoiceUnavailable) as first:
                engine._load()
            with self.assertRaises(OpenVoiceUnavailable):
                engine._load()
        self.assertIn("fetch_models.py --only openvoice-v2", str(first.exception))
        self.assertEqual(fetch.call_count, 1)

    def test_it_never_reaches_for_the_network(self):
        engine = OpenVoiceEngine(device="cpu")
        with mock.patch("huggingface_hub.snapshot_download", side_effect=FileNotFoundError("x")) as fetch:
            with self.assertRaises(OpenVoiceUnavailable):
                engine._load()
        self.assertTrue(fetch.call_args.kwargs["local_files_only"])


if __name__ == "__main__":
    unittest.main()
