"""
Tests for the MMS-TTS engine. Checkpoints are stubbed, so nothing downloads.

These cover the parts that are easy to get wrong without a GPU in the loop:
LRU eviction (a 6 GB card cannot hold many VITS checkpoints), uroman handling,
and failure caching.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
import torch

from mms_engine import (
    MMSLanguageNotSupported,
    MMSRomanizationRequired,
    MMSTTSEngine,
)


class FakeConfig:
    sampling_rate = 16000


class FakeWaveform:
    def __init__(self, samples: int):
        self._tensor = torch.zeros(1, samples)

    def squeeze(self):
        return self._tensor.squeeze()


class FakeOutput:
    def __init__(self, samples: int):
        self.waveform = torch.rand(1, samples) * 0.5 - 0.25


class FakeModel:
    """Stands in for transformers.VitsModel."""

    def __init__(self, samples: int = 16000):
        self.config = FakeConfig()
        self.samples = samples
        self.speaking_rate = None
        self.noise_scale = None
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        return FakeOutput(self.samples)


class FakeTokenizer:
    def __init__(self, is_uroman: bool = False):
        self.is_uroman = is_uroman
        self.last_text = None

    def __call__(self, text, return_tensors=None):
        self.last_text = text
        return {"input_ids": torch.ones(1, max(len(text), 1), dtype=torch.long)}


def patched_engine(engine: MMSTTSEngine, model=None, tokenizer=None, side_effect=None):
    """Patch transformers loading inside ``mms_engine._load``."""
    model = model or FakeModel()
    tokenizer = tokenizer if tokenizer is not None else FakeTokenizer()
    vits = mock.Mock()
    auto = mock.Mock()
    if side_effect:
        vits.from_pretrained.side_effect = side_effect
    else:
        vits.from_pretrained.return_value = model
    auto.from_pretrained.return_value = tokenizer
    return mock.patch.dict(
        "sys.modules",
        {"transformers": mock.Mock(VitsModel=vits, AutoTokenizer=auto)},
    )


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.engine = MMSTTSEngine(device="cpu")

    def test_supported_languages_are_recognised(self):
        self.assertTrue(self.engine.supports("hi"))
        self.assertTrue(self.engine.supports("swh"))

    def test_unsupported_language_is_recognised(self):
        self.assertFalse(self.engine.supports("ja"))

    def test_unsupported_language_raises_with_an_actionable_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MMSLanguageNotSupported) as ctx:
                self.engine.synthesize("hello", "ja", Path(tmp) / "out.wav")
        message = str(ctx.exception)
        self.assertIn("jpn", message)
        self.assertIn("high_quality", message)

    def test_empty_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.engine.synthesize("   ", "hi", Path(tmp) / "out.wav")


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        self.engine = MMSTTSEngine(device="cpu")
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "out.wav"

    def tearDown(self):
        self.tmp.cleanup()

    def test_synthesis_writes_a_readable_wav(self):
        with patched_engine(self.engine):
            result = self.engine.synthesize("namaste", "hi", self.out)
        self.assertTrue(self.out.exists())
        audio, sample_rate = sf.read(self.out)
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(result.model_id, "facebook/mms-tts-hin")
        self.assertAlmostEqual(result.duration_seconds, len(audio) / sample_rate, places=5)
        self.assertFalse(result.romanized)

    def test_speaking_rate_is_passed_to_the_generator(self):
        model = FakeModel()
        with patched_engine(self.engine, model=model):
            self.engine.synthesize("namaste", "hi", self.out, speaking_rate=1.25)
        self.assertAlmostEqual(model.speaking_rate, 1.25)

    def test_output_is_peak_normalized(self):
        class LoudModel(FakeModel):
            def __call__(self, **kwargs):
                output = FakeOutput(1000)
                output.waveform = torch.full((1, 1000), 4.0)
                return output

        with patched_engine(self.engine, model=LoudModel()):
            self.engine.synthesize("loud", "hi", self.out)
        audio, _ = sf.read(self.out)
        self.assertLessEqual(float(np.max(np.abs(audio))), 1.0 + 1e-6)


class UromanTests(unittest.TestCase):
    def setUp(self):
        self.engine = MMSTTSEngine(device="cpu")
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "out.wav"

    def tearDown(self):
        self.tmp.cleanup()

    def test_uroman_checkpoint_with_non_ascii_text_needs_a_romanizer(self):
        tokenizer = FakeTokenizer(is_uroman=True)
        with patched_engine(self.engine, tokenizer=tokenizer):
            with mock.patch.object(self.engine, "_get_romanizer", return_value=None):
                with self.assertRaises(MMSRomanizationRequired) as ctx:
                    self.engine.synthesize("नमस्ते", "hi", self.out)
        self.assertIn("uroman", str(ctx.exception))

    def test_uroman_checkpoint_accepts_ascii_text_without_a_romanizer(self):
        """Already-romanized input needs no romanizer, so it must not error."""
        tokenizer = FakeTokenizer(is_uroman=True)
        with patched_engine(self.engine, tokenizer=tokenizer):
            with mock.patch.object(self.engine, "_get_romanizer", return_value=None):
                result = self.engine.synthesize("namaste", "hi", self.out)
        self.assertFalse(result.romanized)

    def test_romanizer_output_is_what_reaches_the_tokenizer(self):
        tokenizer = FakeTokenizer(is_uroman=True)
        romanizer = mock.Mock()
        romanizer.romanize_string.return_value = "namaste"
        with patched_engine(self.engine, tokenizer=tokenizer):
            with mock.patch.object(self.engine, "_get_romanizer", return_value=romanizer):
                result = self.engine.synthesize("नमस्ते", "hi", self.out)
        self.assertEqual(tokenizer.last_text, "namaste")
        self.assertTrue(result.romanized)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "out.wav"

    def tearDown(self):
        self.tmp.cleanup()

    def test_checkpoints_are_cached_and_reused(self):
        engine = MMSTTSEngine(device="cpu", cache_size=3)
        model = FakeModel()
        with patched_engine(engine, model=model) as patched:
            engine.synthesize("a", "hi", self.out)
            engine.synthesize("b", "hi", self.out)
            vits = __import__("sys").modules["transformers"].VitsModel
            self.assertEqual(vits.from_pretrained.call_count, 1)
        self.assertEqual(engine.loaded_languages(), ["hin"])

    def test_lru_eviction_keeps_the_cache_within_its_limit(self):
        """VRAM is finite; a 6 GB card cannot hold an unbounded checkpoint set."""
        engine = MMSTTSEngine(device="cpu", cache_size=2)
        with patched_engine(engine):
            for code in ("hi", "ta", "swh"):
                engine.synthesize("text", code, self.out)
        self.assertEqual(len(engine.loaded_languages()), 2)
        self.assertNotIn("hin", engine.loaded_languages())

    def test_reuse_refreshes_recency(self):
        engine = MMSTTSEngine(device="cpu", cache_size=2)
        with patched_engine(engine):
            engine.synthesize("text", "hi", self.out)
            engine.synthesize("text", "ta", self.out)
            engine.synthesize("text", "hi", self.out)   # hin becomes most recent
            engine.synthesize("text", "swh", self.out)  # evicts tam, not hin
        self.assertIn("hin", engine.loaded_languages())
        self.assertNotIn("tam", engine.loaded_languages())

    def test_load_failure_is_cached_and_not_retried(self):
        engine = MMSTTSEngine(device="cpu")
        with patched_engine(engine, side_effect=OSError("no network")):
            with self.assertRaises(MMSLanguageNotSupported):
                engine.synthesize("text", "hi", self.out)
            with self.assertRaises(MMSLanguageNotSupported) as ctx:
                engine.synthesize("text", "hi", self.out)
            vits = __import__("sys").modules["transformers"].VitsModel
            self.assertEqual(vits.from_pretrained.call_count, 1)
        self.assertIn("previously failed", str(ctx.exception))

    def test_unload_all_clears_the_cache(self):
        engine = MMSTTSEngine(device="cpu")
        with patched_engine(engine):
            engine.synthesize("text", "hi", self.out)
        engine.unload_all()
        self.assertEqual(engine.loaded_languages(), [])


if __name__ == "__main__":
    unittest.main()
