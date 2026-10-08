"""
Audio watermark (R-32): the decision rule, the key, the resampling around the model, the
router hook, and (when the weights are on disk) the real AudioSeal round trip.
"""

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
import torch

import watermark_engine as w
from watermark_engine import AudioWatermarker, WatermarkUnavailable

KEY = b"unit-test-key"


def speechlike(seconds=2.0, sr=24000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) + 0.02 * rng.standard_normal(t.size)).astype(np.float32)


class FakeGenerator:
    def __init__(self):
        self.seen = []

    def get_watermark(self, wave, sample_rate=None, message=None):
        self.seen.append((wave.shape[-1], sample_rate, message.tolist()[0]))
        return 0.01 * torch.ones_like(wave)


class FakeDetector:
    def __init__(self, probability, bits):
        self.probability, self.bits = probability, bits

    def detect_watermark(self, wave, sample_rate=None):
        return self.probability, torch.tensor([self.bits], dtype=torch.float32)


def marker_with(probability=1.0, bits=None):
    marker = AudioWatermarker()
    marker._generator = FakeGenerator()
    marker._detector = FakeDetector(probability, bits if bits is not None else w.platform_tag(KEY))
    return marker


class KeyAndTagTests(unittest.TestCase):
    def setUp(self):
        w.reset_key_cache()
        self.addCleanup(w.reset_key_cache)

    def test_the_tag_is_sixteen_bits_and_depends_on_the_key(self):
        a, b = w.platform_tag(b"key-a"), w.platform_tag(b"key-b")
        self.assertEqual(len(a), 16)
        self.assertTrue(set(a) <= {0, 1})
        self.assertEqual(a, w.platform_tag(b"key-a"))
        self.assertNotEqual(a, b)

    def test_an_environment_key_wins(self):
        with mock.patch.dict(os.environ, {"WATERMARK_KEY": "from-env"}):
            self.assertEqual(w.signing_key(), b"from-env")

    def test_without_an_env_key_one_is_created_once_private_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "key"
            with mock.patch.dict(os.environ, {"WATERMARK_KEY": "", "PROVENANCE_KEY_FILE": str(path)}):
                first = w.signing_key()
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)  # only the owner can read it
                w.reset_key_cache()
                self.assertEqual(w.signing_key(), first)  # read back, not regenerated
                self.assertGreaterEqual(len(first), 32)


class DecisionRuleTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("watermark_engine.signing_key", return_value=KEY)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tag = w.platform_tag(KEY)
        self.clip = speechlike()

    def detect(self, probability, bits):
        return marker_with(probability, bits).detect(self.clip, 24000)

    def test_our_tag_with_high_probability_is_detected(self):
        report = self.detect(0.99, self.tag)
        self.assertTrue(report.detected)
        self.assertEqual(report.bits_matching, 16)

    def test_a_couple_of_flipped_bits_from_re_encoding_still_count(self):
        flipped = list(self.tag)
        flipped[0], flipped[5] = 1 - flipped[0], 1 - flipped[5]
        self.assertTrue(self.detect(0.9, flipped).detected)  # 14 of 16

    def test_three_flipped_bits_do_not_count(self):
        flipped = list(self.tag)
        for i in (0, 5, 9):
            flipped[i] = 1 - flipped[i]
        report = self.detect(0.9, flipped)
        self.assertFalse(report.detected)
        self.assertEqual(report.bits_matching, 13)

    def test_someone_elses_mark_is_a_mark_but_not_ours(self):
        other = [1 - b for b in self.tag]
        report = self.detect(0.99, other)
        self.assertFalse(report.detected)
        self.assertGreater(report.probability, 0.9)  # present, but the bits are not ours

    def test_our_bits_with_low_confidence_do_not_count(self):
        self.assertFalse(self.detect(0.2, self.tag).detected)

    def test_silence_is_never_detected_and_says_why(self):
        report = marker_with().detect(np.zeros(24000, dtype=np.float32), 24000)
        self.assertFalse(report.detected)
        self.assertIn("silent", " ".join(report.warnings))

    def test_a_very_short_clip_is_flagged_as_unreliable(self):
        report = marker_with().detect(speechlike(0.2), 24000)
        self.assertIn("shorter than half a second", " ".join(report.warnings))

    def test_the_report_carries_its_method(self):
        text = self.detect(0.99, self.tag).to_dict()["method"]
        self.assertIn("audioseal", text)


class EmbedTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("watermark_engine.signing_key", return_value=KEY)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_embedding_keeps_length_and_leaves_the_input_untouched(self):
        clip = speechlike()
        before = clip.copy()
        out = marker_with().embed(clip, 24000)
        self.assertEqual(out.shape, clip.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.array_equal(clip, before))
        self.assertFalse(np.array_equal(out, clip))

    def test_the_model_is_fed_16_khz_and_the_tag_not_the_clips_own_rate(self):
        marker = marker_with()
        marker.embed(speechlike(2.0, 24000), 24000)
        samples, rate, message = marker._generator.seen[0]
        self.assertEqual(rate, 16000)
        self.assertEqual(samples, 32000)       # 2 s at 16 kHz, resampled from 48,000 samples
        self.assertEqual(message, w.platform_tag(KEY))

    def test_a_clip_at_the_model_rate_is_not_resampled(self):
        marker = marker_with()
        marker.embed(speechlike(1.0, 16000), 16000)
        self.assertEqual(marker._generator.seen[0][0], 16000)

    def test_a_mark_that_would_clip_scales_the_clip_down_instead(self):
        loud = np.full(24000, 0.999, dtype=np.float32)
        out = marker_with().embed(loud, 24000)
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)

    def test_stereo_is_mixed_to_mono_and_an_empty_clip_is_refused(self):
        stereo = np.stack([speechlike(), speechlike(seed=1)], axis=1)
        self.assertEqual(marker_with().embed(stereo, 24000).ndim, 1)
        with self.assertRaises(ValueError):
            marker_with().embed(np.zeros(0, dtype=np.float32), 24000)

    def test_resample_changes_length_by_the_rate_ratio_and_fit_pads_or_trims(self):
        self.assertEqual(w._resample(np.ones(48000, dtype=np.float32), 48000, 16000).shape[0], 16000)
        self.assertEqual(w._fit(np.ones(10, dtype=np.float32), 12).shape[0], 12)
        self.assertEqual(w._fit(np.ones(15, dtype=np.float32), 12).shape[0], 12)


class LoadFailureTests(unittest.TestCase):
    def test_missing_weights_raise_with_the_fix_and_the_failure_is_cached(self):
        marker = AudioWatermarker()
        with mock.patch.object(AudioWatermarker, "weights_dir", side_effect=OSError("not cached")):
            with self.assertRaises(WatermarkUnavailable) as caught:
                marker.embed(speechlike(), 24000)
        self.assertIn("fetch_models.py --only audioseal", str(caught.exception))
        with self.assertRaises(WatermarkUnavailable):
            marker.detect(speechlike(), 24000)  # cached: does not retry the load on every clip


class RouterHookTests(unittest.TestCase):
    """VoiceEngineRouter._watermark_output with the marker faked, and the preflight."""

    def setUp(self):
        from voice_engine import VoiceEngineRouter

        self.router = VoiceEngineRouter(device="cpu")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "clip.wav"
        sf.write(self.path, speechlike(1.0), 24000)
        patcher = mock.patch("watermark_engine.signing_key", return_value=KEY)
        patcher.start()
        self.addCleanup(patcher.stop)

    def with_env(self, value):
        return mock.patch.dict(os.environ, {"WATERMARK_ENABLED": value})

    def test_a_marked_clip_is_rewritten_verified_and_reported(self):
        marker = marker_with()
        with self.with_env("true"), mock.patch("watermark_engine.shared_watermarker", return_value=marker):
            result = self.router._watermark_output(self.path)
        self.assertEqual((result["applied"], result["verified"], result["detected"]), (True, True, True))
        self.assertEqual(result["bitsMatching"], 16)
        self.assertEqual(len(marker._generator.seen), 1)
        self.assertNotAlmostEqual(float(np.abs(sf.read(self.path)[0] - speechlike(1.0)).max()), 0.0, places=4)  # the file changed

    def test_switching_it_off_is_explicit_and_visible_in_the_result(self):
        with self.with_env("false"):
            result = self.router._watermark_output(self.path)
        self.assertEqual(result["applied"], False)
        self.assertIn("WATERMARK_ENABLED", result["reason"])

    def test_a_mark_that_cannot_be_read_back_fails_the_request(self):
        marker = marker_with(probability=0.0)
        with self.with_env("true"), mock.patch("watermark_engine.shared_watermarker", return_value=marker):
            with self.assertRaisesRegex(RuntimeError, "could not be read back"):
                self.router._watermark_output(self.path)

    def test_preflight_demands_the_watermark_weights_unless_it_is_switched_off(self):
        from model_registry import ModelWeightStatus
        from voice_engine import ModelWeightsMissing

        missing = [ModelWeightStatus("audioseal", "AudioSeal", "x", False, 0, "not fetched", fix="fetch it")]
        with mock.patch("model_registry.audit_model_weights", return_value=missing):
            with self.with_env("true"), self.assertRaises(ModelWeightsMissing):
                self.router.preflight("kokoro", None, "en")
            with self.with_env("false"):
                self.router.preflight("kokoro", None, "en")  # opt-out: no demand


# --------------------------------------------------------------------------- the real model
def _real_weights_present() -> bool:
    try:
        from model_registry import check_audioseal

        return check_audioseal().present
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(_real_weights_present(), "AudioSeal weights are not on disk (python scripts/fetch_models.py --only audioseal)")
class RealAudioSealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        w.reset_key_cache()
        cls.patcher = mock.patch("watermark_engine.signing_key", return_value=KEY)
        cls.patcher.start()
        cls.marker = AudioWatermarker()
        cls.clip = speechlike(3.0)
        cls.marked = cls.marker.embed(cls.clip, 24000)

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_marked_audio_is_detected_with_every_bit_and_the_original_is_not(self):
        report = self.marker.detect(self.marked, 24000)
        self.assertTrue(report.detected)
        self.assertEqual(report.bits_matching, 16)
        self.assertFalse(self.marker.detect(self.clip, 24000).detected)

    def test_the_mark_is_quiet(self):
        mark = self.marked - self.clip
        ratio_db = 10 * np.log10(np.sum(self.clip**2) / np.sum(mark**2))
        self.assertGreater(ratio_db, 20)  # at least 20 dB below the speech

    def test_it_survives_a_16_bit_wav_round_trip_and_a_resample(self):
        with tempfile.TemporaryDirectory() as tmp:
            sf.write(Path(tmp) / "m.wav", self.marked, 24000, subtype="PCM_16")
            reread, rate = sf.read(Path(tmp) / "m.wav", dtype="float32")
        self.assertTrue(self.marker.detect(reread, rate).detected)
        there_and_back = w._resample(w._resample(self.marked, 24000, 8000), 8000, 24000)
        self.assertTrue(self.marker.detect(there_and_back, 24000).detected)

    def test_someone_elses_key_does_not_validate_our_clip(self):
        with mock.patch("watermark_engine.signing_key", return_value=b"a-different-key"):
            self.assertFalse(self.marker.detect(self.marked, 24000).detected)


if __name__ == "__main__":
    unittest.main()


class WatermarkDeviceTests(unittest.TestCase):
    """Both marks follow the project's device rule: AVATAR_DEVICE, else CUDA when present, else CPU (owner's GPU run, 8 Oct)."""

    def test_device_choice(self):
        import os
        from unittest import mock

        import video_watermark
        import watermark_engine

        for module, cls in ((watermark_engine, watermark_engine.AudioWatermarker), (video_watermark, video_watermark.VideoWatermarker)):
            with self.subTest(module=module.__name__):
                with mock.patch.dict(os.environ, {"AVATAR_DEVICE": ""}), mock.patch("torch.cuda.is_available", return_value=True):
                    self.assertEqual(cls().device, "cuda")
                with mock.patch.dict(os.environ, {"AVATAR_DEVICE": ""}), mock.patch("torch.cuda.is_available", return_value=False):
                    self.assertEqual(cls().device, "cpu")
                with mock.patch.dict(os.environ, {"AVATAR_DEVICE": "cpu"}), mock.patch("torch.cuda.is_available", return_value=True):
                    self.assertEqual(cls().device, "cpu")
                self.assertEqual(cls(device="cpu").device, "cpu")
