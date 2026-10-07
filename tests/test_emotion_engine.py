"""Tests for the Phase 3 emotion prosody engine. Pure DSP, no models."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from emotion_engine import (
    PRESETS,
    EmotionProsodyEngine,
    ProsodyParams,
    blend_prosody,
    dominant_emotion,
    normalize_vector,
    preset_catalogue,
    to_render_emotion_vector,
)
from contracts import EmotionVector


def make_speech_like(duration: float = 2.0, sample_rate: int = 24000) -> np.ndarray:
    """A voiced test tone: 140 Hz fundamental plus harmonics, amplitude-modulated."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    signal = sum(
        (1.0 / harmonic) * np.sin(2 * np.pi * 140 * harmonic * t)
        for harmonic in range(1, 6)
    )
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    voiced = signal * envelope
    # Normalize to a modest peak so the engine's clip guard never engages and
    # the loudness assertions measure the emotion's energy setting alone.
    return (0.3 * voiced / np.max(np.abs(voiced))).astype(np.float32)


class NormalizeVectorTests(unittest.TestCase):
    def test_no_input_is_pure_neutral(self):
        self.assertEqual(normalize_vector(), {"neutral": 1.0})

    def test_named_emotion_at_full_intensity(self):
        self.assertEqual(normalize_vector(emotion="joy", intensity=1.0), {"joy": 1.0})

    def test_partial_intensity_keeps_the_remainder_neutral(self):
        vector = normalize_vector(emotion="joy", intensity=0.4)
        self.assertAlmostEqual(vector["joy"], 0.4)
        self.assertAlmostEqual(vector["neutral"], 0.6)

    def test_oversized_vector_is_renormalized_to_one(self):
        vector = normalize_vector(vector={"joy": 2.0, "anger": 2.0})
        self.assertAlmostEqual(sum(vector.values()), 1.0, places=5)
        self.assertAlmostEqual(vector["joy"], 0.5)

    def test_unknown_names_are_dropped(self):
        vector = normalize_vector(vector={"smugness": 1.0, "joy": 0.5})
        self.assertNotIn("smugness", vector)
        self.assertAlmostEqual(vector["joy"], 0.5)

    def test_negative_weights_are_clamped(self):
        vector = normalize_vector(vector={"joy": -1.0, "anger": 0.5})
        self.assertNotIn("joy", vector)
        self.assertAlmostEqual(vector["anger"], 0.5)

    def test_explicit_vector_wins_over_named_emotion(self):
        vector = normalize_vector(vector={"sorrow": 1.0}, emotion="joy")
        self.assertEqual(vector, {"sorrow": 1.0})


class BlendProsodyTests(unittest.TestCase):
    def test_neutral_blend_is_the_identity_transform(self):
        params = blend_prosody({"neutral": 1.0})
        self.assertAlmostEqual(params.pitch_semitones, 0.0)
        self.assertAlmostEqual(params.rate, 1.0)
        self.assertAlmostEqual(params.energy, 1.0)

    def test_full_emotion_matches_its_preset(self):
        params = blend_prosody({"sorrow": 1.0})
        self.assertAlmostEqual(params.pitch_semitones, PRESETS["sorrow"].prosody.pitch_semitones)
        self.assertAlmostEqual(params.rate, PRESETS["sorrow"].prosody.rate)

    def test_half_strength_lands_between_preset_and_neutral(self):
        params = blend_prosody({"sorrow": 0.5, "neutral": 0.5})
        self.assertAlmostEqual(params.pitch_semitones, PRESETS["sorrow"].prosody.pitch_semitones / 2)
        expected_rate = 0.5 * PRESETS["sorrow"].prosody.rate + 0.5
        self.assertAlmostEqual(params.rate, expected_rate, places=3)

    def test_blend_averages_two_emotions(self):
        params = blend_prosody({"joy": 0.5, "authority": 0.5})
        expected = 0.5 * (
            PRESETS["joy"].prosody.pitch_semitones
            + PRESETS["authority"].prosody.pitch_semitones
        )
        self.assertAlmostEqual(params.pitch_semitones, expected, places=3)

    def test_emotion_directions_are_coherent(self):
        """High-arousal emotions speed up and rise; sorrow slows down and falls."""
        joy = blend_prosody({"joy": 1.0})
        sorrow = blend_prosody({"sorrow": 1.0})
        self.assertGreater(joy.rate, 1.0)
        self.assertGreater(joy.pitch_semitones, 0.0)
        self.assertLess(sorrow.rate, 1.0)
        self.assertLess(sorrow.pitch_semitones, 0.0)
        self.assertLess(sorrow.contour, 0.0)

    def test_authority_is_low_pitched_but_not_quiet(self):
        params = blend_prosody({"authority": 1.0})
        self.assertLess(params.pitch_semitones, 0.0)
        self.assertGreater(params.energy, 1.0)

    def test_parameters_stay_inside_safe_ranges(self):
        for name in PRESETS:
            with self.subTest(emotion=name):
                params = blend_prosody({name: 1.0})
                self.assertLessEqual(abs(params.pitch_semitones), 4.0)
                self.assertGreaterEqual(params.rate, 0.7)
                self.assertLessEqual(params.rate, 1.35)


class RenderVectorProjectionTests(unittest.TestCase):
    def test_projection_validates_against_the_frozen_contract(self):
        payload = to_render_emotion_vector({"joy": 0.6, "authority": 0.4})
        vector = EmotionVector.model_validate(payload)
        self.assertAlmostEqual(vector.happy + vector.neutral, 1.0, places=3)
        self.assertAlmostEqual(vector.joy, 0.6)
        self.assertAlmostEqual(vector.authority, 0.4)

    def test_joy_reads_as_happy_and_sorrow_does_not(self):
        self.assertGreater(to_render_emotion_vector({"joy": 1.0})["happy"], 0.9)
        self.assertLess(to_render_emotion_vector({"sorrow": 1.0})["happy"], 0.1)

    def test_neutral_projection_is_valid(self):
        payload = to_render_emotion_vector({"neutral": 1.0})
        EmotionVector.model_validate(payload)
        self.assertEqual(payload["happy"], 0.0)
        self.assertEqual(payload["neutral"], 1.0)

    def test_blink_rate_stays_inside_contract_bounds(self):
        for name in PRESETS:
            with self.subTest(emotion=name):
                rate = to_render_emotion_vector({name: 1.0})["eyeblinkRate"]
                self.assertGreaterEqual(rate, 0.0)
                self.assertLessEqual(rate, 10.0)

    def test_dominant_emotion_ignores_neutral(self):
        self.assertEqual(dominant_emotion({"neutral": 0.9, "anger": 0.1})[0], "anger")
        self.assertEqual(dominant_emotion({"neutral": 1.0})[0], "neutral")


class ProsodyTransformTests(unittest.TestCase):
    def setUp(self):
        self.engine = EmotionProsodyEngine()
        self.sample_rate = 24000
        self.audio = make_speech_like(2.0, self.sample_rate)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "clip.wav"
        sf.write(self.path, self.audio, self.sample_rate)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def rms(signal: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.asarray(signal, dtype=np.float64) ** 2)))

    def test_neutral_leaves_the_file_untouched(self):
        before = self.path.read_bytes()
        result = self.engine.apply_to_file(self.path, emotion="neutral")
        self.assertFalse(result.applied)
        self.assertEqual(self.path.read_bytes(), before)

    def test_energy_is_the_only_thing_setting_loudness(self):
        """Stretching and shifting change level as a side effect; it is undone."""
        for name in ("joy", "anger", "sorrow", "authority", "calm", "excitement"):
            with self.subTest(emotion=name):
                sf.write(self.path, self.audio, self.sample_rate)
                self.engine.apply_to_file(self.path, emotion=name)
                processed, _ = sf.read(self.path, dtype="float32")
                ratio = self.rms(processed) / self.rms(self.audio)
                self.assertAlmostEqual(
                    ratio, PRESETS[name].prosody.energy, delta=0.08
                )

    def test_rate_changes_duration_in_the_expected_direction(self):
        sf.write(self.path, self.audio, self.sample_rate)
        result = self.engine.apply_to_file(self.path, emotion="sorrow")
        self.assertGreater(result.duration_seconds, 2.0)  # sorrow is slower

        sf.write(self.path, self.audio, self.sample_rate)
        result = self.engine.apply_to_file(self.path, emotion="anger")
        self.assertLess(result.duration_seconds, 2.0)  # anger is faster

    def test_explicit_speed_multiplies_the_emotion_rate(self):
        sf.write(self.path, self.audio, self.sample_rate)
        result = self.engine.apply_to_file(self.path, emotion="neutral", extra_rate=2.0)
        self.assertTrue(result.applied)
        self.assertAlmostEqual(result.duration_seconds, 1.0, delta=0.1)

    def test_output_never_clips(self):
        loud = np.clip(self.audio * 3.0, -1.0, 1.0)
        sf.write(self.path, loud, self.sample_rate)
        self.engine.apply_to_file(self.path, emotion="anger")
        processed, _ = sf.read(self.path, dtype="float32")
        self.assertLessEqual(float(np.max(np.abs(processed))), 1.0)

    def test_transform_is_deterministic(self):
        sf.write(self.path, self.audio, self.sample_rate)
        self.engine.apply_to_file(self.path, emotion="joy")
        first, _ = sf.read(self.path, dtype="float32")

        sf.write(self.path, self.audio, self.sample_rate)
        self.engine.apply_to_file(self.path, emotion="joy")
        second, _ = sf.read(self.path, dtype="float32")
        np.testing.assert_allclose(first, second)

    def test_pitch_shift_moves_the_fundamental(self):
        """Sorrow drops 2 semitones, so the 140 Hz fundamental should fall."""
        sf.write(self.path, self.audio, self.sample_rate)
        self.engine.apply_to_file(self.path, emotion="sorrow")
        processed, sample_rate = sf.read(self.path, dtype="float32")

        def fundamental(signal: np.ndarray) -> float:
            spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
            freqs = np.fft.rfftfreq(len(signal), 1.0 / sample_rate)
            band = (freqs > 80) & (freqs < 220)
            return float(freqs[band][np.argmax(spectrum[band])])

        expected = 140.0 * math.pow(2.0, PRESETS["sorrow"].prosody.pitch_semitones / 12.0)
        self.assertAlmostEqual(fundamental(processed), expected, delta=8.0)

    def test_short_clip_survives_the_contour_stage(self):
        short = make_speech_like(0.2, self.sample_rate)
        sf.write(self.path, short, self.sample_rate)
        result = self.engine.apply_to_file(self.path, emotion="joy")
        self.assertTrue(result.applied)
        processed, _ = sf.read(self.path, dtype="float32")
        self.assertGreater(processed.size, 0)

    def test_identity_params_are_detected(self):
        self.assertTrue(EmotionProsodyEngine._is_identity(ProsodyParams()))
        self.assertFalse(
            EmotionProsodyEngine._is_identity(ProsodyParams(pitch_semitones=2.0))
        )


class CatalogueTests(unittest.TestCase):
    def test_catalogue_exposes_every_roadmap_emotion(self):
        names = {preset["name"] for preset in preset_catalogue()}
        self.assertTrue({"joy", "anger", "sorrow", "authority"}.issubset(names))

    def test_catalogue_entries_are_json_serializable(self):
        import json

        json.dumps(preset_catalogue())


if __name__ == "__main__":
    unittest.main()
