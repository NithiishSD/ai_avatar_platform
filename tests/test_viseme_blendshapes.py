"""Viseme and emotion tables (G2-01, G3-02)."""

import unittest

from alignment_engine import PhonemeToVisemeMapper
from viseme_blendshapes import (
    CANONICAL_VISEMES,
    CLOSURE_VISEMES,
    EMOTION_BLENDSHAPES,
    OPENING_SHAPES,
    VISEME_BLENDSHAPES,
    emotion_weights,
    resolve_viseme,
    viseme_weights,
)


class VisemeTableTests(unittest.TestCase):
    def test_every_aligner_viseme_has_a_mouth_shape(self):
        # The contract between the two halves: nothing the aligner can emit
        # may be unknown to the renderer.
        for viseme in PhonemeToVisemeMapper.get_supported_visemes():
            canonical, known = resolve_viseme(viseme)
            self.assertTrue(known, viseme)
            self.assertIn(canonical, VISEME_BLENDSHAPES)

    def test_table_covers_exactly_the_fifteen(self):
        self.assertEqual(len(CANONICAL_VISEMES), 15)
        self.assertEqual(set(VISEME_BLENDSHAPES), set(CANONICAL_VISEMES))

    def test_weights_are_valid_and_paired_shapes_are_expanded(self):
        for viseme, shapes in VISEME_BLENDSHAPES.items():
            for name, weight in shapes.items():
                self.assertTrue(0.0 < weight <= 1.0, (viseme, name))
            self.assertNotIn("mouthSmile", shapes)
        self.assertIn("mouthPressLeft", VISEME_BLENDSHAPES["viseme_PP"])
        self.assertIn("mouthPressRight", VISEME_BLENDSHAPES["viseme_PP"])

    def test_silence_is_the_rest_pose_and_closures_do_not_open_the_jaw(self):
        self.assertEqual(VISEME_BLENDSHAPES["viseme_sil"], {})
        self.assertNotIn("jawOpen", VISEME_BLENDSHAPES["viseme_PP"])
        self.assertEqual(CLOSURE_VISEMES["viseme_PP"], 1.0)
        self.assertIn("jawOpen", OPENING_SHAPES)

    def test_open_vowel_has_the_widest_jaw(self):
        jaws = {v: s.get("jawOpen", 0.0) for v, s in VISEME_BLENDSHAPES.items()}
        self.assertEqual(max(jaws, key=jaws.get), "viseme_aa")

    def test_roadmap_example_alias_resolves(self):
        self.assertEqual(resolve_viseme("viseme_L"), ("viseme_nn", True))
        self.assertEqual(viseme_weights("viseme_A"), VISEME_BLENDSHAPES["viseme_aa"])

    def test_unknown_viseme_is_reported_not_guessed(self):
        self.assertEqual(resolve_viseme("viseme_B"), ("viseme_sil", False))
        self.assertEqual(viseme_weights("nonsense"), {})


class EmotionWeightTests(unittest.TestCase):
    def test_empty_and_neutral_vectors_leave_the_face_alone(self):
        self.assertEqual(emotion_weights(None), {})
        self.assertEqual(emotion_weights({"happy": 0.0, "neutral": 1.0, "eyeblinkRate": 1.0}), {})

    def test_joy_and_sorrow_move_different_shapes(self):
        joy = emotion_weights({"joy": 1.0})
        sorrow = emotion_weights({"sorrow": 1.0})
        self.assertGreater(joy["mouthSmileLeft"], 0.4)
        self.assertNotIn("mouthSmileLeft", sorrow)
        self.assertGreater(sorrow["browInnerUp"], 0.4)

    def test_emotions_do_not_part_the_lips(self):
        for emotion, shapes in EMOTION_BLENDSHAPES.items():
            self.assertFalse(set(shapes) & set(OPENING_SHAPES), emotion)

    def test_happy_is_used_only_without_a_named_emotion(self):
        only_happy = emotion_weights({"happy": 1.0})
        self.assertAlmostEqual(only_happy["mouthSmileLeft"], 0.5)
        both = emotion_weights({"happy": 1.0, "joy": 1.0})
        self.assertAlmostEqual(both["mouthSmileLeft"], EMOTION_BLENDSHAPES["joy"]["mouthSmileLeft"])

    def test_weights_scale_clamp_and_ignore_garbage(self):
        half = emotion_weights({"joy": 0.5})
        self.assertAlmostEqual(half["mouthSmileLeft"], 0.5 * EMOTION_BLENDSHAPES["joy"]["mouthSmileLeft"])
        stacked = emotion_weights({"joy": 1.0, "excitement": 1.0, "calm": 1.0})
        self.assertLessEqual(max(stacked.values()), 1.0)
        self.assertEqual(emotion_weights({"joy": "loud", "anger": None}), {})


if __name__ == "__main__":
    unittest.main()
