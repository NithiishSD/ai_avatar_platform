"""
Lighting, grade and standing-expression controls (R-44, N-15): ``look.py``, the ``AppearanceSpec``
contract, and the offsets in ``face_animation.build_animation``.

Pictures are synthetic (gradients, noise, a half-lit field) so every expected direction is known
without a face or a model. Each control has a test that it moves the number it is meant to move, in
the asked direction, and that the other controls it should not touch stay put.
"""

import unittest

import cv2
import numpy as np
from pydantic import ValidationError

import face_animation
import look
from contracts import EXPRESSION_SHAPES, AppearanceSpec, AvatarRenderJob


def luma(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))


def photo(seed: int = 0, size: int = 96) -> np.ndarray:
    """A mid-tone RGB picture with some texture and colour."""
    rng = np.random.default_rng(seed)
    base = np.stack([
        np.tile(np.linspace(80, 170, size), (size, 1)),
        np.tile(np.linspace(100, 150, size)[:, None], (1, size)),
        np.full((size, size), 120.0),
    ], axis=-1)
    return np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)


def spec(**kw) -> AppearanceSpec:
    return AppearanceSpec.model_validate(kw)


def apply(img, **kw):
    out, warnings = look.apply_look(img, spec(**kw))
    return out, warnings


class ContractTests(unittest.TestCase):
    def test_empty_spec_is_empty_and_changes_nothing(self):
        img = photo()
        self.assertTrue(AppearanceSpec().is_empty())
        self.assertTrue(spec(expression={}).is_empty())
        out, warnings = look.apply_look(img, AppearanceSpec())
        self.assertIs(out, img)
        self.assertEqual(warnings, [])
        self.assertIs(look.apply_look(img, None)[0], img)

    def test_out_of_range_and_unknown_fields_are_refused(self):
        for bad in ({"brightness": 1.5}, {"gamma": 0.1}, {"headRoll": 40}, {"skinSmoothing": -0.1},
                    {"expression": {"jawOpen": 1.2}}, {"expression": {"cheekPuff": 0.5}}, {"sparkle": 1}):
            with self.assertRaises(ValidationError, msg=str(bad)):
                spec(**bad)

    def test_a_job_without_appearance_validates_as_before_and_with_it_carries_it(self):
        base = {"jobId": "j", "avatarId": "a", "audioUrl": "file:///x.wav", "sampleRate": 24000, "durationSeconds": 1.0,
                "phonemeTimestamps": [{"phoneme": "AA", "viseme": "viseme_aa", "startMs": 0, "endMs": 500}],
                "emotionVector": {"happy": 0, "neutral": 1, "eyeblinkRate": 1}, "renderQuality": "PREVIEW", "targetFps": 25}
        self.assertIsNone(AvatarRenderJob.model_validate(base).appearance)
        job = AvatarRenderJob.model_validate({**base, "appearance": {"brightness": 0.2, "expression": {"browInnerUp": 0.5}}})
        self.assertAlmostEqual(job.appearance.brightness, 0.2)
        self.assertAlmostEqual(job.appearance.expression.browInnerUp, 0.5)

    def test_every_expression_shape_is_a_field(self):
        self.assertEqual(len(EXPRESSION_SHAPES), 25)
        for name in EXPRESSION_SHAPES:
            self.assertAlmostEqual(getattr(spec(expression={name: 0.3}).expression, name), 0.3)


class ToneTests(unittest.TestCase):
    def test_brightness_moves_the_mean_in_the_asked_direction_only(self):
        img = photo()
        up, _ = apply(img, brightness=0.6)
        down, _ = apply(img, brightness=-0.6)
        self.assertGreater(luma(up).mean(), luma(img).mean() + 10)
        self.assertLess(luma(down).mean(), luma(img).mean() - 10)

    def test_contrast_widens_the_spread_without_moving_the_mean(self):
        img = photo()
        hi, _ = apply(img, contrast=0.8)
        lo, _ = apply(img, contrast=-0.8)
        self.assertGreater(luma(hi).std(), luma(img).std() * 1.4)
        self.assertLess(luma(lo).std(), luma(img).std() * 0.5)
        self.assertAlmostEqual(float(luma(hi).mean()), float(luma(img).mean()), delta=4.0)

    def test_gamma_lifts_or_deepens_the_midtones_and_keeps_black_and_white(self):
        img = photo()
        bright, _ = apply(img, gamma=1.8)
        dark, _ = apply(img, gamma=0.6)
        self.assertGreater(luma(bright).mean(), luma(img).mean() + 8)
        self.assertLess(luma(dark).mean(), luma(img).mean() - 8)
        ends = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        out, _ = apply(ends, gamma=1.8)
        self.assertEqual(out.tolist(), ends.tolist())

    def test_saturation_to_minus_one_is_grey_and_plus_one_is_more_colourful(self):
        img = photo()
        grey, _ = apply(img, saturation=-1.0)
        vivid, _ = apply(img, saturation=0.8)

        def spread(a):
            return float(a.astype(np.float32).max(axis=2).mean() - a.astype(np.float32).min(axis=2).mean())

        self.assertLess(spread(grey), 1.0)
        self.assertGreater(spread(vivid), spread(img) * 1.3)

    def test_warmth_shifts_red_against_blue(self):
        img = photo()
        warm, _ = apply(img, warmth=0.8)
        cool, _ = apply(img, warmth=-0.8)

        def rb(a):
            return float(a[..., 0].mean() - a[..., 2].mean())

        self.assertGreater(rb(warm), rb(img) + 10)
        self.assertLess(rb(cool), rb(img) - 10)

    def test_tint_moves_green_against_the_other_channels(self):
        img = photo()
        magenta, _ = apply(img, tint=0.8)
        green, _ = apply(img, tint=-0.8)
        self.assertLess(magenta[..., 1].mean(), img[..., 1].mean() - 5)
        self.assertGreater(green[..., 1].mean(), img[..., 1].mean() + 5)


class AutoLightingTests(unittest.TestCase):
    def test_a_lopsided_picture_is_evened_out(self):
        # Left half in shadow, right half lit: what a window on one side does to a face.
        rng = np.random.default_rng(3)
        left = rng.normal(55, 12, (96, 48)).clip(0, 255)
        right = rng.normal(190, 12, (96, 48)).clip(0, 255)
        grey = np.hstack([left, right]).astype(np.uint8)
        img = np.stack([grey, grey, grey], axis=-1)
        out, _ = apply(img, autoLighting=True)

        def gap(a):
            y = luma(a)
            return float(abs(y[:, :48].mean() - y[:, 48:].mean()))

        self.assertLess(gap(out), gap(img) * 0.85)

    def test_a_colour_cast_is_reduced_and_a_neutral_picture_barely_moves(self):
        cast = photo().astype(np.float32)
        cast[..., 0] *= 1.0
        cast[..., 2] = np.clip(cast[..., 2] * 0.55, 0, 255)  # a strong yellow cast
        cast = cast.astype(np.uint8)
        out, _ = apply(cast, autoLighting=True)

        def imbalance(a):
            m = a.reshape(-1, 3).mean(axis=0)
            return float(m.max() - m.min())

        self.assertLess(imbalance(out), imbalance(cast))
        # a textured, already neutral picture keeps its average and its colour balance
        rng = np.random.default_rng(9)
        grey = np.clip(128 + rng.normal(0, 10, (64, 64)), 0, 255).astype(np.uint8)
        neutral = np.stack([grey, grey, grey], axis=-1)
        out, _ = apply(neutral, autoLighting=True)
        self.assertLess(abs(float(out.mean()) - float(neutral.mean())), 8.0)
        self.assertLess(imbalance(out), 3.0)


class DetailTests(unittest.TestCase):
    def points(self, size=96):
        # A square "face" in the middle, as landmark pixels.
        lo, hi = size // 4, 3 * size // 4
        return [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]

    def noisy(self, size=96):
        rng = np.random.default_rng(1)
        return np.clip(128 + rng.normal(0, 14, (size, size, 3)), 0, 255).astype(np.uint8)

    def test_skin_smoothing_reduces_texture_inside_the_face_and_leaves_the_outside(self):
        img = self.noisy()
        out, warnings = look.apply_look(img, spec(skinSmoothing=1.0), face_points=self.points())
        self.assertEqual(warnings, [])
        inside = (slice(36, 60), slice(36, 60))
        self.assertLess(float(out[inside].std()), float(img[inside].std()) * 0.7)
        # far from the face the feathered mask has faded to nothing
        self.assertTrue(np.array_equal(out[:6, :6], img[:6, :6]))

    def test_skin_smoothing_without_a_face_is_skipped_with_a_warning(self):
        img = self.noisy()
        out, warnings = look.apply_look(img, spec(skinSmoothing=0.8))
        self.assertTrue(np.array_equal(out, img))
        self.assertTrue(any("skinSmoothing" in w for w in warnings))

    def test_sharpness_increases_local_detail(self):
        soft = cv2.GaussianBlur(self.noisy(), (0, 0), 1.6)
        out, _ = apply(soft, sharpness=1.0)
        lap = lambda a: float(cv2.Laplacian(a, cv2.CV_32F).var())  # noqa: E731
        self.assertGreater(lap(out), lap(soft) * 1.5)

    def test_vignette_darkens_corners_not_the_centre(self):
        img = np.full((96, 96, 3), 180, dtype=np.uint8)
        out, _ = apply(img, vignette=1.0)
        self.assertLess(int(out[0, 0, 0]), 180 - 40)
        self.assertLessEqual(abs(int(out[48, 48, 0]) - 180), 2)

    def test_background_blur_blurs_only_where_the_mask_says_background(self):
        img = self.noisy()
        mask = np.zeros(img.shape[:2], dtype=np.float32)
        mask[:, 48:] = 1.0  # right half is "person"
        out, warnings = look.apply_look(img, spec(backgroundBlur=0.8), foreground=lambda _: mask)
        self.assertEqual(warnings, [])
        self.assertLess(float(out[:, :40].std()), float(img[:, :40].std()) * 0.5)
        self.assertTrue(np.array_equal(out[:, 60:], img[:, 60:]))

    def test_background_blur_without_a_segmenter_warns(self):
        img = self.noisy()
        out, warnings = look.apply_look(img, spec(backgroundBlur=0.5))
        self.assertTrue(np.array_equal(out, img))
        self.assertTrue(any("backgroundBlur" in w for w in warnings))


class OffsetTests(unittest.TestCase):
    def test_helpers_read_only_what_is_set(self):
        self.assertIsNone(look.expression_offsets(None))
        self.assertIsNone(look.head_offset(spec(brightness=0.3)))
        s = spec(headRoll=4, headShiftY=-0.02, expression={"browInnerUp": 0.5, "jawOpen": 0.0})
        self.assertEqual(look.head_offset(s), (4.0, 0.0, -0.02, 0.0))
        self.assertEqual(look.expression_offsets(s), {"browInnerUp": 0.5})
        self.assertEqual(look.describe(s), {"headRoll": 4.0, "headShiftY": -0.02, "expression": {"browInnerUp": 0.5, "jawOpen": 0.0}})
        self.assertIsNone(look.describe(AppearanceSpec()))


STAMPS = [{"phoneme": "AA", "viseme": "viseme_aa", "startMs": 0, "endMs": 400},
          {"phoneme": "M", "viseme": "viseme_PP", "startMs": 400, "endMs": 700}]


def track(**kw):
    return face_animation.build_animation(STAMPS, 0.8, 25, seed=1, **kw)


class AnimationOffsetTests(unittest.TestCase):
    def test_no_offsets_is_the_old_track_exactly(self):
        a, b = track(), track(expression_offsets=None, head_offset=None)
        self.assertEqual(a.names, b.names)
        self.assertTrue(np.array_equal(a.weights, b.weights))
        self.assertTrue(np.array_equal(a.head, b.head))

    def test_a_standing_expression_is_held_on_every_frame_and_pairs_expand(self):
        base = track(motion_intensity=0.0)
        out = track(motion_intensity=0.0, expression_offsets={"mouthSmile": 0.3, "eyeSquintLeft": 0.5})
        for name, value in (("mouthSmileLeft", 0.3), ("mouthSmileRight", 0.3), ("eyeSquintLeft", 0.5)):
            col = out.column(name)
            self.assertGreaterEqual(float(col.min()), value - 1e-6, name)
        # a shape the speech never touches is exactly the offset; the other eye is untouched
        self.assertTrue(np.allclose(out.column("eyeSquintLeft"), 0.5))
        self.assertTrue(np.allclose(out.column("eyeSquintRight"), base.column("eyeSquintRight")))

    def test_offsets_never_leave_the_zero_to_one_range(self):
        out = track(expression_offsets={"jawOpen": 1.0, "mouthSmile": 1.0})
        self.assertLessEqual(float(out.weights.max()), 1.0)

    def test_head_offset_adds_to_the_pose_and_creates_one_when_motion_is_off(self):
        moving = track(motion_intensity=1.0)
        shifted = track(motion_intensity=1.0, head_offset=(5.0, 0.0, 0.03, 0.0))
        self.assertTrue(np.allclose(shifted.head[:, 0], moving.head[:, 0] + 5.0, atol=1e-5))
        self.assertTrue(np.allclose(shifted.head[:, 2], moving.head[:, 2] + 0.03, atol=1e-5))
        self.assertTrue(np.array_equal(shifted.head[:, 4], moving.head[:, 4]))
        still = track(motion_intensity=0.0, head_offset=(-3.0, 0.0, 0.0, 0.05))
        self.assertIsNotNone(still.head)
        self.assertTrue(np.allclose(still.head[:, 0], -3.0))
        self.assertTrue(np.allclose(still.head[:, 3], 0.05))
        self.assertIsNone(track(motion_intensity=0.0).head)


if __name__ == "__main__":
    unittest.main()
