"""Blendshape portrait animator (G2-02), rigged on the synthetic-face fixture."""

import unittest

import numpy as np

from face_warp import FaceWarpError, PortraitAnimator, mouth_params
from vision_fixtures import gradient_image, synthetic_analysis


class MouthParamTests(unittest.TestCase):
    def test_rest_is_all_zero(self):
        params = mouth_params({})
        self.assertEqual((params.open, params.wide, params.smile, params.press), (0.0, 0.0, 0.0, 0.0))

    def test_stretch_widens_and_pucker_narrows(self):
        self.assertGreater(mouth_params({"mouthStretchLeft": 1, "mouthStretchRight": 1}).wide, 0.5)
        self.assertLess(mouth_params({"mouthPucker": 1.0}).wide, -0.5)

    def test_smile_and_frown_oppose(self):
        self.assertGreater(mouth_params({"mouthSmileLeft": 1, "mouthSmileRight": 1}).smile, 0.9)
        self.assertLess(mouth_params({"mouthFrownLeft": 1, "mouthFrownRight": 1}).smile, -0.9)


class PortraitAnimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = gradient_image()
        cls.analysis = synthetic_analysis()
        cls.animator = PortraitAnimator(cls.image, cls.analysis)
        lm = np.array([(x * 512, y * 512) for x, y, _ in cls.analysis.landmarks[:468]])
        cls.lm = lm

    def _changed(self, frame):
        return np.abs(frame.astype(int) - self.image.astype(int)).max(axis=2) > 2

    def test_too_few_landmarks_cannot_be_rigged(self):
        short = synthetic_analysis()
        short.landmarks = short.landmarks[:100]
        with self.assertRaises(FaceWarpError):
            PortraitAnimator(self.image, short)

    def test_rest_pose_is_the_photo(self):
        np.testing.assert_array_equal(self.animator.render({}), self.image)
        np.testing.assert_array_equal(self.animator.render(None), self.image)
        self.assertEqual(self.animator.render({}).shape, self.image.shape)

    def test_render_does_not_modify_the_source(self):
        before = self.image.copy()
        self.animator.render({"jawOpen": 0.8})
        np.testing.assert_array_equal(self.image, before)

    def test_open_jaw_moves_the_lower_lip_down_and_chin_with_it(self):
        disp = self.animator.displacements({"jawOpen": 0.65})
        self.assertGreater(disp[14, 1], 4.0)      # inner lower lip: down
        self.assertGreater(disp[152, 1], 1.0)     # chin follows
        self.assertLess(abs(disp[13, 1]), abs(disp[14, 1]) * 0.35)  # upper lip barely moves
        self.assertLess(np.abs(disp[10]).max(), 0.5)                # forehead does not

    def test_larger_weight_opens_wider(self):
        small = self.animator.displacements({"jawOpen": 0.2})[14, 1]
        large = self.animator.displacements({"jawOpen": 0.8})[14, 1]
        self.assertGreater(large, small * 2)

    def test_change_is_confined_to_the_face(self):
        changed = self._changed(self.animator.render({"jawOpen": 0.65, "mouthFunnel": 0.4}))
        self.assertGreater(changed.sum(), 200)
        ys, xs = np.nonzero(changed)
        box = self.analysis.bounding_box
        margin = 0.45 * box.height
        self.assertGreaterEqual(xs.min(), box.x - margin)
        self.assertLessEqual(xs.max(), box.x + box.width + margin)
        self.assertGreaterEqual(ys.min(), box.y - margin)
        self.assertLessEqual(ys.max(), box.y + box.height + margin)
        # A mouth shape moves the lower face; the eyes and brows stay put.
        self.assertGreater(ys.min(), self.lm[168, 1])

    def test_open_mouth_shows_an_interior_that_is_not_in_the_photo(self):
        frame = self.animator.render({"jawOpen": 0.65, "mouthLowerDownLeft": 0.25, "mouthLowerDownRight": 0.25})
        x, y = int(self.lm[13, 0]), int(self.lm[13, 1])
        patch = frame[y + 2 : y + 8, x - 4 : x + 4].astype(int)
        source = self.image[y + 2 : y + 8, x - 4 : x + 4].astype(int)
        self.assertGreater(np.abs(patch - source).mean(), 15)

    def test_blink_changes_the_eyes_only(self):
        changed = self._changed(self.animator.render({"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0}))
        self.assertGreater(changed.sum(), 50)
        ys, _ = np.nonzero(changed)
        self.assertLess(ys.max(), self.lm[1, 1])  # nothing below the nose tip

    def test_one_eye_can_blink_alone(self):
        changed = self._changed(self.animator.render({"eyeBlinkRight": 1.0}))
        _, xs = np.nonzero(changed)
        self.assertTrue(xs.max() < self.lm[168, 0] or xs.min() > self.lm[168, 0])

    def test_smile_lifts_the_corners_and_frown_drops_them(self):
        smile = self.animator.displacements({"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8})
        frown = self.animator.displacements({"mouthFrownLeft": 0.8, "mouthFrownRight": 0.8})
        for corner in (61, 291):
            self.assertLess(smile[corner, 1], -0.5)
            self.assertGreater(frown[corner, 1], 0.5)

    def test_pucker_narrows_and_stretch_widens_the_mouth(self):
        def width(weights):
            d = self.animator.displacements(weights)
            return (self.lm[291, 0] + d[291, 0]) - (self.lm[61, 0] + d[61, 0])

        rest = width({})
        self.assertLess(width({"mouthPucker": 0.8}), rest - 2)
        self.assertGreater(width({"mouthStretchLeft": 0.8, "mouthStretchRight": 0.8}), rest + 2)

    def test_brows_move(self):
        down = self.animator.displacements({"browDownLeft": 1.0, "browDownRight": 1.0})
        up = self.animator.displacements({"browInnerUp": 1.0})
        self.assertGreater(down[105, 1], 0.5)
        self.assertLess(up[107, 1], -0.5)

    def test_output_is_deterministic_and_well_formed(self):
        weights = {"jawOpen": 0.4, "mouthFunnel": 0.3, "eyeBlinkLeft": 0.5, "eyeBlinkRight": 0.5}
        first, second = self.animator.render(weights), self.animator.render(weights)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(first.shape, (512, 512, 3))


if __name__ == "__main__":
    unittest.main()
