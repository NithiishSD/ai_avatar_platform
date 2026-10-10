"""Tests for the visual metrics (N-12, N-13) with the networks mocked: the geometry and the bookkeeping."""

import unittest
from unittest import mock

import numpy as np

import visual_metrics


def fake_mesh(count=478):
    """A mesh where landmark i sits at (i, 2i): distinct points, so index mix-ups are visible."""
    return [(i, 2 * i) for i in range(count)]


class GeometryTests(unittest.TestCase):
    def test_five_points_follow_sface_order_and_use_eye_corner_midpoints(self):
        mesh = fake_mesh()
        points = visual_metrics.five_points(mesh)
        self.assertEqual(points.shape, (5, 2))
        np.testing.assert_allclose(points[0], [(33 + 133) / 2, (66 + 266) / 2])   # subject's right eye (image left)
        np.testing.assert_allclose(points[1], [(362 + 263) / 2, (724 + 526) / 2])  # subject's left eye
        np.testing.assert_allclose(points[2], mesh[1])                              # nose tip
        np.testing.assert_allclose(points[3], mesh[61])                             # right mouth corner
        np.testing.assert_allclose(points[4], mesh[291])                            # left mouth corner

    def test_detection_row_is_box_five_points_and_score(self):
        row = visual_metrics.detection_row(fake_mesh())
        self.assertEqual(row.shape, (15,))
        np.testing.assert_allclose(row[:4], [0, 0, 477, 954])  # box spans the whole mesh
        self.assertEqual(row[-1], 1.0)


class ScoreVideoTests(unittest.TestCase):
    def scorer(self, embeddings, cosines, lpips_values):
        scorer = visual_metrics.VisualScorer(sface_path="unused")
        scorer.embedding = mock.Mock(side_effect=embeddings)
        scorer.cosine = mock.Mock(side_effect=cosines)
        scorer.lpips_distance = mock.Mock(side_effect=lpips_values)
        return scorer

    def frames(self, n):
        return (np.zeros((8, 8, 3), np.uint8) for _ in range(n))

    def test_samples_every_nth_frame_and_reports_both_metrics_against_their_targets(self):
        src = np.zeros((16, 16, 3), np.uint8)
        # source embedding, then frames 0, 5 (of 10), then the other person
        scorer = self.scorer(["src", "f0", "f5", "other"], [0.95, 0.93, 0.2], [0.05, 0.07])
        with mock.patch("video_io.read_frames", return_value=self.frames(10)):
            report = scorer.score_video("v.mp4", src, other_person_rgb=src, every=5)
        self.assertEqual(report["framesScored"], 2)
        self.assertEqual(report["lpips"]["n"], 2)
        self.assertTrue(report["lpips"]["meetsTarget"])        # mean 0.06 < 0.1
        self.assertEqual(report["identity"]["meanPercent"], 94.0)
        self.assertTrue(report["identity"]["meetsTarget"])     # 0.94 >= 0.90
        self.assertEqual(report["identity"]["differentPersonCosine"], 0.2)
        self.assertEqual(report["identity"]["framesAboveThreshold"], 2)

    def test_targets_are_not_met_when_numbers_are_bad(self):
        scorer = self.scorer(["src", "f0"], [0.5], [0.3])
        with mock.patch("video_io.read_frames", return_value=self.frames(1)):
            report = scorer.score_video("v.mp4", np.zeros((8, 8, 3), np.uint8), every=1)
        self.assertFalse(report["lpips"]["meetsTarget"])
        self.assertFalse(report["identity"]["meetsTarget"])

    def test_frames_without_a_face_are_counted_and_never_averaged_in_as_zero(self):
        scorer = self.scorer(["src", None, "f1"], [0.9], [0.05, 0.05])
        with mock.patch("video_io.read_frames", return_value=self.frames(2)):
            report = scorer.score_video("v.mp4", np.zeros((8, 8, 3), np.uint8), every=1)
        self.assertEqual(report["identity"]["framesWithoutFace"], 1)
        self.assertEqual(report["identity"]["cosine"]["n"], 1)
        self.assertEqual(report["identity"]["meanPercent"], 90.0)

    def test_no_face_in_the_source_photo_is_refused_with_the_reason(self):
        scorer = self.scorer([None], [], [])
        with self.assertRaisesRegex(visual_metrics.VisualMetricError, "no face found in the source photo"):
            scorer.score_video("v.mp4", np.zeros((8, 8, 3), np.uint8))

    def test_an_empty_video_is_refused(self):
        scorer = self.scorer(["src"], [], [])
        with mock.patch("video_io.read_frames", return_value=iter(())):
            with self.assertRaisesRegex(visual_metrics.VisualMetricError, "no frames"):
                scorer.score_video("v.mp4", np.zeros((8, 8, 3), np.uint8))

    def test_missing_sface_weights_say_how_to_fetch_them(self):
        with self.assertRaisesRegex(visual_metrics.VisualMetricError, "fetch_vision_models.py"):
            visual_metrics.VisualScorer(sface_path="/nonexistent/sface.onnx")._sface()


if __name__ == "__main__":
    unittest.main()
