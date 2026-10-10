"""Temporal jitter metric (N-09). No model: landmark tracks are synthetic."""

import unittest

import numpy as np

import jitter_metric
from jitter_metric import JitterError, score_landmark_track

W = H = 500


def face(dx=0.0, dy=0.0, mouth=0.0):
    """478 normalised landmarks: eyes 100 px apart, anchors at fixed spots."""
    points = np.full((478, 3), 0.5)
    points[33] = [0.40, 0.45, 0]
    points[263] = [0.60, 0.45, 0]  # inter-ocular = 0.2 * 500 = 100 px
    for i, index in enumerate(jitter_metric.ANCHORS):
        points[index] = [0.3 + 0.02 * i, 0.2 + 0.01 * i, 0]
    points[:, 0] += dx
    points[:, 1] += dy
    points[14] = [0.5, 0.7 + mouth, 0]  # a lip point: not an anchor, free to move
    return points


class ScoreTests(unittest.TestCase):
    def test_a_perfectly_still_face_scores_zero(self):
        score = score_landmark_track([face()] * 10, W, H)
        self.assertEqual(score.mean_pct, 0.0)
        self.assertTrue(score.meets_target)

    def test_speech_movement_is_not_counted_as_jitter(self):
        track = [face(mouth=0.05 * (i % 2)) for i in range(10)]
        self.assertEqual(score_landmark_track(track, W, H).mean_pct, 0.0)

    def test_shake_is_measured_as_a_percentage_of_inter_ocular_distance(self):
        # Alternating by 1 px on a 100 px eye gap = 1% per step.
        track = [face(dx=(i % 2) / W) for i in range(11)]
        score = score_landmark_track(track, W, H)
        self.assertAlmostEqual(score.mean_pct, 1.0, places=6)
        self.assertTrue(score.meets_target)

    def test_target_is_rejected_when_the_mean_is_two_percent_or_more(self):
        track = [face(dx=2.0 * (i % 2) / W) for i in range(11)]  # 2 px = 2%
        score = score_landmark_track(track, W, H)
        self.assertFalse(score.meets_target)
        self.assertEqual(score.to_dict()["meetsTarget"], False)

    def test_result_states_its_method(self):
        self.assertIn("Upper bound", score_landmark_track([face()] * 3, W, H).to_dict()["method"])

    def test_missing_faces_break_the_chain_and_are_reported(self):
        # The jump across the gap is not compared, so it cannot count as jitter.
        track = [face(), face(), None, face(dx=0.1), face(dx=0.1)]
        score = score_landmark_track(track, W, H)
        self.assertEqual(score.mean_pct, 0.0)
        self.assertEqual(score.frames_without_face, 1)
        self.assertTrue(score.warnings)

    def test_nothing_to_compare_is_an_error_not_a_zero(self):
        for track in ([], [face()], [face(), None, face()], [None, None]):
            with self.assertRaises(JitterError):
                score_landmark_track(track, W, H)

    def test_score_video_wires_frames_to_the_detector(self):
        frames = [np.zeros((H, W, 3), dtype=np.uint8)] * 4
        calls = []

        def detect(frame):
            calls.append(1)
            return face()

        score = jitter_metric.score_video(None, landmark_fn=detect, frames=frames)
        self.assertEqual(len(calls), 4)
        self.assertEqual(score.frames, 4)


if __name__ == "__main__":
    unittest.main()
