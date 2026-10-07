"""
Tests for the face analysis engine (Developer 2, Phase 1 / Gate 1).

These mock MediaPipe's landmarker rather than shipping a face photo. That is
partly the same discipline the audio side uses -- repeatable tests that do not
depend on a model download -- and partly an ethics constraint: a talking
avatar built from someone's face needs that person's consent, so the repo
should not carry a face fixture whose rights nobody checked. The first
candidate tried here was MediaPipe's own ``portrait.jpg`` test asset, whose
EXIF turned out to carry a White House Photo Office notice explicitly
forbidding manipulation.

The geometry is what these tests pin: Euler decomposition, bounding boxes,
crop expansion at image edges, and the array normalization every path shares.
"""

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from face_engine import (
    CLASSIC_MESH_POINTS,
    BackgroundSegmenter,
    BoundingBox,
    FaceAnalysis,
    FaceEngineUnavailable,
    FaceMeshEngine,
    HeadPose,
    NoFaceDetected,
    _as_rgb_array,
    _euler_from_matrix,
    _pose_from_landmarks,
)


def rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def as4x4(rotation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    return matrix


class Landmark:
    """Stands in for MediaPipe's NormalizedLandmark."""

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x, self.y, self.z = x, y, z


class Category:
    """Stands in for MediaPipe's blendshape Category."""

    def __init__(self, name: str, score: float):
        self.category_name, self.score = name, score


class Result:
    def __init__(self, landmarks=None, blendshapes=None, matrices=None):
        self.face_landmarks = landmarks or []
        self.face_blendshapes = blendshapes or []
        self.facial_transformation_matrixes = matrices or []


def mesh(count: int = 478, spread: float = 0.2) -> list:
    """A deterministic pseudo-face: points inside a centred box."""
    rng = np.random.default_rng(7)
    pts = 0.5 + (rng.random((count, 3)) - 0.5) * spread
    landmarks = [Landmark(float(x), float(y), float(z)) for x, y, z in pts]
    # Pin the indices the pose fallback reads so its output is predictable.
    landmarks[33] = Landmark(0.40, 0.45)   # left eye outer
    landmarks[263] = Landmark(0.60, 0.45)  # right eye outer
    landmarks[1] = Landmark(0.50, 0.52)    # nose tip
    landmarks[10] = Landmark(0.50, 0.30)   # forehead
    landmarks[152] = Landmark(0.50, 0.75)  # chin
    return landmarks


class EulerTests(unittest.TestCase):
    def test_identity_is_level(self):
        yaw, pitch, roll = _euler_from_matrix(np.eye(4))
        for value in (yaw, pitch, roll):
            self.assertAlmostEqual(value, 0.0, places=6)

    def test_rotation_about_y_is_reported_as_yaw(self):
        """Head turn left/right is the Y term, not the Z term."""
        yaw, pitch, roll = _euler_from_matrix(as4x4(rot_y(25)))
        self.assertAlmostEqual(yaw, 25.0, places=4)
        self.assertAlmostEqual(pitch, 0.0, places=4)
        self.assertAlmostEqual(roll, 0.0, places=4)

    def test_rotation_about_x_is_reported_as_pitch(self):
        yaw, pitch, roll = _euler_from_matrix(as4x4(rot_x(-15)))
        self.assertAlmostEqual(pitch, -15.0, places=4)
        self.assertAlmostEqual(yaw, 0.0, places=4)
        self.assertAlmostEqual(roll, 0.0, places=4)

    def test_rotation_about_z_is_reported_as_roll(self):
        yaw, pitch, roll = _euler_from_matrix(as4x4(rot_z(12)))
        self.assertAlmostEqual(roll, 12.0, places=4)
        self.assertAlmostEqual(yaw, 0.0, places=4)
        self.assertAlmostEqual(pitch, 0.0, places=4)

    def test_composed_rotation_round_trips(self):
        combined = rot_z(10) @ rot_y(20) @ rot_x(-8)
        yaw, pitch, roll = _euler_from_matrix(as4x4(combined))
        self.assertAlmostEqual(yaw, 20.0, places=3)
        self.assertAlmostEqual(pitch, -8.0, places=3)
        self.assertAlmostEqual(roll, 10.0, places=3)

    def test_gimbal_lock_does_not_produce_nan(self):
        """Looking straight up collapses sy; roll must degrade, not explode."""
        yaw, pitch, roll = _euler_from_matrix(as4x4(rot_y(90)))
        for value in (yaw, pitch, roll):
            self.assertFalse(math.isnan(value))
        self.assertAlmostEqual(abs(yaw), 90.0, places=3)


class ArrayNormalizationTests(unittest.TestCase):
    def test_grayscale_is_expanded_to_three_channels(self):
        out = _as_rgb_array(np.zeros((8, 6), dtype=np.uint8))
        self.assertEqual(out.shape, (8, 6, 3))

    def test_alpha_channel_is_dropped(self):
        out = _as_rgb_array(np.zeros((4, 4, 4), dtype=np.uint8))
        self.assertEqual(out.shape, (4, 4, 3))

    def test_float_input_is_clipped_to_uint8(self):
        raw = np.array([[[-20.0, 130.0, 900.0]]], dtype=np.float32)
        out = _as_rgb_array(raw)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(list(out[0, 0]), [0, 130, 255])

    def test_result_is_contiguous(self):
        """MediaPipe requires a contiguous buffer, so a view must be copied."""
        source = np.zeros((10, 10, 4), dtype=np.uint8)
        self.assertTrue(_as_rgb_array(source).flags["C_CONTIGUOUS"])

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            _as_rgb_array("/nonexistent/face.png")

    def test_unsupported_type_is_rejected(self):
        with self.assertRaises(TypeError):
            _as_rgb_array({"not": "an image"})


class BoundingBoxTests(unittest.TestCase):
    def box(self) -> BoundingBox:
        return BoundingBox(100, 200, 50, 80, image_width=500, image_height=400)

    def test_normalized_is_a_fraction_of_the_image(self):
        norm = self.box().normalized
        self.assertAlmostEqual(norm["x"], 0.2)
        self.assertAlmostEqual(norm["y"], 0.5)
        self.assertAlmostEqual(norm["width"], 0.1)
        self.assertAlmostEqual(norm["height"], 0.2)

    def test_zero_sized_image_does_not_divide_by_zero(self):
        norm = BoundingBox(0, 0, 0, 0, 0, 0).normalized
        self.assertEqual(norm["x"], 0.0)
        self.assertEqual(norm["width"], 0.0)

    def test_expanding_grows_by_a_fraction_of_the_box(self):
        grown = self.box().expanded(0.1)
        self.assertEqual(grown.x, 95)
        self.assertEqual(grown.y, 192)
        self.assertEqual(grown.width, 60)
        self.assertEqual(grown.height, 96)

    def test_expanding_clamps_to_the_image_edges(self):
        """A face against the frame edge must not crop outside the array."""
        edge = BoundingBox(0, 0, 100, 100, image_width=120, image_height=120)
        grown = edge.expanded(1.0)
        self.assertEqual(grown.x, 0)
        self.assertEqual(grown.y, 0)
        self.assertLessEqual(grown.x + grown.width, 120)
        self.assertLessEqual(grown.y + grown.height, 120)


class FaceAnalysisTests(unittest.TestCase):
    def analysis(self, count: int) -> FaceAnalysis:
        return FaceAnalysis(
            landmarks=[(0.5, 0.5, 0.0)] * count,
            bounding_box=BoundingBox(0, 0, 10, 10, 100, 100),
            head_pose=HeadPose(0, 0, 0, "test"),
            image_width=100,
            image_height=100,
        )

    def test_iris_landmarks_are_detected_above_the_classic_count(self):
        self.assertFalse(self.analysis(CLASSIC_MESH_POINTS).has_iris_landmarks)
        self.assertTrue(self.analysis(478).has_iris_landmarks)

    def test_dict_reports_both_counts(self):
        """The roadmap says 468; the refined model returns 478."""
        payload = self.analysis(478).to_dict(include_landmarks=False)
        self.assertEqual(payload["landmarkCount"], 478)
        self.assertEqual(payload["classicMeshPoints"], CLASSIC_MESH_POINTS)
        self.assertNotIn("landmarks", payload)

    def test_landmarks_can_be_omitted_from_the_payload(self):
        with_points = self.analysis(478).to_dict(include_landmarks=True)
        self.assertEqual(len(with_points["landmarks"]), 478)

    def test_pixel_landmarks_scale_to_the_image(self):
        analysis = FaceAnalysis(
            landmarks=[(0.25, 0.5, 0.0)],
            bounding_box=BoundingBox(0, 0, 1, 1, 200, 100),
            head_pose=HeadPose(0, 0, 0, "test"),
            image_width=200,
            image_height=100,
        )
        self.assertEqual(analysis.pixel_landmarks(), [(50, 50)])


class PoseFallbackTests(unittest.TestCase):
    def test_level_eyes_give_no_roll(self):
        points = [(lm.x, lm.y, 0.0) for lm in mesh()]
        pose = _pose_from_landmarks(points, 100, 100)
        self.assertAlmostEqual(pose.roll, 0.0, places=6)
        self.assertEqual(pose.source, "landmark-geometry")

    def test_tilted_eyes_produce_roll_signed_like_the_matrix(self):
        # The image-right eye dropped = a clockwise tilt on screen. Real
        # MediaPipe reports that as *negative* roll (+10 deg CCW -> +10.0), so
        # the fallback must too. This test used to expect a positive value,
        # which encoded the fallback's inverted sign.
        landmarks = mesh()
        landmarks[263] = Landmark(0.60, 0.55)
        points = [(lm.x, lm.y, 0.0) for lm in landmarks]
        self.assertLess(_pose_from_landmarks(points, 100, 100).roll, -10.0)

    def test_counter_clockwise_tilt_is_positive_roll(self):
        landmarks = mesh()
        landmarks[263] = Landmark(0.60, 0.35)  # image-right eye raised
        points = [(lm.x, lm.y, 0.0) for lm in landmarks]
        self.assertGreater(_pose_from_landmarks(points, 100, 100).roll, 10.0)

    def test_tilting_the_whole_face_changes_roll_not_yaw_or_pitch(self):
        # Rotating every landmark rigidly is a pure tilt. Measured along the
        # image axes, a 10 deg tilt used to read as 14 deg of yaw on a real
        # photo; in the face's own frame yaw and pitch must not move.
        points = [(lm.x, lm.y, 0.0) for lm in mesh()]
        upright = _pose_from_landmarks(points, 100, 100)
        angle = math.radians(-10.0)  # -10 in image coordinates = 10 deg CCW on screen
        cx, cy = 0.5, 0.5

        def turn(x, y):
            dx, dy = x - cx, y - cy
            return (cx + dx * math.cos(angle) - dy * math.sin(angle),
                    cy + dx * math.sin(angle) + dy * math.cos(angle), 0.0)

        tilted = _pose_from_landmarks([turn(x, y) for x, y, _ in points], 100, 100)
        self.assertAlmostEqual(tilted.roll - upright.roll, 10.0, places=4)
        self.assertAlmostEqual(tilted.yaw, upright.yaw, places=4)
        self.assertAlmostEqual(tilted.pitch, upright.pitch, places=4)

    def test_nose_toward_the_image_right_is_positive_yaw(self):
        landmarks = mesh()
        landmarks[1] = Landmark(landmarks[1].x + 0.05, landmarks[1].y)
        points = [(lm.x, lm.y, 0.0) for lm in landmarks]
        self.assertGreater(_pose_from_landmarks(points, 100, 100).yaw, 0.0)

    def test_pose_is_clamped_to_the_reliable_range(self):
        landmarks = mesh()
        landmarks[1] = Landmark(0.95, 0.52)  # nose far right of both eyes
        points = [(lm.x, lm.y, 0.0) for lm in landmarks]
        pose = _pose_from_landmarks(points, 100, 100)
        self.assertLessEqual(pose.yaw, 90.0)
        self.assertGreaterEqual(pose.yaw, -90.0)


class AnalyzeTests(unittest.TestCase):
    def engine_with(self, result: Result) -> FaceMeshEngine:
        engine = FaceMeshEngine()
        landmarker = mock.Mock()
        landmarker.detect.return_value = result
        engine._landmarker = landmarker
        return engine

    def image(self) -> np.ndarray:
        return np.zeros((100, 200, 3), dtype=np.uint8)

    def test_no_face_raises_rather_than_returning_none(self):
        """An empty result must not be mistakable for a neutral face."""
        engine = self.engine_with(Result(landmarks=[]))
        with self.assertRaises(NoFaceDetected):
            engine.analyze(self.image())

    def test_full_mesh_is_returned_with_pose_from_the_matrix(self):
        engine = self.engine_with(
            Result(
                landmarks=[mesh()],
                blendshapes=[[Category("jawOpen", 0.42)]],
                matrices=[as4x4(rot_y(20))],
            )
        )
        analysis = engine.analyze(self.image())
        self.assertEqual(analysis.landmark_count, 478)
        self.assertEqual(analysis.head_pose.source, "transformation-matrix")
        self.assertAlmostEqual(analysis.head_pose.yaw, 20.0, places=3)
        self.assertAlmostEqual(analysis.blendshapes["jawOpen"], 0.42, places=5)

    def test_pose_falls_back_to_geometry_without_a_matrix(self):
        engine = self.engine_with(Result(landmarks=[mesh()], matrices=[]))
        analysis = engine.analyze(self.image())
        self.assertEqual(analysis.head_pose.source, "landmark-geometry")

    def test_bounding_box_stays_inside_the_image(self):
        engine = self.engine_with(Result(landmarks=[mesh(spread=2.0)]))
        analysis = engine.analyze(self.image())
        box = analysis.bounding_box
        self.assertGreaterEqual(box.x, 0)
        self.assertGreaterEqual(box.y, 0)
        self.assertLessEqual(box.x + box.width, 200)
        self.assertLessEqual(box.y + box.height, 100)

    def test_crop_returns_an_array_and_the_box_it_came_from(self):
        engine = self.engine_with(Result(landmarks=[mesh()]))
        crop, box = engine.crop_face(self.image(), margin=0.2)
        self.assertEqual(crop.shape[0], box.height)
        self.assertEqual(crop.shape[1], box.width)
        self.assertGreater(crop.size, 0)


class MissingModelTests(unittest.TestCase):
    def test_absent_bundle_reports_how_to_fetch_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = FaceMeshEngine(model_path=Path(tmp) / "absent.task")
            self.assertFalse(engine.available)
            with self.assertRaises(FaceEngineUnavailable) as ctx:
                engine.analyze(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertIn("fetch_vision_models", str(ctx.exception))

    def test_load_failure_is_cached_and_re_raised(self):
        """Mirrors the voice router's cached failure, without its silence."""
        with tempfile.TemporaryDirectory() as tmp:
            engine = FaceMeshEngine(model_path=Path(tmp) / "absent.task")
            for _ in range(2):
                with self.assertRaises(FaceEngineUnavailable):
                    engine._load()

    def test_segmenter_absent_bundle_reports_how_to_fetch_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            seg = BackgroundSegmenter(model_path=Path(tmp) / "absent.tflite")
            self.assertFalse(seg.available)
            with self.assertRaises(FaceEngineUnavailable) as ctx:
                seg.foreground_mask(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertIn("fetch_vision_models", str(ctx.exception))

    def test_close_is_safe_when_nothing_loaded(self):
        FaceMeshEngine().close()
        BackgroundSegmenter().close()


if __name__ == "__main__":
    unittest.main()
