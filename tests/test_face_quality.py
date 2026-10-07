"""Avatar quality gate (G1-06), multi-face analysis and the multiclass segmenter."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2  # noqa: F401 - must be imported before sys.modules is patched below
import numpy as np

import face_engine
from face_engine import (
    MAX_ABS_YAW_DEG,
    MIN_FACE_HEIGHT_PX,
    MULTICLASS_CATEGORIES,
    BoundingBox,
    FaceEngineUnavailable,
    FaceMeshEngine,
    MulticlassSegmenter,
    assess_face_quality,
)
from vision_fixtures import synthetic_analysis


def face(yaw=0.0, pitch=0.0, roll=0.0, height=266, **blendshapes):
    analysis = synthetic_analysis(yaw=yaw)
    analysis.head_pose = face_engine.HeadPose(yaw, pitch, roll, "test")
    box = analysis.bounding_box
    analysis.bounding_box = BoundingBox(box.x, box.y, box.width, height, 512, 512)
    analysis.blendshapes = {"jawOpen": 0.0, "eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0, **blendshapes}
    return analysis


def codes(report, severity):
    issues = report.errors if severity == "error" else report.warnings
    return [issue.code for issue in issues]


class QualityGateTests(unittest.TestCase):
    def test_a_good_photo_passes_clean(self):
        report = assess_face_quality([face()])
        self.assertTrue(report.passed)
        self.assertEqual(report.issues, [])
        self.assertIsNotNone(report.analysis)
        self.assertEqual(report.to_dict(), {"passed": True, "faceCount": 1, "errors": [], "warnings": []})

    def test_no_face(self):
        report = assess_face_quality([])
        self.assertFalse(report.passed)
        self.assertEqual(codes(report, "error"), ["no_face"])
        self.assertIsNone(report.analysis)

    def test_multiple_faces(self):
        report = assess_face_quality([face(), face(), face()])
        self.assertEqual(codes(report, "error"), ["multiple_faces"])
        self.assertIn("3 faces", report.errors[0].message)

    def test_yaw_limit_is_thirty_degrees_either_way(self):
        self.assertEqual(MAX_ABS_YAW_DEG, 30.0)
        self.assertTrue(assess_face_quality([face(yaw=29.9)]).passed)
        self.assertTrue(assess_face_quality([face(yaw=-30.0)]).passed)
        for yaw in (30.5, -45.0):
            self.assertEqual(codes(assess_face_quality([face(yaw=yaw)]), "error"), ["yaw_too_large"])

    def test_small_face(self):
        report = assess_face_quality([face(height=MIN_FACE_HEIGHT_PX - 1)])
        self.assertEqual(codes(report, "error"), ["face_too_small"])
        self.assertTrue(assess_face_quality([face(height=MIN_FACE_HEIGHT_PX)]).passed)

    def test_soft_problems_warn_but_do_not_reject(self):
        report = assess_face_quality(
            [face(pitch=30.0, roll=-25.0, jawOpen=0.5, eyeBlinkLeft=0.9)]
        )
        self.assertTrue(report.passed)
        self.assertEqual(
            codes(report, "warning"), ["pitch_large", "roll_large", "mouth_open", "eyes_closed"]
        )

    def test_errors_accumulate(self):
        report = assess_face_quality([face(yaw=50.0, height=40), face()])
        self.assertEqual(codes(report, "error"), ["multiple_faces", "yaw_too_large", "face_too_small"])

    def test_messages_tell_the_uploader_what_to_do(self):
        for faces in ([], [face(), face()], [face(yaw=60)], [face(height=10)]):
            for issue in assess_face_quality(faces).errors:
                self.assertRegex(issue.message, r"Use|crop")


class _Landmark:
    def __init__(self, x, y):
        self.x, self.y, self.z = x, y, 0.0


class MultiFaceAnalysisTests(unittest.TestCase):
    def _engine(self, faces):
        engine = FaceMeshEngine()
        result = mock.Mock(face_landmarks=faces, facial_transformation_matrixes=[], face_blendshapes=[])
        engine._load = mock.Mock(return_value=mock.Mock(detect=mock.Mock(return_value=result)))
        return engine

    def _square(self, x0, y0, size):
        return [_Landmark(x0 + size * (i % 22) / 21.0, y0 + size * (i // 22) / 21.0) for i in range(478)]

    def test_faces_come_back_largest_first(self):
        engine = self._engine([self._square(0.1, 0.1, 0.1), self._square(0.4, 0.3, 0.4)])
        with mock.patch.dict("sys.modules", {"mediapipe": mock.MagicMock()}):
            faces = engine.analyze_faces(np.zeros((200, 200, 3), dtype=np.uint8))
        self.assertEqual(len(faces), 2)
        self.assertGreater(faces[0].bounding_box.width, faces[1].bounding_box.width)
        self.assertEqual(faces[0].bounding_box.x, 80)

    def test_no_face_is_an_empty_list_and_a_failed_gate(self):
        engine = self._engine([])
        with mock.patch.dict("sys.modules", {"mediapipe": mock.MagicMock()}):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            self.assertEqual(engine.analyze_faces(image), [])
            self.assertEqual(codes(engine.check_quality(image), "error"), ["no_face"])


class MulticlassSegmenterTests(unittest.TestCase):
    def test_missing_model_names_the_fetch_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            segmenter = MulticlassSegmenter(Path(tmp) / "missing.tflite")
            self.assertFalse(segmenter.available)
            for _ in range(2):
                with self.assertRaises(FaceEngineUnavailable) as ctx:
                    segmenter.category_masks(np.zeros((8, 8, 3), dtype=np.uint8))
                self.assertIn("fetch_vision_models.py --only multiclass-segmenter", str(ctx.exception))

    def _masks(self, count, shape=(4, 4, 1)):
        return [mock.Mock(numpy_view=mock.Mock(return_value=np.full(shape, i / 10.0, dtype=np.float32))) for i in range(count)]

    def test_one_mask_per_category_at_image_resolution(self):
        segmenter = MulticlassSegmenter()
        result = mock.Mock(confidence_masks=self._masks(len(MULTICLASS_CATEGORIES)))
        segmenter._load = mock.Mock(return_value=mock.Mock(segment=mock.Mock(return_value=result)))
        with mock.patch.dict("sys.modules", {"mediapipe": mock.MagicMock()}):
            masks = segmenter.category_masks(np.zeros((16, 12, 3), dtype=np.uint8))
        self.assertEqual(tuple(masks), MULTICLASS_CATEGORIES)
        for mask in masks.values():
            self.assertEqual(mask.shape, (16, 12))
            self.assertEqual(mask.dtype, np.float32)
        self.assertAlmostEqual(float(masks[MULTICLASS_CATEGORIES[1]].mean()), 0.1, places=5)

    def test_wrong_model_file_is_detected(self):
        segmenter = MulticlassSegmenter()
        result = mock.Mock(confidence_masks=self._masks(2))
        segmenter._load = mock.Mock(return_value=mock.Mock(segment=mock.Mock(return_value=result)))
        with mock.patch.dict("sys.modules", {"mediapipe": mock.MagicMock()}):
            with self.assertRaises(FaceEngineUnavailable) as ctx:
                segmenter.category_masks(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertIn("wrong one", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
