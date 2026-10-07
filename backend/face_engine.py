"""
Face analysis engine -- Developer 2, Phase 1 (Integration Gate 1).

Produces the vision half of the frozen alignment contract: 468 MediaPipe Face
Mesh landmarks, a face bounding box, head pose, and the 52 ARKit blendshapes,
from a single reference photo.

Two notes on the implementation.

MediaPipe 1.x removed ``mp.solutions`` entirely, so this uses the Tasks API
(``vision.FaceLandmarker``) with a downloaded ``.task`` bundle rather than the
``face_mesh`` solution most tutorials still show.

The landmarker returns 478 points, not 468: the classic mesh plus 10 refined
iris landmarks (5 per eye). Both counts are reported, because the roadmap and
Developer 1's contract talk about 468 and a silent off-by-ten in a landmark
index would be painful to debug later.

Blendshapes matter more than they first appear. ``output_face_blendshapes``
gives ARKit-compatible weights including ``jawOpen``, ``mouthFunnel`` and
``mouthPucker``, which map directly onto Developer 1's 15-viseme vocabulary.
That is the cheapest possible route to Gate 2: a talking avatar that needs no
diffusion model on a 6 GB card.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

from model_registry import (  # noqa: E402  (paths shared with the weight audit)
    FACE_LANDMARKER_TASK,
    MULTICLASS_SEGMENTER_TFLITE,
    SELFIE_SEGMENTER_TFLITE,
)

# The classic Face Mesh topology. Anything beyond this index is an iris point
# added by the refined model.
CLASSIC_MESH_POINTS = 468

# Landmark indices used for head pose and framing. These are stable across
# MediaPipe versions and are the canonical choices from the Face Mesh topology.
LM_NOSE_TIP = 1
LM_CHIN = 152
LM_LEFT_EYE_OUTER = 33
LM_RIGHT_EYE_OUTER = 263
LM_LEFT_MOUTH = 61
LM_RIGHT_MOUTH = 291
LM_FOREHEAD = 10


# Detect more than one face even though only one is ever animated: the
# quality gate has to be able to *see* a second face in order to reject it.
DEFAULT_NUM_FACES = 4

# Quality gate thresholds (task G1-06). Beyond ~30 degrees of yaw the far
# half of the mouth is foreshortened enough that a 2-D lip-sync looks wrong.
MAX_ABS_YAW_DEG = 30.0
WARN_ABS_PITCH_DEG = 25.0
WARN_ABS_ROLL_DEG = 20.0
# Below this the mouth is a handful of pixels wide and there is nothing to
# animate.
MIN_FACE_HEIGHT_PX = 96
WARN_MOUTH_OPEN = 0.25
WARN_EYES_CLOSED = 0.6


class FaceEngineUnavailable(RuntimeError):
    """Raised when MediaPipe or its model bundle is missing."""


class NoFaceDetected(RuntimeError):
    """Raised when a photo contains no detectable face."""


@dataclass(frozen=True)
class BoundingBox:
    """Face bounds in pixels, with the normalized form for renderers."""

    x: int
    y: int
    width: int
    height: int
    image_width: int
    image_height: int

    @property
    def normalized(self) -> Dict[str, float]:
        return {
            "x": self.x / self.image_width if self.image_width else 0.0,
            "y": self.y / self.image_height if self.image_height else 0.0,
            "width": self.width / self.image_width if self.image_width else 0.0,
            "height": self.height / self.image_height if self.image_height else 0.0,
        }

    def expanded(self, margin: float) -> "BoundingBox":
        """
        Grow the box by ``margin`` of its size, clamped to the image.

        Lip-sync models want context around the mouth, not a tight crop, so
        the cropper defaults to a margin rather than these exact bounds.
        """
        dx = int(self.width * margin)
        dy = int(self.height * margin)
        x = max(0, self.x - dx)
        y = max(0, self.y - dy)
        return BoundingBox(
            x=x,
            y=y,
            width=min(self.image_width - x, self.width + 2 * dx),
            height=min(self.image_height - y, self.height + 2 * dy),
            image_width=self.image_width,
            image_height=self.image_height,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "normalized": self.normalized,
        }


@dataclass(frozen=True)
class HeadPose:
    """
    Head rotation in degrees, signed as a viewer of the image sees it.

    Measured on real MediaPipe output (T1.3, 7 Oct 2026) with known-answer
    edits of a photo - in-plane rotation, keystone warps that tilt the face
    plane, and mirroring:

    * ``yaw``   > 0 when the face turns toward the image's right edge
                  (the subject's own left). Mirroring the image negates it.
    * ``pitch`` > 0 when the face looks down. Mirroring leaves it unchanged.
    * ``roll``  > 0 when the head tilts counter-clockwise on screen. Rotating
                  the photo by +10 degrees CCW moved roll by +10.0.

    Both estimators below (the transformation matrix and the landmark
    geometry fallback) follow this convention; a test holds them to it.
    """

    yaw: float
    pitch: float
    roll: float
    source: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
            "source": self.source,
        }


@dataclass
class FaceAnalysis:
    """Everything one photo yields for the render pipeline."""

    landmarks: List[Tuple[float, float, float]]
    bounding_box: BoundingBox
    head_pose: HeadPose
    blendshapes: Dict[str, float] = field(default_factory=dict)
    image_width: int = 0
    image_height: int = 0

    @property
    def landmark_count(self) -> int:
        return len(self.landmarks)

    @property
    def has_iris_landmarks(self) -> bool:
        return self.landmark_count > CLASSIC_MESH_POINTS

    def pixel_landmarks(self) -> List[Tuple[int, int]]:
        """Landmarks in image pixel space, for drawing and cropping."""
        return [
            (int(x * self.image_width), int(y * self.image_height))
            for x, y, _ in self.landmarks
        ]

    def to_dict(self, include_landmarks: bool = True) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "landmarkCount": self.landmark_count,
            "classicMeshPoints": min(self.landmark_count, CLASSIC_MESH_POINTS),
            "hasIrisLandmarks": self.has_iris_landmarks,
            "boundingBox": self.bounding_box.to_dict(),
            "headPose": self.head_pose.to_dict(),
            "blendshapes": {k: round(v, 4) for k, v in self.blendshapes.items()},
            "imageWidth": self.image_width,
            "imageHeight": self.image_height,
        }
        if include_landmarks:
            payload["landmarks"] = [
                {"x": round(x, 6), "y": round(y, 6), "z": round(z, 6)}
                for x, y, z in self.landmarks
            ]
        return payload


def _running_mode_image():
    """``RunningMode.IMAGE`` across the places MediaPipe has kept it."""
    from mediapipe.tasks.python import vision

    for holder in (vision, getattr(vision, "FaceLandmarkerOptions", None)):
        mode = getattr(holder, "RunningMode", None)
        if mode is not None and hasattr(mode, "IMAGE"):
            return mode.IMAGE
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as vtrm

    return vtrm.VisionTaskRunningMode.IMAGE


def _as_rgb_array(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load ``image`` as a contiguous uint8 RGB array."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        from PIL import Image

        with Image.open(path) as handle:
            return np.ascontiguousarray(handle.convert("RGB"))
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)}")
    array = image
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _euler_from_matrix(matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Yaw, pitch and roll in degrees from a 4x4 facial transformation matrix.

    Standard ZYX decomposition. For ``R = Rz(a) @ Ry(b) @ Rx(c)``:

        a = atan2(R[1,0], R[0,0])   rotation about Z
        b = atan2(-R[2,0], sy)      rotation about Y
        c = atan2(R[2,1], R[2,2])   rotation about X

    MediaPipe's canonical face space has +Y up and +X to the subject's left,
    so for a head the Y term is yaw (turning left/right), the X term is pitch
    (nodding) and the Z term is roll (tilting). Naming them by axis order
    instead of by anatomy is an easy way to ship a swapped label.

    The gimbal-lock branch keeps roll from going wild when a face looks
    sharply up or down and ``sy`` collapses.

    Signs follow ``HeadPose``'s viewer-frame convention, verified against
    known-answer image edits rather than assumed from the axis names.
    """
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    if sy > 1e-6:
        about_z = math.atan2(rotation[1, 0], rotation[0, 0])
        about_y = math.atan2(-rotation[2, 0], sy)
        about_x = math.atan2(rotation[2, 1], rotation[2, 2])
    else:
        about_z = 0.0
        about_y = math.atan2(-rotation[2, 0], sy)
        about_x = math.atan2(-rotation[1, 2], rotation[1, 1])
    yaw, pitch, roll = about_y, about_x, about_z
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _pose_from_landmarks(
    landmarks: Sequence[Tuple[float, float, float]],
    width: int,
    height: int,
) -> HeadPose:
    """
    Geometric head-pose estimate, used when no transformation matrix is given.

    Yaw comes from how far the nose sits between the eye corners, pitch from
    the nose's height between forehead and chin, and roll from the tilt of the
    eye line. Approximate, but it keeps the pose field populated rather than
    absent.

    Yaw and pitch are measured in the *face's* frame - along the eye line and
    perpendicular to it - not along the image axes. Measured in image axes, a
    head that was only tilted 10 degrees read 14 degrees of yaw, because
    tilting moves the nose sideways on screen. For an upright face both frames
    coincide, so frontal results are unchanged.
    """
    def point(index: int) -> Tuple[float, float]:
        x, y, _ = landmarks[index]
        return x * width, y * height

    left_eye = point(LM_LEFT_EYE_OUTER)
    right_eye = point(LM_RIGHT_EYE_OUTER)
    nose = point(LM_NOSE_TIP)
    chin = point(LM_CHIN)
    forehead = point(LM_FOREHEAD)

    # LM_LEFT_EYE_OUTER (33) is the eye on the image's left - MediaPipe names
    # it the subject's right - so this vector points toward the image's right.
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    # Image y grows downward, so a counter-clockwise tilt on screen gives a
    # negative eye_dy. Negating makes CCW positive, matching the matrix; the
    # bare atan2 had the opposite sign (+10 deg CCW read as -10.2).
    roll = -math.degrees(math.atan2(eye_dy, eye_dx))

    eye_len = math.hypot(eye_dx, eye_dy)
    yaw = 0.0
    pitch = 0.0
    if eye_len > 1e-6:
        # Unit vectors of the face frame: `across` along the eye line toward
        # the image's right, `down` perpendicular to it, toward the chin.
        across = (eye_dx / eye_len, eye_dy / eye_len)
        down = (-across[1], across[0])

        def along(vector: Tuple[float, float], axis: Tuple[float, float]) -> float:
            return vector[0] * axis[0] + vector[1] * axis[1]

        eye_mid = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        # -1 when the nose sits at the left eye, +1 at the right; scaled to
        # the ~+-60 deg range the mesh stays reliable over.
        nose_offset = along((nose[0] - eye_mid[0], nose[1] - eye_mid[1]), across)
        yaw = (nose_offset / (eye_len / 2)) * 60.0

        vertical = along((chin[0] - forehead[0], chin[1] - forehead[1]), down)
        if abs(vertical) > 1e-6:
            nose_depth = along((nose[0] - forehead[0], nose[1] - forehead[1]), down)
            # 0.5 is a neutral nose height between forehead and chin.
            pitch = ((nose_depth / vertical) - 0.5) * 120.0

    return HeadPose(
        yaw=max(-90.0, min(90.0, yaw)),
        pitch=max(-90.0, min(90.0, pitch)),
        roll=roll,
        source="landmark-geometry",
    )


@dataclass(frozen=True)
class QualityIssue:
    """One reason a photo is unsuitable, or less than ideal, as an avatar."""

    code: str
    message: str
    severity: str = "error"  # "error" rejects the photo; "warning" does not

    def to_dict(self) -> Dict[str, str]:
        return {"code": self.code, "message": self.message, "severity": self.severity}


@dataclass
class FaceQualityReport:
    """The avatar quality gate's verdict on one photo."""

    face_count: int
    issues: List[QualityIssue] = field(default_factory=list)
    analysis: Optional[FaceAnalysis] = None

    @property
    def errors(self) -> List[QualityIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[QualityIssue]:
        return [i for i in self.issues if i.severity != "error"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "faceCount": self.face_count,
            "errors": [i.to_dict() for i in self.errors],
            "warnings": [i.to_dict() for i in self.warnings],
        }


def assess_face_quality(faces: Sequence[FaceAnalysis]) -> FaceQualityReport:
    """
    Decide whether a photo can become a talking avatar (task G1-06).

    Rejects: no face, more than one face, a face turned more than 30 degrees,
    a face too small to animate. Everything else that hurts the result but
    does not prevent a render (tilt, an open mouth, closed eyes) is a warning,
    so the user is told without being blocked.

    Messages are written for the person who uploaded the photo: each one says
    what is wrong and what to do about it.
    """
    if not faces:
        return FaceQualityReport(
            face_count=0,
            issues=[
                QualityIssue(
                    "no_face",
                    "No face was found. Use a well-lit photo of one person "
                    "looking at the camera, with the whole face visible.",
                )
            ],
        )

    primary = faces[0]
    issues: List[QualityIssue] = []

    if len(faces) > 1:
        issues.append(
            QualityIssue(
                "multiple_faces",
                f"{len(faces)} faces were found. An avatar needs exactly one: "
                "crop the photo to a single person.",
            )
        )

    yaw = primary.head_pose.yaw
    if abs(yaw) > MAX_ABS_YAW_DEG:
        issues.append(
            QualityIssue(
                "yaw_too_large",
                f"The head is turned {abs(yaw):.0f} degrees to the side; the limit "
                f"is {MAX_ABS_YAW_DEG:.0f}. Use a photo facing the camera.",
            )
        )

    if primary.bounding_box.height < MIN_FACE_HEIGHT_PX:
        issues.append(
            QualityIssue(
                "face_too_small",
                f"The face is only {primary.bounding_box.height} px tall; at least "
                f"{MIN_FACE_HEIGHT_PX} px is needed. Use a closer or "
                "higher-resolution photo.",
            )
        )

    if abs(primary.head_pose.pitch) > WARN_ABS_PITCH_DEG:
        issues.append(
            QualityIssue(
                "pitch_large",
                f"The head is tilted {abs(primary.head_pose.pitch):.0f} degrees "
                "up or down; lip sync looks best on a level face.",
                severity="warning",
            )
        )
    if abs(primary.head_pose.roll) > WARN_ABS_ROLL_DEG:
        issues.append(
            QualityIssue(
                "roll_large",
                f"The head leans {abs(primary.head_pose.roll):.0f} degrees; "
                "an upright face animates more naturally.",
                severity="warning",
            )
        )
    if primary.blendshapes.get("jawOpen", 0.0) > WARN_MOUTH_OPEN:
        issues.append(
            QualityIssue(
                "mouth_open",
                "The mouth is open in this photo. A relaxed, closed mouth gives "
                "the cleanest lip sync.",
                severity="warning",
            )
        )
    blink = max(
        primary.blendshapes.get("eyeBlinkLeft", 0.0),
        primary.blendshapes.get("eyeBlinkRight", 0.0),
    )
    if blink > WARN_EYES_CLOSED:
        issues.append(
            QualityIssue(
                "eyes_closed",
                "The eyes look closed or nearly closed; open eyes make a more "
                "convincing avatar.",
                severity="warning",
            )
        )

    return FaceQualityReport(face_count=len(faces), issues=issues, analysis=primary)


class FaceMeshEngine:
    """
    MediaPipe Face Mesh wrapper with lazy loading and cached failure.

    Mirrors ``VoiceEngineRouter``'s lifecycle so the vision side behaves the
    same way under a missing model, but unlike the voice router this one
    raises rather than silently substituting: there is no sensible fallback
    for "no landmarks", and a silent fallback is what hid three broken models
    on the audio side for three phases.
    """

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        num_faces: int = DEFAULT_NUM_FACES,
        min_detection_confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path or FACE_LANDMARKER_TASK)
        self.num_faces = num_faces
        self.min_detection_confidence = min_detection_confidence
        self._landmarker = None
        self._load_error: Optional[str] = None

    @property
    def available(self) -> bool:
        """True when the model bundle is on disk. Does not load it."""
        return self.model_path.is_file()

    def _load(self):
        if self._landmarker is not None:
            return self._landmarker
        if self._load_error is not None:
            raise FaceEngineUnavailable(self._load_error)

        if not self.model_path.is_file():
            self._load_error = (
                f"MediaPipe face landmarker bundle not found at {self.model_path}. "
                "Fetch it with: python scripts/fetch_vision_models.py"
            )
            raise FaceEngineUnavailable(self._load_error)

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(self.model_path)
                ),
                running_mode=_running_mode_image(),
                num_faces=self.num_faces,
                min_face_detection_confidence=self.min_detection_confidence,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            logger.info("Loaded MediaPipe FaceLandmarker from %s", self.model_path)
            return self._landmarker
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Could not initialise MediaPipe FaceLandmarker: {exc}"
            raise FaceEngineUnavailable(self._load_error) from exc

    def analyze_faces(
        self, image: Union[str, Path, np.ndarray]
    ) -> List[FaceAnalysis]:
        """
        Every face the landmarker finds, largest first.

        An empty list means no face: this method never raises for that, so
        the quality gate can report "no face" and "two faces" through the
        same path. ``analyze`` is the strict single-face entry point.
        """
        array = _as_rgb_array(image)
        height, width = array.shape[:2]
        landmarker = self._load()

        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=array)
        result = landmarker.detect(mp_image)

        faces = getattr(result, "face_landmarks", None) or []
        matrices = getattr(result, "facial_transformation_matrixes", None) or []
        shape_lists = getattr(result, "face_blendshapes", None) or []

        analyses: List[FaceAnalysis] = []
        for index, face in enumerate(faces):
            points: List[Tuple[float, float, float]] = [
                (lm.x, lm.y, getattr(lm, "z", 0.0)) for lm in face
            ]

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x0 = max(0, int(min(xs) * width))
            y0 = max(0, int(min(ys) * height))
            bbox = BoundingBox(
                x=x0,
                y=y0,
                width=min(width - x0, int((max(xs) - min(xs)) * width)),
                height=min(height - y0, int((max(ys) - min(ys)) * height)),
                image_width=width,
                image_height=height,
            )

            if index < len(matrices):
                yaw, pitch, roll = _euler_from_matrix(matrices[index])
                pose = HeadPose(yaw, pitch, roll, source="transformation-matrix")
            else:
                pose = _pose_from_landmarks(points, width, height)

            blendshapes: Dict[str, float] = {}
            if index < len(shape_lists):
                for category in shape_lists[index]:
                    name = getattr(category, "category_name", None) or str(
                        getattr(category, "index", "")
                    )
                    blendshapes[name] = float(category.score)

            analyses.append(
                FaceAnalysis(
                    landmarks=points,
                    bounding_box=bbox,
                    head_pose=pose,
                    blendshapes=blendshapes,
                    image_width=width,
                    image_height=height,
                )
            )

        analyses.sort(
            key=lambda a: a.bounding_box.width * a.bounding_box.height, reverse=True
        )
        return analyses

    def analyze(
        self, image: Union[str, Path, np.ndarray]
    ) -> FaceAnalysis:
        """
        Detect the primary (largest) face and return its mesh, bounds, pose
        and shapes.

        Raises ``NoFaceDetected`` rather than returning ``None`` so a caller
        cannot mistake an empty result for a neutral one.
        """
        faces = self.analyze_faces(image)
        if not faces:
            raise NoFaceDetected(
                "No face detected. The photo needs a single, reasonably "
                "front-facing, unobstructed face."
            )
        return faces[0]

    def check_quality(
        self, image: Union[str, Path, np.ndarray]
    ) -> "FaceQualityReport":
        """Run the avatar quality gate on a photo. See ``assess_face_quality``."""
        return assess_face_quality(self.analyze_faces(image))

    def crop_face(
        self,
        image: Union[str, Path, np.ndarray],
        margin: float = 0.25,
        analysis: Optional[FaceAnalysis] = None,
    ) -> Tuple[np.ndarray, BoundingBox]:
        """
        Crop to the face with ``margin`` of padding, for the lip-sync stage.

        Returns the crop and the box it came from, so a rendered mouth can be
        composited back into the original frame at the right place.
        """
        array = _as_rgb_array(image)
        analysis = analysis or self.analyze(array)
        box = analysis.bounding_box.expanded(margin)
        crop = array[box.y : box.y + box.height, box.x : box.x + box.width]
        return crop, box

    def close(self) -> None:
        """Release the native landmarker. Safe to call repeatedly."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                logger.debug("Landmarker close failed: %s", exc)
            self._landmarker = None


class BackgroundSegmenter:
    """
    Selfie segmentation for background replacement (Phase 1, Dev 2).

    Returns a float confidence mask rather than a hard boolean so callers can
    feather the edge; a binary matte on a 256x256 model output looks cut out.
    """

    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        self.model_path = Path(model_path or SELFIE_SEGMENTER_TFLITE)
        self._segmenter = None
        self._load_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.model_path.is_file()

    def _load(self):
        if self._segmenter is not None:
            return self._segmenter
        if self._load_error is not None:
            raise FaceEngineUnavailable(self._load_error)
        if not self.model_path.is_file():
            self._load_error = (
                f"Selfie segmenter not found at {self.model_path}. "
                "Fetch it with: python scripts/fetch_vision_models.py"
            )
            raise FaceEngineUnavailable(self._load_error)
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(self.model_path)
                ),
                running_mode=_running_mode_image(),
                output_category_mask=False,
                output_confidence_masks=True,
            )
            self._segmenter = vision.ImageSegmenter.create_from_options(options)
            logger.info("Loaded MediaPipe ImageSegmenter from %s", self.model_path)
            return self._segmenter
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Could not initialise MediaPipe ImageSegmenter: {exc}"
            raise FaceEngineUnavailable(self._load_error) from exc

    def foreground_mask(
        self, image: Union[str, Path, np.ndarray]
    ) -> np.ndarray:
        """A float32 mask in [0, 1] at the image's own resolution."""
        array = _as_rgb_array(image)
        segmenter = self._load()

        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=array)
        result = segmenter.segment(mp_image)
        masks = getattr(result, "confidence_masks", None) or []
        if not masks:
            raise FaceEngineUnavailable("segmenter returned no confidence mask")

        # The selfie model emits background first, person second; fall back to
        # the only mask when a single-channel variant is loaded.
        mask = masks[1] if len(masks) > 1 else masks[0]
        data = np.asarray(mask.numpy_view(), dtype=np.float32)
        # The view arrives as (H, W, 1). Squeezing to (H, W) here keeps the
        # contract simple: compositing adds the channel axis itself, and
        # leaving it on would broadcast to 4-D the moment feathering is off.
        if data.ndim == 3 and data.shape[2] == 1:
            data = data[:, :, 0]
        if data.shape[:2] != array.shape[:2]:
            import cv2

            data = cv2.resize(
                data,
                (array.shape[1], array.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return np.clip(data, 0.0, 1.0)

    def replace_background(
        self,
        image: Union[str, Path, np.ndarray],
        background: Union[np.ndarray, Tuple[int, int, int]] = (18, 24, 38),
        feather: int = 3,
    ) -> np.ndarray:
        """Composite the subject over a colour or image background."""
        array = _as_rgb_array(image)
        mask = self.foreground_mask(array)
        if feather > 0:
            import cv2

            k = feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        if isinstance(background, tuple):
            back = np.zeros_like(array)
            back[:, :] = background
        else:
            back = _as_rgb_array(background)
            if back.shape[:2] != array.shape[:2]:
                import cv2

                back = cv2.resize(back, (array.shape[1], array.shape[0]))

        alpha = mask[:, :, None]
        blended = array.astype(np.float32) * alpha + back.astype(np.float32) * (1 - alpha)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def close(self) -> None:
        if self._segmenter is not None:
            try:
                self._segmenter.close()
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                # Never fatal: the handle is dropped below either way.
                logger.debug("Segmenter close failed: %s", exc)
            self._segmenter = None


# Category order of MediaPipe's ``selfie_multiclass_256x256`` model.
MULTICLASS_CATEGORIES = (
    "background",
    "hair",
    "body_skin",
    "face_skin",
    "clothes",
    "others",
)


class MulticlassSegmenter:
    """
    Hair / skin / clothes segmentation for the customization studio.

    The plain selfie segmenter only knows person versus background. Repainting
    hair or clothing needs to know *which part* of the person a pixel is, and
    this model answers that with one confidence mask per category.
    """

    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        self.model_path = Path(model_path or MULTICLASS_SEGMENTER_TFLITE)
        self._segmenter = None
        self._load_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.model_path.is_file()

    def _load(self):
        if self._segmenter is not None:
            return self._segmenter
        if self._load_error is not None:
            raise FaceEngineUnavailable(self._load_error)
        if not self.model_path.is_file():
            self._load_error = (
                f"Multiclass segmenter not found at {self.model_path}. Fetch it "
                "with: python scripts/fetch_vision_models.py --only multiclass-segmenter"
            )
            raise FaceEngineUnavailable(self._load_error)
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(self.model_path)
                ),
                running_mode=_running_mode_image(),
                output_category_mask=False,
                output_confidence_masks=True,
            )
            self._segmenter = vision.ImageSegmenter.create_from_options(options)
            logger.info("Loaded MediaPipe multiclass segmenter from %s", self.model_path)
            return self._segmenter
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Could not initialise the multiclass segmenter: {exc}"
            raise FaceEngineUnavailable(self._load_error) from exc

    def category_masks(
        self, image: Union[str, Path, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """One float32 mask in [0, 1] per category, at the image's resolution."""
        array = _as_rgb_array(image)
        segmenter = self._load()

        import cv2
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=array)
        result = segmenter.segment(mp_image)
        masks = getattr(result, "confidence_masks", None) or []
        if len(masks) < len(MULTICLASS_CATEGORIES):
            raise FaceEngineUnavailable(
                f"multiclass segmenter returned {len(masks)} masks, expected "
                f"{len(MULTICLASS_CATEGORIES)}; the model file may be the wrong one"
            )

        out: Dict[str, np.ndarray] = {}
        # Lengths were checked just above, so strict=True only documents it.
        for name, mask in zip(MULTICLASS_CATEGORIES, masks, strict=True):
            data = np.asarray(mask.numpy_view(), dtype=np.float32)
            if data.ndim == 3 and data.shape[2] == 1:
                data = data[:, :, 0]
            if data.shape[:2] != array.shape[:2]:
                data = cv2.resize(
                    data,
                    (array.shape[1], array.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            out[name] = np.clip(data, 0.0, 1.0)
        return out

    def close(self) -> None:
        if self._segmenter is not None:
            try:
                self._segmenter.close()
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                # Never fatal: the handle is dropped below either way.
                logger.debug("Segmenter close failed: %s", exc)
            self._segmenter = None


def vision_models_present() -> Dict[str, bool]:
    """Which vision bundles are on disk, for the startup weight audit."""
    return {
        "face_landmarker": FACE_LANDMARKER_TASK.is_file(),
        "selfie_segmenter": SELFIE_SEGMENTER_TFLITE.is_file(),
        "multiclass_segmenter": MULTICLASS_SEGMENTER_TFLITE.is_file(),
    }


# ---------------------------------------------------------------------------
# Process-wide engine
# ---------------------------------------------------------------------------
# The landmarker is a native object that is not documented as thread-safe, and
# the API serves requests from a thread pool while the render worker runs on
# its own thread. One shared instance behind one lock keeps a single copy of
# the model in memory and makes concurrent detection impossible.
import threading  # noqa: E402

FACE_ENGINE_LOCK = threading.RLock()
_shared_engine: Optional[FaceMeshEngine] = None


def shared_face_engine() -> FaceMeshEngine:
    """The process-wide ``FaceMeshEngine``. Hold ``FACE_ENGINE_LOCK`` to use it."""
    global _shared_engine
    with FACE_ENGINE_LOCK:
        if _shared_engine is None:
            _shared_engine = FaceMeshEngine()
        return _shared_engine
