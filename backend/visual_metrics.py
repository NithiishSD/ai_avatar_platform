"""
Visual quality of a rendered avatar video against its source photo (N-12, N-13).

Two numbers, each with its method named:

* **LPIPS** (learned perceptual image patch similarity, AlexNet features, ``lpips`` package): how
  different two pictures *look* to a trained network. 0 is identical; the target is below 0.1. It is
  computed between the source photo resized to the video's size (which is what the renderer starts
  from) and sampled frames of the video. A talking mouth makes frames differ from the still photo, so
  this is an upper bound on "how much the render changed the picture", not a pure quality score.
* **Identity similarity** (SFace, OpenCV ``FaceRecognizerSF``): the cosine similarity of face
  embeddings of the photo and of a frame. OpenCV documents 0.363 as the cut-off for "same person" on
  this model. The target is stated as ">90% facial similarity"; the number is reported as a percentage
  of cosine similarity and is *not* a probability, so both the raw cosine and the same-person
  threshold are returned beside it.

SFace expects a detector's output (a box and five points). There is no YuNet detector in this
project, so the five points are taken from the MediaPipe face mesh the renderer already uses; the
mapping is in ``five_points``.

A comparison with a **different person's** face is always computed as a negative control, so a
similarity number is never read without knowing what "not the same person" scores on this model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from metrics import spread

SFACE_SAME_PERSON_COSINE = 0.363  # OpenCV's published threshold for this model (cosine distance type)
LPIPS_TARGET = 0.1                # N-12
IDENTITY_TARGET = 0.90            # N-13, as a cosine similarity; see the module docstring

# MediaPipe mesh indices for SFace's five points. "Right" and "left" follow YuNet/SFace's convention,
# the subject's own right, which is the left side of the image: eyes by the midpoint of their two
# corners, then the nose tip, then the two mouth corners.
_RIGHT_EYE = (33, 133)
_LEFT_EYE = (362, 263)
_NOSE_TIP = 1
_MOUTH_RIGHT, _MOUTH_LEFT = 61, 291


class VisualMetricError(RuntimeError):
    """A metric could not be computed; the message says why and how to fix it."""


def five_points(landmarks_px: List[Tuple[int, int]]) -> np.ndarray:
    """The five SFace landmarks, shape (5, 2), from the mesh in pixel coordinates."""
    pts = np.asarray(landmarks_px, dtype=np.float32)
    mid = lambda a, b: (pts[a] + pts[b]) / 2.0  # noqa: E731 - a one-line helper reads better inline
    return np.stack([
        mid(*_RIGHT_EYE), mid(*_LEFT_EYE), pts[_NOSE_TIP], pts[_MOUTH_RIGHT], pts[_MOUTH_LEFT],
    ])


def detection_row(landmarks_px: List[Tuple[int, int]]) -> np.ndarray:
    """A YuNet-style detection row ``[x, y, w, h, 5 x (px, py), score]`` as ``alignCrop`` expects."""
    pts = np.asarray(landmarks_px, dtype=np.float32)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return np.concatenate([[x0, y0, x1 - x0, y1 - y0], five_points(landmarks_px).reshape(-1), [1.0]]).astype(np.float32)


class VisualScorer:
    """Loads SFace and LPIPS once; ``score_video`` and ``compare_faces`` reuse them."""

    def __init__(self, sface_path: Optional[Union[str, Path]] = None) -> None:
        from model_registry import SFACE_MODEL

        self._sface_path = Path(sface_path) if sface_path else SFACE_MODEL
        self._recognizer: Any = None
        self._lpips: Any = None

    # -- loading -------------------------------------------------------------------------------
    def _sface(self):
        if self._recognizer is None:
            import cv2

            if not self._sface_path.is_file():
                raise VisualMetricError(
                    f"SFace weights not found at {self._sface_path}. Fetch them with: "
                    "PYTHONPATH=backend backend/.conda/bin/python scripts/fetch_vision_models.py --only sface"
                )
            self._recognizer = cv2.FaceRecognizerSF.create(str(self._sface_path), "")
        return self._recognizer

    def _lpips_net(self):
        if self._lpips is None:
            try:
                import lpips
            except ImportError as err:
                raise VisualMetricError("the 'lpips' package is not installed: pip install lpips") from err
            self._lpips = lpips.LPIPS(net="alex", verbose=False).eval()
        return self._lpips

    # -- the two metrics -----------------------------------------------------------------------
    def embedding(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """SFace embedding of the largest face, or ``None`` if no face is found."""
        import cv2

        from face_engine import FACE_ENGINE_LOCK, shared_face_engine

        with FACE_ENGINE_LOCK:
            faces = shared_face_engine().analyze_faces(rgb)
        if not faces:
            return None
        row = detection_row(faces[0].pixel_landmarks())
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        recognizer = self._sface()
        return recognizer.feature(recognizer.alignCrop(bgr, row))

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        import cv2

        return float(self._sface().match(a, b, cv2.FaceRecognizerSF_FR_COSINE))

    def lpips_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """LPIPS between two same-size RGB uint8 images."""
        import torch

        def to_tensor(image: np.ndarray):
            # LPIPS wants NCHW floats in [-1, 1].
            return torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0

        with torch.no_grad():
            return float(self._lpips_net()(to_tensor(a), to_tensor(b)).item())

    # -- a whole video -------------------------------------------------------------------------
    def score_video(
        self,
        video: Union[str, Path],
        source_rgb: np.ndarray,
        other_person_rgb: Optional[np.ndarray] = None,
        every: int = 5,
        max_frames: int = 60,
    ) -> Dict[str, Any]:
        """Score every ``every``-th frame (at most ``max_frames``) of ``video`` against the source photo."""
        import cv2

        import video_io

        source_embedding = self.embedding(source_rgb)
        if source_embedding is None:
            raise VisualMetricError("no face found in the source photo, so identity cannot be scored")
        lpips_values: List[float] = []
        cosines: List[float] = []
        no_face = 0
        sampled = 0
        reference: Optional[np.ndarray] = None
        for index, frame in enumerate(video_io.read_frames(video)):
            if index % every:
                continue
            if sampled >= max_frames:
                break
            sampled += 1
            if reference is None:
                # What the renderer started from: the photo scaled to the video's size.
                reference = cv2.resize(source_rgb, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
            lpips_values.append(self.lpips_distance(reference, frame))
            embedding = self.embedding(frame)
            if embedding is None:
                no_face += 1
            else:
                cosines.append(self.cosine(source_embedding, embedding))
        if not sampled:
            raise VisualMetricError("the video has no frames")
        report: Dict[str, Any] = {
            "framesScored": sampled,
            "everyNthFrame": every,
            "lpips": {**spread(lpips_values), "target": LPIPS_TARGET, "method": "LPIPS AlexNet (lpips package), photo resized to video size vs frame"},
            "identity": {
                "cosine": spread(cosines),
                "meanPercent": round(100 * float(np.mean(cosines)), 1) if cosines else None,
                "framesWithoutFace": no_face,
                "sameIdentityThreshold": SFACE_SAME_PERSON_COSINE,
                "framesAboveThreshold": sum(1 for c in cosines if c >= SFACE_SAME_PERSON_COSINE),
                "target": IDENTITY_TARGET,
                "method": "SFace (OpenCV FaceRecognizerSF) cosine, face points from the MediaPipe mesh",
            },
        }
        if other_person_rgb is not None:
            other = self.embedding(other_person_rgb)
            report["identity"]["differentPersonCosine"] = None if other is None else round(self.cosine(source_embedding, other), 3)
        report["lpips"]["meetsTarget"] = bool(lpips_values) and float(np.mean(lpips_values)) < LPIPS_TARGET
        report["identity"]["meetsTarget"] = bool(cosines) and float(np.mean(cosines)) >= IDENTITY_TARGET
        return report


_shared: Optional[VisualScorer] = None


def shared_scorer() -> VisualScorer:
    global _shared
    if _shared is None:
        _shared = VisualScorer()
    return _shared

