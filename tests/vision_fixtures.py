"""
Shared fixtures for the vision tests.

The repository carries no face photo (see ``test_face_engine.py`` for why).
What it does carry is ``fixtures/synthetic_face.json``: the landmarks MediaPipe
found on a Stable Diffusion face. That is the geometry of a person who does
not exist, which is enough to rig and render against a flat test image
without shipping, or downloading, anything.
"""

import json
from pathlib import Path

import numpy as np

from face_engine import BoundingBox, FaceAnalysis, FaceQualityReport, HeadPose, assess_face_quality

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synthetic_face.json"


def synthetic_analysis(yaw=None) -> FaceAnalysis:
    raw = json.loads(FIXTURE.read_text())
    width, height = raw["imageWidth"], raw["imageHeight"]
    box, pose = raw["boundingBox"], raw["headPose"]
    return FaceAnalysis(
        landmarks=[tuple(point) for point in raw["landmarks"]],
        bounding_box=BoundingBox(box["x"], box["y"], box["width"], box["height"], width, height),
        head_pose=HeadPose(
            pose["yaw"] if yaw is None else yaw, pose["pitch"], pose["roll"], "transformation-matrix"
        ),
        blendshapes=dict(raw["blendshapes"]),
        image_width=width,
        image_height=height,
    )


def gradient_image(size: int = 512) -> np.ndarray:
    """A smooth colour gradient: no face in it, but warps are visible on it."""
    ramp = np.linspace(40, 215, size, dtype=np.float32)
    image = np.stack(
        [
            np.tile(ramp[None, :], (size, 1)),
            np.tile(ramp[:, None], (1, size)),
            np.full((size, size), 128, dtype=np.float32),
        ],
        axis=-1,
    )
    return image.astype(np.uint8)


class FakeFaceEngine:
    """Stands in for ``FaceMeshEngine``: returns canned faces, loads nothing."""

    def __init__(self, faces=None):
        self.faces = [synthetic_analysis()] if faces is None else list(faces)
        self.calls = 0

    def analyze_faces(self, image):
        self.calls += 1
        return list(self.faces)

    def analyze(self, image):
        from face_engine import NoFaceDetected

        if not self.faces:
            raise NoFaceDetected("No face detected.")
        self.calls += 1
        return self.faces[0]

    def check_quality(self, image) -> FaceQualityReport:
        return assess_face_quality(self.analyze_faces(image))
