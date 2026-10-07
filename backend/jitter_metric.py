"""
Temporal jitter metric (N-09): do the face's landmarks stay put from frame to frame?

The requirement is "temporal jitter < 2% frame-to-frame, measured as landmark
variance across output frames". A talking avatar legitimately moves its mouth,
jaw and eyelids, so measuring every landmark would score speech itself as
jitter. This metric looks only at **anchor points the animation never
drives**: the forehead, temples, nose bridge and ears. In a correct render
they are fixed, so any movement is shake -- from the warp, from video
compression, or from landmark-detector noise, and the three cannot be told
apart from the video alone. The number is therefore an upper bound on the
renderer's own jitter, and the result says so in its method string.

For each pair of consecutive frames: the mean distance the anchors moved,
divided by the inter-ocular distance (outer eye corners), as a percentage.
Dividing by face size makes the figure independent of video resolution.
The target is met when the mean over the clip is below ``TARGET_PCT``.

Golden rule 2: the result carries its method; nothing here is a MOS-style
number that could be quoted without saying how it was obtained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TARGET_PCT = 2.0

# MediaPipe face-mesh indices of points the render never moves.
LM_LEFT_EYE_OUTER = 33
LM_RIGHT_EYE_OUTER = 263
ANCHORS = (
    10, 151, 9, 8,            # forehead midline
    168, 6, 197, 195, 5,      # nose bridge
    21, 54, 103, 67, 109,     # upper face oval, image-left
    338, 297, 332, 284, 251,  # upper face oval, image-right
    234, 454,                 # ears' edge
)

METHOD = (
    "mean frame-to-frame displacement of {n} static anchor landmarks (forehead, nose bridge, "
    "temples, ears; MediaPipe Face Landmarker per frame) divided by inter-ocular distance. "
    "Upper bound on renderer jitter: video compression and detector noise are included."
)

# Maps an RGB frame to (landmarks (N,2+) normalised 0..1, width, height), or None if no face.
LandmarkFn = Callable[[np.ndarray], Optional[np.ndarray]]


class JitterError(RuntimeError):
    """The clip cannot be measured; the message says why."""


@dataclass
class JitterScore:
    frames: int
    mean_pct: float
    p95_pct: float
    max_pct: float
    frames_without_face: int = 0
    method: str = METHOD.format(n=len(ANCHORS))
    warnings: List[str] = field(default_factory=list)

    @property
    def meets_target(self) -> bool:
        return self.mean_pct < TARGET_PCT

    def to_dict(self) -> Dict[str, object]:
        return {
            "frames": self.frames,
            "meanPct": round(self.mean_pct, 3),
            "p95Pct": round(self.p95_pct, 3),
            "maxPct": round(self.max_pct, 3),
            "targetPct": TARGET_PCT,
            "meetsTarget": self.meets_target,
            "framesWithoutFace": self.frames_without_face,
            "method": self.method,
            "warnings": self.warnings,
        }


def score_landmark_track(tracks: Sequence[Optional[np.ndarray]], width: int, height: int) -> JitterScore:
    """
    Jitter from per-frame landmark arrays (``None`` = no face in that frame).

    Pure arithmetic with no model, so tests can feed it synthetic tracks.
    A frame without a face breaks the chain: its neighbours are not compared
    with each other, because the gap hides how far the face really moved.
    """
    scale = np.array([width, height], dtype=np.float64)
    steps: List[float] = []
    # (anchor points, inter-ocular distance in px) of the previous usable frame
    previous: Optional[Tuple[np.ndarray, float]] = None
    missing = 0
    for landmarks in tracks:
        if landmarks is None:
            missing += 1
            previous = None
            continue
        points = np.asarray(landmarks, dtype=np.float64)[:, :2] * scale
        eye_gap = float(np.linalg.norm(points[LM_LEFT_EYE_OUTER] - points[LM_RIGHT_EYE_OUTER]))
        if eye_gap <= 0:
            missing += 1
            previous = None
            continue
        anchors = points[list(ANCHORS)]
        if previous is not None:
            moved = np.linalg.norm(anchors - previous[0], axis=1).mean()
            steps.append(100.0 * float(moved) / previous[1])
        previous = (anchors, eye_gap)
    if not steps:
        raise JitterError(
            "fewer than two consecutive frames had a detectable face, so there is nothing to compare"
        )
    values = np.asarray(steps)
    warnings = []
    if missing:
        warnings.append(f"{missing} frame(s) had no detectable face and were skipped")
    return JitterScore(
        frames=len(tracks),
        mean_pct=float(values.mean()),
        p95_pct=float(np.percentile(values, 95)),
        max_pct=float(values.max()),
        frames_without_face=missing,
        warnings=warnings,
    )


def _face_landmarks(frame: np.ndarray) -> Optional[np.ndarray]:
    from face_engine import FACE_ENGINE_LOCK, shared_face_engine

    with FACE_ENGINE_LOCK:
        faces = shared_face_engine().analyze_faces(frame)
    return np.asarray(faces[0].landmarks) if faces else None


def score_video(path, landmark_fn: Optional[LandmarkFn] = None, frames: Optional[Iterable[np.ndarray]] = None) -> JitterScore:
    """Measure the jitter of a rendered MP4 (``frames``/``landmark_fn`` are test seams)."""
    import video_io

    if frames is None:
        info = video_io.probe(path)
        width, height = info.width, info.height
        frames = video_io.read_frames(path)
    else:
        frames = list(frames)
        height, width = frames[0].shape[:2]
    detect = landmark_fn or _face_landmarks
    return score_landmark_track([detect(f) for f in frames], width, height)
