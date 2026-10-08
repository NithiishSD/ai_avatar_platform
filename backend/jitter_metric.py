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

Terms, explained once:

* A **landmark** is one numbered point a face detector places on a face (the
  tip of the nose, the outer corner of an eye). MediaPipe's face mesh places
  478 of them, in coordinates normalised to 0..1 of the image size.
* **Inter-ocular distance** is the pixel distance between the outer eye
  corners, used as a ruler for face size.
* **p95** is the 95th percentile: 95 % of frame steps moved less than this,
  which shows bursts of shake that the mean would average away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# The N-09 threshold, in percent of inter-ocular distance per frame step.
TARGET_PCT = 2.0

# MediaPipe face-mesh indices of points the render never moves.
# The indices are positions in MediaPipe's fixed mesh topology, so index 33 is
# always the same spot on every face.
LM_LEFT_EYE_OUTER = 33
LM_RIGHT_EYE_OUTER = 263
ANCHORS = (
    10, 151, 9, 8,            # forehead midline
    168, 6, 197, 195, 5,      # nose bridge
    21, 54, 103, 67, 109,     # upper face oval, image-left
    338, 297, 332, 284, 251,  # upper face oval, image-right
    234, 454,                 # ears' edge
)

# Formatted with the anchor count and stored on every result (golden rule 2).
METHOD = (
    "mean frame-to-frame displacement of {n} static anchor landmarks (forehead, nose bridge, "
    "temples, ears; MediaPipe Face Landmarker per frame) divided by inter-ocular distance. "
    "Upper bound on renderer jitter: video compression and detector noise are included."
)

# Maps an RGB frame to its landmarks, shape (N, 2+) normalised 0..1, or None if no face.
LandmarkFn = Callable[[np.ndarray], Optional[np.ndarray]]


class JitterError(RuntimeError):
    """The clip cannot be measured; the message says why."""


@dataclass
class JitterScore:
    """
    One clip's jitter figures, in percent of inter-ocular distance per frame step.

    ``frames`` counts every frame read; ``frames_without_face`` how many of them
    could not be measured. ``method`` says how the numbers were obtained.
    """

    frames: int
    mean_pct: float
    p95_pct: float
    max_pct: float
    frames_without_face: int = 0
    method: str = METHOD.format(n=len(ANCHORS))
    warnings: List[str] = field(default_factory=list)

    @property
    def meets_target(self) -> bool:
        """True when the mean step is below ``TARGET_PCT`` (strictly)."""
        return self.mean_pct < TARGET_PCT

    def to_dict(self) -> Dict[str, object]:
        """The camelCase JSON shape used by the API and reports, rounded to 3 places."""
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
        # Keep x, y only (MediaPipe also gives depth) and convert to pixels, so the
        # distance is measured in the same units horizontally and vertically.
        points = np.asarray(landmarks, dtype=np.float64)[:, :2] * scale
        eye_gap = float(np.linalg.norm(points[LM_LEFT_EYE_OUTER] - points[LM_RIGHT_EYE_OUTER]))
        # A zero eye gap means a degenerate detection; dividing by it would blow up.
        if eye_gap <= 0:
            missing += 1
            previous = None
            continue
        anchors = points[list(ANCHORS)]
        if previous is not None:
            # Per-anchor Euclidean distance, averaged over anchors, then divided by the
            # previous frame's eye gap to make it a share of face size.
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
    """
    Landmarks of the first face in an RGB frame, or None when there is none.

    Imported lazily so the pure arithmetic above runs in tests without MediaPipe.
    The shared landmarker is not documented as thread-safe and is used by the API
    and the render worker, hence the lock around the call.
    """
    from face_engine import FACE_ENGINE_LOCK, shared_face_engine

    with FACE_ENGINE_LOCK:
        faces = shared_face_engine().analyze_faces(frame)
    return np.asarray(faces[0].landmarks) if faces else None


def score_video(path, landmark_fn: Optional[LandmarkFn] = None, frames: Optional[Iterable[np.ndarray]] = None) -> JitterScore:
    """Measure the jitter of a rendered MP4 (``frames``/``landmark_fn`` are test seams)."""
    import video_io

    # Real use reads the video from disk; tests pass in-memory ``frames`` instead.
    if frames is None:
        info = video_io.probe(path)
        width, height = info.width, info.height
        frames = video_io.read_frames(path)
    else:
        frames = list(frames)
        height, width = frames[0].shape[:2]
    detect = landmark_fn or _face_landmarks
    return score_landmark_track([detect(f) for f in frames], width, height)
