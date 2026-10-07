"""
Blendshape-driven portrait animator (task G2-02) -- the CPU lip-sync engine.

One photo in, one frame out per set of ARKit blendshape weights. No neural
network, no GPU: this is the render path that is guaranteed to exist on a
6 GB laptop card that may already be holding a TTS model, and the one the
demo falls back on (visibly, by name) if a neural lip-sync model is missing.

How it works
------------
MediaPipe gives 468 landmarks on the photo. Those landmarks, plus a fixed
ring of anchor points just outside the face, are triangulated once. For every
frame the landmarks are displaced -- lower lip and chin down for an open jaw,
corners out for a stretch, upper lids down for a blink -- and the photo is
re-sampled through the resulting piecewise-affine map. The ring never moves,
so the warp fades to nothing at the edge of the face and the rest of the
picture is untouched.

Two details carry most of the visual quality:

* **The mouth is a slit, not a membrane.** Points above and below the lip
  line are displaced by *different* fields, so opening the jaw separates the
  lips instead of stretching the skin between them. The fields meet again at
  the mouth corners, which keeps the mesh continuous.
* **A closed mouth has no inside.** Parting the lips of a closed-mouth photo
  reveals pixels that were never photographed. The mesh is cut along the lip
  line and the gap is filled with a procedural interior -- a dark cavity, an
  upper row of teeth, a tongue when wide open -- shaded from the photo's own
  lip and skin colours. When the source photo already shows an open mouth,
  its real interior is warped instead.
* **Eyelids are painted, not stretched.** Dragging the lash line down over
  the eye smears lashes and eye shadow into a bruise. A blink instead paints
  a lid in the photo's own skin tone from the original lash line down to
  wherever the lid has reached, with a lash line along its edge.

Displacements are expressed in a face-local frame (mouth-corner axis, mouth
width, mouth-to-chin height), so the same weights produce the same expression
on a tilted head or a differently sized face.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from face_engine import CLASSIC_MESH_POINTS, FaceAnalysis

logger = logging.getLogger(__name__)

# --- MediaPipe Face Mesh index sets ----------------------------------------
# Lip chains run from the image-left corner to the image-right corner, so the
# same position in two chains is (roughly) the same place along the mouth.
INNER_UPPER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
INNER_LOWER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)
OUTER_UPPER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
OUTER_LOWER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
MOUTH_CORNERS = (61, 291, 78, 308)

FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
)

LM_CHIN = 152
LM_SUBNASALE = 2

# Eyes are named for the subject (ARKit convention): the subject's right eye
# is the one on the left of the image.
EYE_RIGHT = {
    "upper": (246, 161, 160, 159, 158, 157, 173),
    "lower": (7, 163, 144, 145, 153, 154, 155),
    "crease": (247, 30, 29, 27, 28, 56, 190),
}
EYE_LEFT = {
    "upper": (466, 388, 387, 386, 385, 384, 398),
    "lower": (249, 390, 373, 374, 380, 381, 382),
    "crease": (467, 260, 259, 257, 258, 286, 414),
}
# Brow chains run from the outer end to the inner end.
BROW_RIGHT = ((70, 63, 105, 66, 107), (46, 53, 52, 65, 55))
BROW_LEFT = ((300, 293, 334, 296, 336), (276, 283, 282, 295, 285))
_BROW_INNER_PROFILE = np.array([0.15, 0.35, 0.65, 0.9, 1.0])
_BROW_OUTER_PROFILE = np.array([1.0, 0.9, 0.6, 0.3, 0.1])

# --- Deformation gains ------------------------------------------------------
# Fractions of the face-local units named in each comment.
RING_SCALE = 1.4           # anchor ring radius, relative to the face oval
OPEN_GAIN = 0.50           # lip gap at jawOpen=1, in mouth-to-chin heights
LOWER_GAIN = 0.10          # extra lower-lip drop, same unit
UPPER_GAIN = 0.20          # upper-lip raise, in nose-to-mouth heights
WIDE_GAIN = 0.18           # corner spread, in half mouth widths
PUCKER_GAIN = 0.32         # corner pinch, same unit
SMILE_GAIN = 0.18          # corner lift, in mouth-to-chin heights
PRESS_GAIN = 0.35          # lip thinning, fraction of distance to the slit
BROW_GAIN = 0.35           # brow travel at weight 1, in nose-to-mouth heights
LID_CLOSE = 0.96           # share of the eye opening a full blink covers

# A photo counts as closed-mouthed when its lip gap is below this share of
# the mouth width; only then is the procedural interior drawn.
CLOSED_MOUTH_GAP = 0.06

TEETH_RGB = np.array([236.0, 226.0, 208.0])
TONGUE_RGB = np.array([172.0, 84.0, 90.0])


class FaceWarpError(RuntimeError):
    """The photo cannot be rigged (for example, too few landmarks)."""


@dataclass(frozen=True)
class MouthParams:
    """The handful of numbers the mouth deformation is actually driven by."""

    open: float = 0.0    # lip aperture from the jaw
    upper: float = 0.0   # upper lip raise
    lower: float = 0.0   # lower lip drop beyond the jaw
    wide: float = 0.0    # +stretch / -pucker
    smile: float = 0.0   # +smile / -frown
    press: float = 0.0   # lips pressed together


def _pair(weights: Mapping[str, float], name: str) -> float:
    return 0.5 * (
        float(weights.get(f"{name}Left", 0.0)) + float(weights.get(f"{name}Right", 0.0))
    )


def mouth_params(weights: Mapping[str, float]) -> MouthParams:
    """Collapse ARKit mouth blendshapes onto the deformation parameters."""
    jaw = float(weights.get("jawOpen", 0.0))
    close = float(weights.get("mouthClose", 0.0))
    funnel = float(weights.get("mouthFunnel", 0.0))
    pucker = float(weights.get("mouthPucker", 0.0))
    roll_lower = float(weights.get("mouthRollLower", 0.0))
    roll_upper = float(weights.get("mouthRollUpper", 0.0))
    smile = _pair(weights, "mouthSmile")
    return MouthParams(
        open=max(0.0, jaw * (1.0 - min(1.0, close))),
        upper=max(0.0, _pair(weights, "mouthUpperUp") + 0.35 * funnel - 0.4 * roll_upper),
        lower=max(0.0, _pair(weights, "mouthLowerDown") + 0.35 * funnel - 0.6 * roll_lower),
        wide=float(
            np.clip(
                0.9 * _pair(weights, "mouthStretch")
                + 0.5 * smile
                + 0.4 * _pair(weights, "mouthDimple")
                - 1.0 * pucker
                - 0.55 * funnel,
                -1.0,
                1.0,
            )
        ),
        smile=float(np.clip(smile - _pair(weights, "mouthFrown"), -1.0, 1.0)),
        press=float(
            np.clip(_pair(weights, "mouthPress") + 0.25 * (roll_lower + roll_upper), 0.0, 1.0)
        ),
    )


class PortraitAnimator:
    """
    Rig one photo, then render any number of frames from it.

    Construction does the expensive work once (frame, triangulation, basis
    fields); ``render`` is a few milliseconds of numpy and one ``cv2.remap``.
    """

    def __init__(self, image_rgb: np.ndarray, analysis: FaceAnalysis) -> None:
        if analysis.landmark_count < CLASSIC_MESH_POINTS:
            raise FaceWarpError(
                f"need {CLASSIC_MESH_POINTS} face landmarks to rig a photo, "
                f"got {analysis.landmark_count}"
            )
        self.base = np.ascontiguousarray(image_rgb[:, :, :3], dtype=np.uint8)
        self.height, self.width = self.base.shape[:2]
        self._photo_mouth = mouth_params(analysis.blendshapes)

        landmarks = np.array(
            [
                (x * self.width, y * self.height)
                for x, y, _ in analysis.landmarks[:CLASSIC_MESH_POINTS]
            ],
            dtype=np.float64,
        )
        self._build_frame(landmarks)
        self._build_mesh(landmarks)
        self._build_mouth_fields()
        self._build_eye_fields()
        self._build_brow_fields()
        self._sample_colours()

    # ------------------------------------------------------------------
    # Rig construction
    # ------------------------------------------------------------------

    def _build_frame(self, lm: np.ndarray) -> None:
        """The face-local axes and units every displacement is measured in."""
        left, right = lm[61], lm[291]
        axis = right - left
        span = float(np.linalg.norm(axis))
        if span < 4.0:
            raise FaceWarpError("the mouth is too small in this photo to animate")
        self.ex = axis / span
        self.ey = np.array([-self.ex[1], self.ex[0]])
        self.mouth_centre = 0.5 * (lm[13] + lm[14])
        if float(np.dot(lm[LM_CHIN] - self.mouth_centre, self.ey)) < 0:
            self.ey = -self.ey
        self.half_width = span / 2.0
        self.chin_height = max(
            4.0, float(np.dot(lm[LM_CHIN] - self.mouth_centre, self.ey))
        )
        self.nose_height = max(
            4.0, float(np.dot(self.mouth_centre - lm[LM_SUBNASALE], self.ey))
        )

    def _local(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(u, v_px)``: across the mouth in half-widths, down it in pixels."""
        rel = points - self.mouth_centre
        return rel @ self.ex / self.half_width, rel @ self.ey

    def _build_mesh(self, lm: np.ndarray) -> None:
        from scipy.spatial import Delaunay

        oval = lm[list(FACE_OVAL)]
        centre = oval.mean(axis=0)
        ring = centre + (oval - centre) * RING_SCALE
        ring[:, 0] = np.clip(ring[:, 0], 0, self.width - 1)
        ring[:, 1] = np.clip(ring[:, 1], 0, self.height - 1)
        ring = np.unique(np.round(ring, 1), axis=0)

        self.ctrl = np.vstack([lm, ring])
        self.n_landmarks = len(lm)
        self.tris = Delaunay(self.ctrl).simplices.astype(np.int32)

        x0 = int(max(0, np.floor(self.ctrl[:, 0].min())))
        y0 = int(max(0, np.floor(self.ctrl[:, 1].min())))
        x1 = int(min(self.width, np.ceil(self.ctrl[:, 0].max()) + 1))
        y1 = int(min(self.height, np.ceil(self.ctrl[:, 1].max()) + 1))
        self.roi = (x0, y0, x1, y1)
        grid_y, grid_x = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        self._grid_x, self._grid_y = grid_x, grid_y

    def _build_mouth_fields(self) -> None:
        """
        Precompute the displacement field of each mouth parameter at weight 1.

        ``render`` then only has to take a weighted sum. Each field is an
        ``(N, 2)`` array of pixel offsets in the *local* frame (x across the
        mouth, y down it), converted to image space at the end.
        """
        n = len(self.ctrl)
        u, v_px = self._local(self.ctrl)

        upper_chain = self.ctrl[list(INNER_UPPER)]
        lower_chain = self.ctrl[list(INNER_LOWER)]
        chain_u, upper_v = self._local(upper_chain)
        _, lower_v = self._local(lower_chain)
        order = np.argsort(chain_u)
        chain_u = chain_u[order]
        slit = 0.5 * (upper_v + lower_v)[order]
        gap = np.maximum(0.0, (lower_v - upper_v)[order])

        slit_at = np.interp(u, chain_u, slit)
        self._gap_at = np.interp(u, chain_u, gap, left=0.0, right=0.0)
        self.rest_gap = float(np.interp(0.0, chain_u, gap))
        self._signed = v_px - slit_at  # distance below the lip line

        below = self._signed > 0
        for index in INNER_UPPER[1:-1] + OUTER_UPPER[1:-1]:
            below[index] = False
        for index in INNER_LOWER[1:-1] + OUTER_LOWER[1:-1]:
            below[index] = True
        corner = np.zeros(n, dtype=bool)
        corner[list(MOUTH_CORNERS)] = True
        # The anchor ring never moves.
        free = np.ones(n, dtype=bool)
        free[self.n_landmarks :] = False
        self._below = below & ~corner & free
        self._above = ~below & ~corner & free

        inside = np.clip(1.0 - u**2, 0.0, 1.0) ** 0.75  # 1 mid-mouth, 0 at corners
        v_low = v_px / self.chin_height
        v_up = -v_px / self.nose_height
        # Full strength from the lips to the chin, fading below it.
        chin_fall = np.where(v_low <= 1.0, 1.0, np.exp(-(((v_low - 1.0) / 0.35) ** 2)))

        zeros = np.zeros(n)
        open_y = np.where(self._below, OPEN_GAIN * self.chin_height * inside * chin_fall, 0.0)
        lower_y = np.where(
            self._below,
            LOWER_GAIN * self.chin_height * inside * np.exp(-((v_low / 0.35) ** 2)),
            0.0,
        )
        upper_y = np.where(
            self._above,
            -UPPER_GAIN * self.nose_height * inside * np.exp(-((v_up / 0.6) ** 2)),
            0.0,
        )

        # Corner motion applies to both sides of the slit, so it is continuous.
        r_v = v_px / self.half_width
        lateral = np.where(
            np.abs(u) <= 1.0,
            u,
            np.sign(u) * np.exp(-(((np.abs(u) - 1.0) / 0.9) ** 2)),
        )
        band = np.exp(-((r_v / 0.9) ** 2)) * free
        wide_x = self.half_width * lateral * band
        smile_y = (
            -SMILE_GAIN
            * self.chin_height
            * np.abs(lateral) ** 1.5
            * np.exp(-((r_v / 0.7) ** 2))
            * free
        )
        # Corners follow an opening jaw a little, which rounds the aperture.
        corner_y = (
            0.25
            * OPEN_GAIN
            * self.chin_height
            * np.exp(-(((np.abs(u) - 1.0) / 0.45) ** 2))
            * np.exp(-((r_v / 0.5) ** 2))
            * free
        )

        lip_thickness = max(
            2.0,
            float(
                np.abs(
                    self._local(self.ctrl[[0, 17]])[1] - np.interp(0.0, chain_u, slit)
                ).mean()
            ),
        )
        press_y = (
            -PRESS_GAIN
            * self._signed
            * np.exp(-((self._signed / (1.2 * lip_thickness)) ** 2))
            * (np.abs(u) < 1.0)
            * free
        )

        self._f_open = np.stack([zeros, open_y + corner_y], axis=1)
        self._f_lower = np.stack([zeros, lower_y], axis=1)
        self._f_upper = np.stack([zeros, upper_y], axis=1)
        self._f_wide = np.stack([wide_x, zeros], axis=1)
        self._f_smile = np.stack([zeros, smile_y], axis=1)
        self._f_press = np.stack([zeros, press_y], axis=1)
        # Closing an already-open mouth: bring the lower side up by the gap.
        self._f_close = np.stack(
            [zeros, np.where(self._below, -self._gap_at * chin_fall, 0.0)], axis=1
        )
        self._procedural_interior = self.rest_gap < CLOSED_MOUTH_GAP * 2 * self.half_width

        if self._procedural_interior:
            # On a closed mouth the upper and lower inner-lip landmarks sit on
            # top of each other, so triangles joining the two sides are
            # slivers of lip-line pixels. Left in, they stretch into dark
            # streaks across the lips as the mouth opens. Cut them out: the
            # hole they leave is exactly the opening the interior is painted
            # into.
            tri_u = u[self.tris]
            across = (
                self._above[self.tris].any(axis=1)
                & self._below[self.tris].any(axis=1)
                & (np.abs(tri_u) < 1.0).all(axis=1)
            )
            self.tris = self.tris[~across]

    def _build_eye_fields(self) -> None:
        n = len(self.ctrl)
        self._f_squint: Dict[str, np.ndarray] = {}
        self._f_wide_eye: Dict[str, np.ndarray] = {}
        for side, eye in (("Right", EYE_RIGHT), ("Left", EYE_LEFT)):
            upper = self.ctrl[list(eye["upper"])]
            lower = self.ctrl[list(eye["lower"])]
            opening = lower - upper  # image-space vector, lid to lid

            squint = np.zeros((n, 2))
            squint[list(eye["lower"])] = -0.25 * opening
            squint[list(eye["upper"])] = 0.10 * opening

            wide = np.zeros((n, 2))
            wide[list(eye["upper"])] = -0.20 * opening
            wide[list(eye["crease"])] = -0.10 * opening

            self._f_squint[side] = squint
            self._f_wide_eye[side] = wide

    def _build_brow_fields(self) -> None:
        n = len(self.ctrl)
        travel = BROW_GAIN * self.nose_height
        up = -self.ey * travel
        self._f_brow_inner = np.zeros((n, 2))
        self._f_brow_outer: Dict[str, np.ndarray] = {}
        self._f_brow_down: Dict[str, np.ndarray] = {}
        for side, chains in (("Right", BROW_RIGHT), ("Left", BROW_LEFT)):
            outer = np.zeros((n, 2))
            down = np.zeros((n, 2))
            for chain in chains:
                idx = list(chain)
                self._f_brow_inner[idx] = up[None, :] * _BROW_INNER_PROFILE[:, None]
                outer[idx] = up[None, :] * _BROW_OUTER_PROFILE[:, None]
                down[idx] = -up[None, :] * (0.5 + 0.5 * _BROW_INNER_PROFILE[:, None])
            self._f_brow_outer[side] = outer
            self._f_brow_down[side] = down

    def _sample_colours(self) -> None:
        """Lip and skin colour of this photo, for shading the mouth interior."""
        import cv2

        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        lips = np.round(
            self.ctrl[list(OUTER_UPPER) + list(OUTER_LOWER[::-1])]
        ).astype(np.int32)
        # Colours are 1-tuples, not bare ints: OpenCV's type stubs declare a
        # Scalar (a sequence), and (255,) draws exactly the same pixels as 255
        # on a single-channel image - including int32 label maps above 255.
        cv2.fillPoly(mask, [lips], (255,))
        lip_pixels = self.base[mask > 0]
        lip = lip_pixels.mean(axis=0) if len(lip_pixels) else np.array([150.0, 80.0, 80.0])
        self._cavity_rgb = np.clip(lip * np.array([0.30, 0.18, 0.18]), 8, 90)

        face = np.zeros_like(mask)
        cv2.fillPoly(face, [np.round(self.ctrl[list(FACE_OVAL)]).astype(np.int32)], (255,))
        skin_pixels = self.base[face > 0]
        luma = float(skin_pixels.mean()) if len(skin_pixels) else 150.0
        # Teeth lit like the rest of the face: never brighter than the skin
        # allows, never so dark they vanish.
        self._light = float(np.clip(luma / 165.0, 0.5, 1.08))

        # Eyelid colour: the skin between the lid crease and the brow, which
        # is what a closed lid actually looks like. The strip right above the
        # lashes is too often shadow, liner or lashes to sample.
        self._lid_rgb: Dict[str, np.ndarray] = {}
        skin = skin_pixels.mean(axis=0) if len(skin_pixels) else np.array([190.0, 150.0, 130.0])
        for side, eye, brow in (
            ("Right", EYE_RIGHT, BROW_RIGHT),
            ("Left", EYE_LEFT, BROW_LEFT),
        ):
            crease = self.ctrl[list(eye["crease"])][1:-1]
            brow_line = self.ctrl[list(brow[1])].mean(axis=0)
            samples = crease + 0.45 * (brow_line[None, :] - crease)
            xs = np.clip(np.round(samples[:, 0]).astype(int), 0, self.width - 1)
            ys = np.clip(np.round(samples[:, 1]).astype(int), 0, self.height - 1)
            colour = np.median(self.base[ys, xs].astype(np.float64), axis=0)
            # Guard against a sample landing in a dark brow or a fringe.
            if colour.mean() < 0.55 * skin.mean():
                colour = skin * 0.92
            self._lid_rgb[side] = colour

    # ------------------------------------------------------------------
    # Per-frame work
    # ------------------------------------------------------------------

    def displacements(self, weights: Mapping[str, float]) -> np.ndarray:
        """Image-space offset of every control point for these weights."""
        target = mouth_params(weights)
        photo = self._photo_mouth

        open_amount = max(0.0, target.open - photo.open)
        # A smile asked for on an already-smiling photo should not stack.
        if target.smile > 0:
            smile = max(0.0, target.smile - max(0.0, photo.smile))
        else:
            smile = target.smile
        wide_gain = WIDE_GAIN if target.wide >= 0 else PUCKER_GAIN

        # A photo taken mid-word has to be able to close: bring the lips
        # together by however much less open the target is than the photo.
        close = min(1.0, 2.0 * target.press)
        if photo.open > 1e-3 and target.open < photo.open:
            close = max(close, (photo.open - target.open) / photo.open)
        upper = max(0.0, target.upper - photo.upper)
        lower = max(0.0, target.lower - photo.lower)

        # Fields that move the two lips differently. Only these can push the
        # lower lip through the upper one, so only these are clamped; the
        # corner fields below move both sides together and must not be.
        local = (
            open_amount * self._f_open
            + lower * self._f_lower
            + target.press * self._f_press
            + close * self._f_close
        )
        floor = -self._gap_at
        local[self._below, 1] = np.maximum(local[self._below, 1], floor[self._below])
        local = (
            local
            + upper * self._f_upper
            + target.wide * wide_gain * self._f_wide
            + smile * self._f_smile
        )

        image = local[:, :1] * self.ex[None, :] + local[:, 1:] * self.ey[None, :]

        def w(name: str) -> float:
            return float(np.clip(weights.get(name, 0.0), 0.0, 1.0))

        for side in ("Right", "Left"):
            closed = w(f"eyeBlink{side}")
            image = image + (1.0 - closed) * w(f"eyeSquint{side}") * self._f_squint[side]
            image = image + (1.0 - closed) * w(f"eyeWide{side}") * self._f_wide_eye[side]
            image = image + w(f"browOuterUp{side}") * self._f_brow_outer[side]
            image = image + w(f"browDown{side}") * self._f_brow_down[side]
        image = image + w("browInnerUp") * self._f_brow_inner

        image[self.n_landmarks :] = 0.0
        return image

    def render(self, weights: Optional[Mapping[str, float]] = None) -> np.ndarray:
        """Render one RGB frame at the photo's own resolution."""
        import cv2

        out = self.base.copy()
        if not weights:
            return out
        disp = self.displacements(weights)
        moved = np.abs(disp).max(axis=1) > 0.05
        blinks = {
            side: float(np.clip(weights.get(f"eyeBlink{side}", 0.0), 0.0, 1.0))
            for side in ("Right", "Left")
        }
        dst = self.ctrl + disp
        if not moved.any():
            self._paint_eyelids(out, dst, blinks, np.zeros(2))
            return out

        active = np.nonzero(moved[self.tris].any(axis=1))[0]
        tri = self.tris[active]
        d = dst[tri]
        s = self.ctrl[tri]
        area2 = (d[:, 1, 0] - d[:, 0, 0]) * (d[:, 2, 1] - d[:, 0, 1]) - (
            d[:, 2, 0] - d[:, 0, 0]
        ) * (d[:, 1, 1] - d[:, 0, 1])
        keep = np.abs(area2) > 0.5
        d, s = d[keep], s[keep]
        if len(d) == 0:
            self._paint_eyelids(out, dst, blinks, np.zeros(2))
            return out

        x0, y0, x1, y1 = self.roi
        label = np.full((y1 - y0, x1 - x0), -1, dtype=np.int32)
        fixed = np.round((d - np.array([x0, y0])) * 16.0).astype(np.int32)
        for k in range(len(fixed)):
            cv2.fillConvexPoly(label, fixed[k], (int(k),), lineType=cv2.LINE_8, shift=4)

        # One affine per triangle, mapping a destination pixel back to the
        # source photo: [x, y, 1] @ inverse[k] -> (source x, source y).
        a = np.concatenate([d, np.ones((len(d), 3, 1))], axis=2)
        inverse = np.linalg.solve(a, s)

        ys, xs = np.nonzero(label >= 0)
        k = label[ys, xs]
        px = (xs + x0).astype(np.float64)
        py = (ys + y0).astype(np.float64)
        map_x = self._grid_x.copy()
        map_y = self._grid_y.copy()
        map_x[ys, xs] = px * inverse[k, 0, 0] + py * inverse[k, 1, 0] + inverse[k, 2, 0]
        map_y[ys, xs] = px * inverse[k, 0, 1] + py * inverse[k, 1, 1] + inverse[k, 2, 1]

        roi = cv2.remap(
            self.base,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        if self._procedural_interior:
            self._paint_mouth_interior(roi, dst, float(weights.get("tongueOut", 0.0)))
        self._paint_eyelids(roi, dst, blinks, np.array([x0, y0], dtype=np.float64))
        out[y0:y1, x0:x1] = roi
        return out

    def _paint_eyelids(
        self,
        canvas: np.ndarray,
        dst: np.ndarray,
        blinks: Mapping[str, float],
        origin: np.ndarray,
    ) -> None:
        """Paint closing eyelids onto ``canvas`` (whose top-left is ``origin``)."""
        import cv2

        for side, eye in (("Right", EYE_RIGHT), ("Left", EYE_LEFT)):
            amount = blinks.get(side, 0.0)
            if amount <= 0.02:
                continue
            upper = dst[list(eye["upper"])] - origin
            lower = dst[list(eye["lower"])] - origin
            opening = lower - upper
            eye_height = float(np.linalg.norm(opening, axis=1).max())
            if eye_height < 1.0:
                continue
            edge = upper + LID_CLOSE * amount * opening

            pad = int(np.ceil(eye_height)) + 4
            points = np.vstack([upper, edge])
            bx0 = int(max(0, np.floor(points[:, 0].min()) - pad))
            by0 = int(max(0, np.floor(points[:, 1].min()) - pad))
            bx1 = int(min(canvas.shape[1], np.ceil(points[:, 0].max()) + pad + 1))
            by1 = int(min(canvas.shape[0], np.ceil(points[:, 1].max()) + pad + 1))
            if bx1 - bx0 < 2 or by1 - by0 < 2:
                continue
            box = np.array([bx0, by0])
            shape = (by1 - by0, bx1 - bx0)

            # `box=box` binds this iteration's value at definition time.
            # Closures look loop variables up when *called*, so without it a
            # call after the loop moved on would use the wrong eye's box.
            def fixed(pts: np.ndarray, box: np.ndarray = box) -> np.ndarray:
                return np.round((pts - box) * 16.0).astype(np.int32)

            # The lid starts a little above the lash line so its top edge
            # blends into the existing upper lid instead of drawing a seam.
            top = upper - 0.35 * opening
            lid = np.zeros(shape, dtype=np.uint8)
            cv2.fillPoly(
                lid, [fixed(np.vstack([top, edge[::-1]]))], (255,), lineType=cv2.LINE_AA, shift=4
            )
            softness = max(0.7, eye_height / 9.0)
            alpha = cv2.GaussianBlur(lid.astype(np.float32) / 255.0, (0, 0), softness)

            # Lash line along the moving edge.
            lash = np.zeros(shape, dtype=np.uint8)
            cv2.polylines(
                lash,
                [fixed(edge)],
                False,
                (255,),
                thickness=max(1, int(round(eye_height * 0.16))),
                lineType=cv2.LINE_AA,
                shift=4,
            )
            lash_alpha = cv2.GaussianBlur(lash.astype(np.float32) / 255.0, (0, 0), softness * 0.7)

            colour = self._lid_rgb[side]
            layer = np.empty(shape + (3,), dtype=np.float32)
            layer[:] = colour
            layer = layer * (1 - 0.72 * lash_alpha[:, :, None])

            a = np.clip(np.maximum(alpha, lash_alpha * 0.9), 0.0, 1.0)[:, :, None]
            region = canvas[by0:by1, bx0:bx1].astype(np.float32)
            canvas[by0:by1, bx0:bx1] = np.clip(
                region * (1 - a) + layer * a, 0, 255
            ).astype(np.uint8)

    def _paint_mouth_interior(
        self, roi: np.ndarray, dst: np.ndarray, tongue_out: float
    ) -> None:
        """Fill the parted lips with a cavity, teeth and tongue, in place."""
        import cv2

        x0, y0, _, _ = self.roi
        origin = np.array([x0, y0])
        upper = dst[list(INNER_UPPER)] - origin
        lower = dst[list(INNER_LOWER)] - origin
        gaps = np.maximum(0.0, (lower - upper) @ self.ey)
        gap = float(gaps.max())
        if gap < 1.5:
            return

        height, width = roi.shape[:2]
        pad = 3
        bx0 = int(max(0, np.floor(min(upper[:, 0].min(), lower[:, 0].min())) - pad))
        by0 = int(max(0, np.floor(min(upper[:, 1].min(), lower[:, 1].min())) - pad))
        bx1 = int(min(width, np.ceil(max(upper[:, 0].max(), lower[:, 0].max())) + pad + 1))
        by1 = int(min(height, np.ceil(max(upper[:, 1].max(), lower[:, 1].max())) + pad + 1))
        if bx1 - bx0 < 2 or by1 - by0 < 2:
            return
        box_origin = np.array([bx0, by0])
        upper_b = upper - box_origin
        lower_b = lower - box_origin
        shape = (by1 - by0, bx1 - bx0)

        def poly(points: np.ndarray) -> np.ndarray:
            return np.round(points * 16.0).astype(np.int32)

        mouth = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(
            mouth, [poly(np.vstack([upper_b, lower_b[::-1]]))], (255,), lineType=cv2.LINE_AA, shift=4
        )

        layer = np.empty(shape + (3,), dtype=np.float32)
        layer[:] = self._cavity_rgb

        chain_u = np.linspace(-1.0, 1.0, len(upper_b))
        light = self._light

        # Tongue first: it sits behind the teeth.
        if gap > 0.22 * self.half_width or tongue_out > 0.05:
            centre = 0.5 * (upper_b[5] + lower_b[5]) + self.ey * gap * (0.22 - 0.3 * tongue_out)
            tongue = np.zeros(shape, dtype=np.uint8)
            cv2.ellipse(
                tongue,
                (int(round(centre[0])), int(round(centre[1]))),
                (max(1, int(0.5 * self.half_width)), max(1, int(gap * 0.34))),
                float(np.degrees(np.arctan2(self.ex[1], self.ex[0]))),
                0,
                360,
                (255,),
                -1,
                lineType=cv2.LINE_AA,
            )
            alpha = (tongue.astype(np.float32) / 255.0)[:, :, None] * 0.85
            layer = layer * (1 - alpha) + (TONGUE_RGB * light * 0.8) * alpha

        # Upper teeth hang from the upper lip; they show as soon as it parts.
        teeth_depth = np.minimum(gaps * 0.5, 0.2 * self.half_width * (1 - 0.35 * chain_u**2))
        teeth_edge = upper_b + self.ey[None, :] * teeth_depth[:, None]
        teeth = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(
            teeth, [poly(np.vstack([upper_b, teeth_edge[::-1]]))], (255,), lineType=cv2.LINE_AA, shift=4
        )
        alpha = (teeth.astype(np.float32) / 255.0)[:, :, None]
        layer = layer * (1 - alpha) + (TEETH_RGB * light) * alpha

        # Lower teeth only on a wide opening.
        if gap > 0.5 * self.half_width:
            depth = np.minimum(gaps * 0.12, 0.07 * self.half_width * (1 - 0.35 * chain_u**2))
            edge = lower_b - self.ey[None, :] * depth[:, None]
            lower_teeth = np.zeros(shape, dtype=np.uint8)
            cv2.fillPoly(
                lower_teeth,
                [poly(np.vstack([edge, lower_b[::-1]]))],
                (255,),
                lineType=cv2.LINE_AA,
                shift=4,
            )
            alpha = (lower_teeth.astype(np.float32) / 255.0)[:, :, None]
            layer = layer * (1 - alpha) + (TEETH_RGB * light * 0.82) * alpha

        # Darken toward the corners, where the lips shade the mouth.
        grid_y, grid_x = np.mgrid[0 : shape[0], 0 : shape[1]]
        rel_x = grid_x + bx0 + x0 - self.mouth_centre[0]
        rel_y = grid_y + by0 + y0 - self.mouth_centre[1]
        across = np.abs(rel_x * self.ex[0] + rel_y * self.ex[1]) / self.half_width
        layer *= (1.0 - 0.45 * np.clip(across, 0.0, 1.0) ** 2)[:, :, None]

        # The lips overhang the opening and shade whatever is just inside it.
        depth_in = cv2.distanceTransform((mouth > 127).astype(np.uint8), cv2.DIST_L2, 3)
        rim = np.clip(depth_in / max(1.5, 0.22 * gap), 0.0, 1.0)
        layer *= (0.5 + 0.5 * rim)[:, :, None]

        softness = max(0.6, self.half_width / 45.0)
        layer = cv2.GaussianBlur(layer, (0, 0), softness)
        alpha = cv2.GaussianBlur(mouth.astype(np.float32) / 255.0, (0, 0), softness * 0.8)
        alpha = alpha[:, :, None]

        region = roi[by0:by1, bx0:bx1].astype(np.float32)
        roi[by0:by1, bx0:bx1] = np.clip(
            region * (1 - alpha) + layer * alpha, 0, 255
        ).astype(np.uint8)
