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

Where it sits in the pipeline
-----------------------------
``face_engine`` analyses the photo (landmarks + the photo's own blendshapes).
``viseme_blendshapes`` turns phoneme timings into per-frame blendshape
weights. This module turns those weights into pixels. ``render_engine`` uses
it for offline video jobs and ``live_engine`` for the streaming avatar; both
build one ``PortraitAnimator`` per photo and call ``render`` once per frame.

Concepts used throughout, explained once here
---------------------------------------------
**Landmarks.** MediaPipe Face Mesh returns 468 numbered points on the face
(478 with the iris points, which this module ignores). The numbering is fixed,
so index 13 is always the middle of the upper inner lip. They arrive
normalised to 0..1 and are scaled to pixels here.

**Blendshapes.** Named weights in 0..1 from Apple's ARKit vocabulary
(``jawOpen``, ``mouthSmileLeft``, ``eyeBlinkRight``...). Each says how far one
facial action is applied. They are the input of ``render``.

**Displacement field.** For every control point, a 2-D offset in pixels. Each
blendshape-like parameter gets one precomputed field at weight 1; a frame is a
weighted sum of fields. Linear blending is what makes ``render`` cheap.

**Delaunay triangulation.** A way to join a cloud of points into triangles
that avoids long, thin slivers (no point lies inside another triangle's
circumcircle). Done once on the rest pose; the same triangles are reused with
moved corners on every frame.

**Piecewise-affine warp.** Inside each triangle the motion is one *affine*
map (rotation, scale, shear, translation: 6 numbers), fixed exactly by where
the triangle's three corners go. Neighbouring triangles share edges, so the
whole picture moves continuously, like a rubber sheet pinned at the points.

**Backward mapping with ``cv2.remap``.** To fill output pixel (x, y) we ask
"where in the source photo did this pixel come from?" and sample there. Doing
it the other way round (pushing source pixels forward) leaves holes and
overlaps. ``remap`` takes two float maps, source-x and source-y per output
pixel, and interpolates the photo at those positions.

**Alpha compositing.** Painted layers (lids, teeth, cavity) are blended as
``out = under * (1 - a) + layer * a`` with a soft mask ``a`` in 0..1. Blurring
the mask with a Gaussian gives a feathered edge instead of a hard cut-out.

**Gaussian falloff.** ``exp(-(d / s) ** 2)`` is 1 at distance 0 and fades
smoothly to near 0 by about ``2 * s``. It is used everywhere below to make a
displacement strong at its centre and vanish smoothly, with no visible seam.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

# CLASSIC_MESH_POINTS is 468: the face mesh without the 10 iris points.
from face_engine import CLASSIC_MESH_POINTS, FaceAnalysis

logger = logging.getLogger(__name__)

# --- MediaPipe Face Mesh index sets ----------------------------------------
# Each tuple below is a list of landmark numbers. Indexing the (468, 2) array
# with one of them picks those points out in that order.
# Lip chains run from the image-left corner to the image-right corner, so the
# same position in two chains is (roughly) the same place along the mouth.
# INNER_* trace where the lips meet (the slit); OUTER_* trace the visible lip
# border against the skin. Both start at 78/61 and end at 308/291: the corners.
INNER_UPPER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
INNER_LOWER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)
OUTER_UPPER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
OUTER_LOWER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
# Outer and inner corner points. They belong to neither lip, so they stay
# where both lips' fields meet and keep the mesh joined at the corners.
MOUTH_CORNERS = (61, 291, 78, 308)

# The outline of the face, in order around it. Scaled outward, it becomes the
# fixed anchor ring that pins the warp to the rest of the photo.
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
)

# Bottom of the chin, and the point where the nose meets the upper lip
# ("subnasale"). Their distances from the mouth set the vertical units.
LM_CHIN = 152
LM_SUBNASALE = 2

# Eyes are named for the subject (ARKit convention): the subject's right eye
# is the one on the left of the image.
# "upper"/"lower" are the lash lines of each lid, outer to inner corner;
# "crease" is the fold above the upper lid, which lifts when the eye widens.
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
# Each brow has two chains: the upper edge and the lower edge of the hair.
BROW_RIGHT = ((70, 63, 105, 66, 107), (46, 53, 52, 65, 55))
BROW_LEFT = ((300, 293, 334, 296, 336), (276, 283, 282, 295, 285))
# How much of the brow travel each of the five chain points gets, outer to
# inner. browInnerUp lifts mostly the inner end (the "worried" look);
# browOuterUp lifts mostly the outer end (the "surprised" arch).
_BROW_INNER_PROFILE = np.array([0.15, 0.35, 0.65, 0.9, 1.0])
_BROW_OUTER_PROFILE = np.array([1.0, 0.9, 0.6, 0.3, 0.1])

# --- Deformation gains ------------------------------------------------------
# Fractions of the face-local units named in each comment.
# Being ratios rather than pixels, they look the same on a 300 px face and a
# 1500 px face. They are hand-tuned visual constants, not measured values.
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

# Base colours for the painted interior, in RGB 0..255. They are scaled by the
# photo's brightness at render time (``_light``) so they match its lighting.
# Teeth are an off-white: pure white reads as fake against real skin.
TEETH_RGB = np.array([236.0, 226.0, 208.0])
TONGUE_RGB = np.array([172.0, 84.0, 90.0])


class FaceWarpError(RuntimeError):
    """
    The photo cannot be rigged (for example, too few landmarks).

    Raised only from ``PortraitAnimator.__init__``; once a photo is rigged,
    ``render`` does not raise for any set of weights.
    """


# frozen=True makes instances immutable (assigning a field raises), so a
# MouthParams can be shared between frames without anyone changing it.
@dataclass(frozen=True)
class MouthParams:
    """
    The handful of numbers the mouth deformation is actually driven by.

    ARKit has about 25 mouth blendshapes, many of which overlap. The warp only
    needs six independent motions, so ``mouth_params`` folds the 25 into these.
    All are roughly in -1..1 (most are 0..1).
    """

    open: float = 0.0    # lip aperture from the jaw
    upper: float = 0.0   # upper lip raise
    lower: float = 0.0   # lower lip drop beyond the jaw
    wide: float = 0.0    # +stretch / -pucker
    smile: float = 0.0   # +smile / -frown
    press: float = 0.0   # lips pressed together


def _pair(weights: Mapping[str, float], name: str) -> float:
    """
    Mean of the ``<name>Left`` and ``<name>Right`` weights (0 when missing).

    The mouth is warped symmetrically, so a left/right pair of ARKit shapes
    collapses to one number.
    """
    return 0.5 * (
        float(weights.get(f"{name}Left", 0.0)) + float(weights.get(f"{name}Right", 0.0))
    )


def mouth_params(weights: Mapping[str, float]) -> MouthParams:
    """
    Collapse ARKit mouth blendshapes onto the deformation parameters.

    Missing names count as 0, so a sparse dict (only the shapes a viseme uses)
    is fine. The mixing coefficients below are hand-tuned for appearance.
    Returns a ``MouthParams``.
    """
    jaw = float(weights.get("jawOpen", 0.0))
    close = float(weights.get("mouthClose", 0.0))
    funnel = float(weights.get("mouthFunnel", 0.0))
    pucker = float(weights.get("mouthPucker", 0.0))
    roll_lower = float(weights.get("mouthRollLower", 0.0))
    roll_upper = float(weights.get("mouthRollUpper", 0.0))
    smile = _pair(weights, "mouthSmile")
    return MouthParams(
        # In ARKit, mouthClose means "lips together while the jaw is open", so
        # it cancels the aperture the jaw would otherwise produce.
        open=max(0.0, jaw * (1.0 - min(1.0, close))),
        # A funnel (the "oo" shape) pushes both lips away from the slit;
        # rolling a lip inward hides it, which reads as less raise/drop.
        upper=max(0.0, _pair(weights, "mouthUpperUp") + 0.35 * funnel - 0.4 * roll_upper),
        lower=max(0.0, _pair(weights, "mouthLowerDown") + 0.35 * funnel - 0.6 * roll_lower),
        # One signed axis for corner spread: stretch, smile and dimple pull the
        # corners apart; pucker and funnel pull them together. np.clip keeps
        # the sum in -1..1 when several are active at once.
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
        # Smile and frown are opposites on one axis: positive lifts the
        # corners, negative drops them.
        smile=float(np.clip(smile - _pair(weights, "mouthFrown"), -1.0, 1.0)),
        # Rolling the lips in also thins them, so it adds a little press.
        press=float(
            np.clip(_pair(weights, "mouthPress") + 0.25 * (roll_lower + roll_upper), 0.0, 1.0)
        ),
    )


class PortraitAnimator:
    """
    Rig one photo, then render any number of frames from it.

    Construction does the expensive work once (frame, triangulation, basis
    fields); ``render`` is a few milliseconds of numpy and one ``cv2.remap``.

    Usage::

        animator = PortraitAnimator(image_rgb, face_engine_analysis)
        for weights in track:              # one blendshape dict per frame
            frame = animator.render(weights)

    Raises ``FaceWarpError`` when the photo has fewer than 468 landmarks or a
    mouth too small to animate.
    """

    def __init__(self, image_rgb: np.ndarray, analysis: FaceAnalysis) -> None:
        """Rig ``image_rgb`` (H x W x 3 or 4, uint8) using its face analysis."""
        if analysis.landmark_count < CLASSIC_MESH_POINTS:
            raise FaceWarpError(
                f"need {CLASSIC_MESH_POINTS} face landmarks to rig a photo, "
                f"got {analysis.landmark_count}"
            )
        # [:, :, :3] drops an alpha channel if present. ascontiguousarray makes
        # one packed copy in memory, which OpenCV functions require.
        self.base = np.ascontiguousarray(image_rgb[:, :, :3], dtype=np.uint8)
        self.height, self.width = self.base.shape[:2]
        # The expression already in the photo. Targets are applied *relative*
        # to it, so a smiling photo asked to smile does not smile twice.
        self._photo_mouth = mouth_params(analysis.blendshapes)

        # Normalised (x, y, z) -> pixel (x, y). Depth z is not needed for a
        # 2-D warp, and the iris points beyond 468 are dropped.
        landmarks = np.array(
            [
                (x * self.width, y * self.height)
                for x, y, _ in analysis.landmarks[:CLASSIC_MESH_POINTS]
            ],
            dtype=np.float64,
        )
        # Order matters: every builder after these two reads the axes and
        # units set by _build_frame and the control points set by _build_mesh.
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
        """
        The face-local axes and units every displacement is measured in.

        Sets ``ex`` (unit vector corner to corner), ``ey`` (unit vector
        perpendicular to it, pointing toward the chin), ``mouth_centre`` and
        three lengths in pixels: ``half_width``, ``chin_height`` and
        ``nose_height``. Raises ``FaceWarpError`` for a mouth under 4 px.
        """
        # Outer mouth corners.
        left, right = lm[61], lm[291]
        axis = right - left
        span = float(np.linalg.norm(axis))
        # Below a few pixels the mouth has no room to move; the units below
        # would also approach zero and blow up every ratio.
        if span < 4.0:
            raise FaceWarpError("the mouth is too small in this photo to animate")
        self.ex = axis / span
        # Rotating (x, y) by 90 degrees gives (-y, x): a perpendicular axis.
        # Together ex and ey follow the head's roll, so a tilted head still
        # opens its jaw along its own "down".
        self.ey = np.array([-self.ex[1], self.ex[0]])
        # Midpoint of the inner upper (13) and inner lower (14) lip centres.
        self.mouth_centre = 0.5 * (lm[13] + lm[14])
        # The 90-degree turn could point either way; flip ey if it points away
        # from the chin, so "positive y" always means "toward the chin".
        if float(np.dot(lm[LM_CHIN] - self.mouth_centre, self.ey)) < 0:
            self.ey = -self.ey
        self.half_width = span / 2.0
        # Distances are projected onto ey (a dot product), not straight-line,
        # so they measure the vertical extent in the face's own frame. The
        # 4 px floor stops a division by near-zero on odd landmark layouts.
        self.chin_height = max(
            4.0, float(np.dot(lm[LM_CHIN] - self.mouth_centre, self.ey))
        )
        self.nose_height = max(
            4.0, float(np.dot(self.mouth_centre - lm[LM_SUBNASALE], self.ey))
        )

    def _local(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ``(u, v_px)``: across the mouth in half-widths, down it in pixels.

        ``u`` is -1 at one outer corner, 0 at the centre and +1 at the other.
        Projecting onto an axis is a dot product; ``rel @ self.ex`` does it for
        every row of ``points`` at once.
        """
        rel = points - self.mouth_centre
        return rel @ self.ex / self.half_width, rel @ self.ey

    def _build_mesh(self, lm: np.ndarray) -> None:
        """
        Build the control points, the triangles over them, and the pixel grid.

        Control points are the 468 landmarks followed by the anchor ring, so
        indices below ``n_landmarks`` are face points and the rest never move.
        """
        # Imported here, not at module level, so importing this module stays
        # cheap; the same pattern is used for cv2 below.
        from scipy.spatial import Delaunay

        # Push the face outline outward from its centre by RING_SCALE. Pixels
        # between the face and this ring are stretched to absorb the motion;
        # pixels outside it are never touched.
        oval = lm[list(FACE_OVAL)]
        centre = oval.mean(axis=0)
        ring = centre + (oval - centre) * RING_SCALE
        # Keep the ring inside the image: remap can only sample real pixels.
        ring[:, 0] = np.clip(ring[:, 0], 0, self.width - 1)
        ring[:, 1] = np.clip(ring[:, 1], 0, self.height - 1)
        # Clipping can pile several ring points onto the same image-edge
        # position. Duplicates would make zero-area triangles, so drop them.
        ring = np.unique(np.round(ring, 1), axis=0)

        self.ctrl = np.vstack([lm, ring])
        self.n_landmarks = len(lm)
        # .simplices is an (M, 3) array: each row holds the three control-point
        # indices of one triangle. int32 is what the OpenCV calls expect later.
        self.tris = Delaunay(self.ctrl).simplices.astype(np.int32)

        # Region of interest: the bounding box of all control points. Only
        # this rectangle can change, so render works on it and pastes it back.
        x0 = int(max(0, np.floor(self.ctrl[:, 0].min())))
        y0 = int(max(0, np.floor(self.ctrl[:, 1].min())))
        x1 = int(min(self.width, np.ceil(self.ctrl[:, 0].max()) + 1))
        y1 = int(min(self.height, np.ceil(self.ctrl[:, 1].max()) + 1))
        self.roi = (x0, y0, x1, y1)
        # mgrid gives the (y, x) image coordinates of every ROI pixel. As remap
        # maps they mean "sample each pixel from itself": the identity warp.
        # render starts from a copy of these and overwrites only moved pixels.
        # float32 is the map type cv2.remap accepts.
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

        # Model the lip line (the "slit") as a curve v = slit(u), sampled at
        # the inner-lip chain points. The upper and lower chains share their
        # u positions closely, so pairing them by index is good enough.
        upper_chain = self.ctrl[list(INNER_UPPER)]
        lower_chain = self.ctrl[list(INNER_LOWER)]
        chain_u, upper_v = self._local(upper_chain)
        _, lower_v = self._local(lower_chain)
        # np.interp needs its x samples in increasing order.
        order = np.argsort(chain_u)
        chain_u = chain_u[order]
        slit = 0.5 * (upper_v + lower_v)[order]
        # Opening between the lips at each sample; never negative.
        gap = np.maximum(0.0, (lower_v - upper_v)[order])

        # np.interp evaluates those sampled curves at every control point's u.
        # Outside the mouth the slit is held flat at its end values, while the
        # gap is 0 (left=/right=), since there is no opening past the corners.
        slit_at = np.interp(u, chain_u, slit)
        self._gap_at = np.interp(u, chain_u, gap, left=0.0, right=0.0)
        # The photo's own lip gap at the mouth centre, in pixels.
        self.rest_gap = float(np.interp(0.0, chain_u, gap))
        self._signed = v_px - slit_at  # distance below the lip line

        # Split points into "moves with the lower lip/jaw" and "moves with the
        # upper lip". The sign of the distance to the slit decides, except for
        # the lip landmarks themselves: on a closed mouth they sit almost on
        # the slit, and noise could put an upper-lip point on the wrong side.
        # [1:-1] skips each chain's corners, which are handled separately.
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
        # Boolean masks over all control points: corners and ring points are
        # in neither set, so the lip-separating fields leave them alone.
        self._below = below & ~corner & free
        self._above = ~below & ~corner & free

        # Profile across the mouth. The 0.75 power widens the plateau so the
        # middle of the lips opens evenly instead of to a point.
        inside = np.clip(1.0 - u**2, 0.0, 1.0) ** 0.75  # 1 mid-mouth, 0 at corners
        # Vertical position in face units: 1.0 at the chin below, 1.0 at the
        # nose above.
        v_low = v_px / self.chin_height
        v_up = -v_px / self.nose_height
        # Full strength from the lips to the chin, fading below it.
        # The whole jaw moves as one, so no falloff above the chin; below it
        # the fade carries the motion smoothly into the neck.
        chin_fall = np.where(v_low <= 1.0, 1.0, np.exp(-(((v_low - 1.0) / 0.35) ** 2)))

        zeros = np.zeros(n)
        # Each field below is one column of y offsets (x is zero for most).
        # Jaw open: everything below the slit moves down, scaled by the
        # profile across the mouth and the fade under the chin.
        open_y = np.where(self._below, OPEN_GAIN * self.chin_height * inside * chin_fall, 0.0)
        # Lower-lip drop: like open, but concentrated on the lip itself (a
        # narrow Gaussian in v) so the chin stays put.
        lower_y = np.where(
            self._below,
            LOWER_GAIN * self.chin_height * inside * np.exp(-((v_low / 0.35) ** 2)),
            0.0,
        )
        # Upper-lip raise: negative y is up. Fades out toward the nose.
        upper_y = np.where(
            self._above,
            -UPPER_GAIN * self.nose_height * inside * np.exp(-((v_up / 0.6) ** 2)),
            0.0,
        )

        # Corner motion applies to both sides of the slit, so it is continuous.
        r_v = v_px / self.half_width
        # Sideways push: proportional to u inside the mouth (centre fixed,
        # corners move most), then fading out past the corners into the cheek.
        lateral = np.where(
            np.abs(u) <= 1.0,
            u,
            np.sign(u) * np.exp(-(((np.abs(u) - 1.0) / 0.9) ** 2)),
        )
        # Restrict it to a horizontal band around the mouth; "* free" zeroes
        # the ring (multiplying by a bool array acts as a 0/1 mask).
        band = np.exp(-((r_v / 0.9) ** 2)) * free
        wide_x = self.half_width * lateral * band
        # Smile: lift (negative y) that grows toward the corners, so the
        # centre of the mouth stays and the corners curl up.
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

        # Lip thickness: mean distance from the slit to the outer lip border at
        # the centre (landmark 0 on top, 17 below), at least 2 px.
        lip_thickness = max(
            2.0,
            float(
                np.abs(
                    self._local(self.ctrl[[0, 17]])[1] - np.interp(0.0, chain_u, slit)
                ).mean()
            ),
        )
        # Press: move points toward the slit (the minus sign flips their signed
        # distance), strongest within about one lip thickness of it. Both lips
        # thin, which is what pressed lips look like.
        press_y = (
            -PRESS_GAIN
            * self._signed
            * np.exp(-((self._signed / (1.2 * lip_thickness)) ** 2))
            * (np.abs(u) < 1.0)
            * free
        )

        # Pack each (x, y) pair into an (N, 2) field, still in the local frame;
        # displacements() rotates the weighted sum into image space.
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
        # 2 * half_width is the full mouth width that CLOSED_MOUTH_GAP is a
        # share of.
        self._procedural_interior = self.rest_gap < CLOSED_MOUTH_GAP * 2 * self.half_width

        if self._procedural_interior:
            # On a closed mouth the upper and lower inner-lip landmarks sit on
            # top of each other, so triangles joining the two sides are
            # slivers of lip-line pixels. Left in, they stretch into dark
            # streaks across the lips as the mouth opens. Cut them out: the
            # hole they leave is exactly the opening the interior is painted
            # into.
            # u[self.tris] looks up u for each triangle's three corners: shape
            # (M, 3). A triangle "crosses the slit" when it has a corner on
            # each side and lies between the corners (|u| < 1).
            tri_u = u[self.tris]
            across = (
                self._above[self.tris].any(axis=1)
                & self._below[self.tris].any(axis=1)
                & (np.abs(tri_u) < 1.0).all(axis=1)
            )
            self.tris = self.tris[~across]

    def _build_eye_fields(self) -> None:
        """
        Precompute squint and widen fields per eye, at weight 1.

        Blinks are not a field: they are painted in ``_paint_eyelids``, for
        the reason given in the module docstring. These fields are in image
        space already, since they are built from image-space lid vectors.
        """
        n = len(self.ctrl)
        self._f_squint: Dict[str, np.ndarray] = {}
        self._f_wide_eye: Dict[str, np.ndarray] = {}
        for side, eye in (("Right", EYE_RIGHT), ("Left", EYE_LEFT)):
            upper = self.ctrl[list(eye["upper"])]
            lower = self.ctrl[list(eye["lower"])]
            opening = lower - upper  # image-space vector, lid to lid

            # Squint: the lower lid rises a quarter of the opening and the upper
            # lid drops a little. All other points stay at zero offset.
            squint = np.zeros((n, 2))
            squint[list(eye["lower"])] = -0.25 * opening
            squint[list(eye["upper"])] = 0.10 * opening

            # Widen: the upper lid and, less, the crease above it lift.
            wide = np.zeros((n, 2))
            wide[list(eye["upper"])] = -0.20 * opening
            wide[list(eye["crease"])] = -0.10 * opening

            self._f_squint[side] = squint
            self._f_wide_eye[side] = wide

    def _build_brow_fields(self) -> None:
        """
        Precompute brow fields, in image space, at weight 1.

        ARKit has one ``browInnerUp`` for both brows but separate left/right
        ``browOuterUp`` and ``browDown``, hence one array and two dicts.
        """
        n = len(self.ctrl)
        travel = BROW_GAIN * self.nose_height
        # "Up" along the face's own vertical (ey points toward the chin).
        up = -self.ey * travel
        self._f_brow_inner = np.zeros((n, 2))
        self._f_brow_outer: Dict[str, np.ndarray] = {}
        self._f_brow_down: Dict[str, np.ndarray] = {}
        for side, chains in (("Right", BROW_RIGHT), ("Left", BROW_LEFT)):
            outer = np.zeros((n, 2))
            down = np.zeros((n, 2))
            for chain in chains:
                idx = list(chain)
                # Broadcasting: up[None, :] is (1, 2), the profile[:, None] is
                # (5, 1); their product is (5, 2), one offset per chain point.
                self._f_brow_inner[idx] = up[None, :] * _BROW_INNER_PROFILE[:, None]
                outer[idx] = up[None, :] * _BROW_OUTER_PROFILE[:, None]
                # Lowering is strongest at the inner end (a frown pulls the
                # brows together and down) but moves the whole brow at least half.
                down[idx] = -up[None, :] * (0.5 + 0.5 * _BROW_INNER_PROFILE[:, None])
            self._f_brow_outer[side] = outer
            self._f_brow_down[side] = down

    def _sample_colours(self) -> None:
        """Lip and skin colour of this photo, for shading the mouth interior."""
        import cv2

        # A mask is a single-channel image: 255 inside the region, 0 outside.
        # Indexing the photo with ``mask > 0`` then gives just those pixels.
        # The lip polygon is the outer upper border followed by the outer
        # lower border reversed, so the outline goes round without crossing.
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        lips = np.round(
            self.ctrl[list(OUTER_UPPER) + list(OUTER_LOWER[::-1])]
        ).astype(np.int32)
        # Colours are 1-tuples, not bare ints: OpenCV's type stubs declare a
        # Scalar (a sequence), and (255,) draws exactly the same pixels as 255
        # on a single-channel image - including int32 label maps above 255.
        cv2.fillPoly(mask, [lips], (255,))
        lip_pixels = self.base[mask > 0]
        # Fallback colours (here and below) apply only if a polygon covers no
        # pixel, which would otherwise make .mean() of an empty array NaN.
        lip = lip_pixels.mean(axis=0) if len(lip_pixels) else np.array([150.0, 80.0, 80.0])
        # The cavity is a dark, reddish version of the lip colour: red kept
        # more than green and blue. Clipped so it is never pure black (which
        # looks like a hole) nor bright enough to read as skin.
        self._cavity_rgb = np.clip(lip * np.array([0.30, 0.18, 0.18]), 8, 90)

        face = np.zeros_like(mask)
        cv2.fillPoly(face, [np.round(self.ctrl[list(FACE_OVAL)]).astype(np.int32)], (255,))
        skin_pixels = self.base[face > 0]
        # Mean over all three channels: a rough brightness ("luma") of the face.
        luma = float(skin_pixels.mean()) if len(skin_pixels) else 150.0
        # Teeth lit like the rest of the face: never brighter than the skin
        # allows, never so dark they vanish.
        # 165 is the face brightness at which the paint colours are used as-is.
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
            # Skip the crease's end points, which sit near the eye corners.
            crease = self.ctrl[list(eye["crease"])][1:-1]
            # Centre of the brow's lower edge.
            brow_line = self.ctrl[list(brow[1])].mean(axis=0)
            # Sample 45% of the way from the crease toward the brow.
            samples = crease + 0.45 * (brow_line[None, :] - crease)
            xs = np.clip(np.round(samples[:, 0]).astype(int), 0, self.width - 1)
            ys = np.clip(np.round(samples[:, 1]).astype(int), 0, self.height - 1)
            # Median, not mean: one sample on a stray hair cannot drag it.
            colour = np.median(self.base[ys, xs].astype(np.float64), axis=0)
            # Guard against a sample landing in a dark brow or a fringe.
            if colour.mean() < 0.55 * skin.mean():
                colour = skin * 0.92
            self._lid_rgb[side] = colour

    # ------------------------------------------------------------------
    # Per-frame work
    # ------------------------------------------------------------------

    def displacements(self, weights: Mapping[str, float]) -> np.ndarray:
        """
        Image-space offset of every control point for these weights.

        Returns an ``(N, 2)`` array, one ``(dx, dy)`` in pixels per control
        point; the anchor-ring rows are always zero. Public so tests
        (``tests/test_face_warp.py``) can check the motion without rendering.
        """
        target = mouth_params(weights)
        photo = self._photo_mouth

        # Mouth amounts are deltas from the photo's own expression: the photo
        # is the rest pose, so only the extra opening beyond it is applied.
        open_amount = max(0.0, target.open - photo.open)
        # A smile asked for on an already-smiling photo should not stack.
        if target.smile > 0:
            smile = max(0.0, target.smile - max(0.0, photo.smile))
        else:
            smile = target.smile
        # One signed field serves both directions; pucker gets its own,
        # larger gain because pinching reads weaker than stretching.
        wide_gain = WIDE_GAIN if target.wide >= 0 else PUCKER_GAIN

        # A photo taken mid-word has to be able to close: bring the lips
        # together by however much less open the target is than the photo.
        # Pressed lips must also be closed lips, so press alone closes fully
        # by press=0.5.
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
        # A lower-side point may rise by at most the photo's lip gap there:
        # that brings the lips together and no further. Triangles would fold
        # over (flip inside out) if it rose past the upper lip.
        floor = -self._gap_at
        local[self._below, 1] = np.maximum(local[self._below, 1], floor[self._below])
        local = (
            local
            + upper * self._f_upper
            + target.wide * wide_gain * self._f_wide
            + smile * self._f_smile
        )

        # Local (x, y) -> image space: x along ex plus y along ey. Slicing
        # with :1 and 1: keeps a column shape (N, 1) so broadcasting with the
        # (1, 2) axes gives (N, 2).
        image = local[:, :1] * self.ex[None, :] + local[:, 1:] * self.ey[None, :]

        def w(name: str) -> float:
            """One weight, defaulting to 0 and clamped to 0..1."""
            return float(np.clip(weights.get(name, 0.0), 0.0, 1.0))

        for side in ("Right", "Left"):
            closed = w(f"eyeBlink{side}")
            # As the lid closes, squint and widen fade out: they move lid
            # points that the painted lid is about to cover.
            image = image + (1.0 - closed) * w(f"eyeSquint{side}") * self._f_squint[side]
            image = image + (1.0 - closed) * w(f"eyeWide{side}") * self._f_wide_eye[side]
            image = image + w(f"browOuterUp{side}") * self._f_brow_outer[side]
            image = image + w(f"browDown{side}") * self._f_brow_down[side]
        image = image + w("browInnerUp") * self._f_brow_inner

        # Belt and braces: the ring is what keeps the warp local, so force it
        # still even if a field leaked a value onto it.
        image[self.n_landmarks :] = 0.0
        return image

    def render(self, weights: Optional[Mapping[str, float]] = None) -> np.ndarray:
        """
        Render one RGB frame at the photo's own resolution.

        ``weights`` is one frame of ARKit blendshape weights. ``None`` or an
        empty dict returns an unchanged copy of the photo. The result is a new
        H x W x 3 uint8 array; the rigged photo is never modified.

        Steps: compute displacements, find the triangles that move, rasterise
        them into a label map, build per-pixel source coordinates from each
        triangle's affine map, ``cv2.remap`` the photo, then paint the mouth
        interior and eyelids on top.
        """
        import cv2

        out = self.base.copy()
        if not weights:
            return out
        disp = self.displacements(weights)
        # A point "moved" if either coordinate shifted more than 1/20 px;
        # smaller shifts are invisible and not worth warping.
        moved = np.abs(disp).max(axis=1) > 0.05
        blinks = {
            side: float(np.clip(weights.get(f"eyeBlink{side}", 0.0), 0.0, 1.0))
            for side in ("Right", "Left")
        }
        # Destination positions of every control point this frame.
        dst = self.ctrl + disp
        # Nothing moved (for example a blink-only frame): skip the warp but
        # still paint lids. Origin (0, 0) because ``out`` is the full image.
        if not moved.any():
            self._paint_eyelids(out, dst, blinks, np.zeros(2))
            return out

        # Only triangles with at least one moved corner need re-sampling;
        # every other pixel already equals the photo.
        active = np.nonzero(moved[self.tris].any(axis=1))[0]
        tri = self.tris[active]
        # d and s: (M, 3, 2) destination and source corner positions.
        d = dst[tri]
        s = self.ctrl[tri]
        # Twice the signed triangle area (a 2-D cross product). A near-zero
        # area means a collapsed triangle whose affine map has no inverse, and
        # np.linalg.solve below would fail on it, so drop those.
        area2 = (d[:, 1, 0] - d[:, 0, 0]) * (d[:, 2, 1] - d[:, 0, 1]) - (
            d[:, 2, 0] - d[:, 0, 0]
        ) * (d[:, 1, 1] - d[:, 0, 1])
        keep = np.abs(area2) > 0.5
        d, s = d[keep], s[keep]
        if len(d) == 0:
            self._paint_eyelids(out, dst, blinks, np.zeros(2))
            return out

        # Label map: for each ROI pixel, the index of the destination triangle
        # covering it, or -1 for "not moved". Drawing each triangle filled
        # with its own index is a fast way to ask "which triangle is this
        # pixel in?" for every pixel at once.
        x0, y0, x1, y1 = self.roi
        label = np.full((y1 - y0, x1 - x0), -1, dtype=np.int32)
        # Fixed-point coordinates: OpenCV's ``shift=4`` means the integer
        # points carry 4 fractional bits, so multiply by 2**4 = 16. That keeps
        # sub-pixel corner positions; plain integers would make neighbouring
        # triangles jitter by a pixel from frame to frame.
        fixed = np.round((d - np.array([x0, y0])) * 16.0).astype(np.int32)
        # LINE_8 (no anti-aliasing) because the values are labels, not colours:
        # blending two indices on an edge would produce a third, wrong index.
        for k in range(len(fixed)):
            cv2.fillConvexPoly(label, fixed[k], (int(k),), lineType=cv2.LINE_8, shift=4)

        # One affine per triangle, mapping a destination pixel back to the
        # source photo: [x, y, 1] @ inverse[k] -> (source x, source y).
        # For each triangle, a is the 3x3 matrix of destination corners as
        # rows [x, y, 1]. Solving a @ M = s for the 3x2 matrix M gives the
        # affine map that sends each destination corner to its source corner.
        # np.linalg.solve does all M triangles in one batched call.
        a = np.concatenate([d, np.ones((len(d), 3, 1))], axis=2)
        inverse = np.linalg.solve(a, s)

        # Every pixel covered by a moved triangle, and which triangle it is.
        ys, xs = np.nonzero(label >= 0)
        k = label[ys, xs]
        # Back to full-image pixel coordinates, since the maps are absolute.
        px = (xs + x0).astype(np.float64)
        py = (ys + y0).astype(np.float64)
        # Start from the identity maps, then overwrite the moved pixels with
        # [px, py, 1] @ inverse[k], written out per output coordinate.
        map_x = self._grid_x.copy()
        map_y = self._grid_y.copy()
        map_x[ys, xs] = px * inverse[k, 0, 0] + py * inverse[k, 1, 0] + inverse[k, 2, 0]
        map_y[ys, xs] = px * inverse[k, 0, 1] + py * inverse[k, 1, 1] + inverse[k, 2, 1]

        # remap samples the *whole* photo, so a source position outside the
        # ROI still finds real pixels. INTER_LINEAR blends the 4 nearest
        # pixels for fractional positions. BORDER_REFLECT_101 mirrors the
        # image at its edge for samples that fall off it, avoiding black seams.
        # The output has the maps' shape: just the ROI.
        roi = cv2.remap(
            self.base,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        # Paint after warping, so the paint sits on top of the moved lips and
        # lids.
        if self._procedural_interior:
            self._paint_mouth_interior(roi, dst, float(weights.get("tongueOut", 0.0)))
        self._paint_eyelids(roi, dst, blinks, np.array([x0, y0], dtype=np.float64))
        # Paste the warped rectangle back into the untouched full frame.
        out[y0:y1, x0:x1] = roi
        return out

    def _paint_eyelids(
        self,
        canvas: np.ndarray,
        dst: np.ndarray,
        blinks: Mapping[str, float],
        origin: np.ndarray,
    ) -> None:
        """
        Paint closing eyelids onto ``canvas`` (whose top-left is ``origin``).

        ``canvas`` is modified in place. ``origin`` converts image-space
        landmark positions to canvas pixels: the ROI corner when ``canvas`` is
        the warped ROI, (0, 0) when it is the full frame.
        """
        import cv2

        for side, eye in (("Right", EYE_RIGHT), ("Left", EYE_LEFT)):
            amount = blinks.get(side, 0.0)
            # A tiny blink weight is invisible; skip the work.
            if amount <= 0.02:
                continue
            upper = dst[list(eye["upper"])] - origin
            lower = dst[list(eye["lower"])] - origin
            opening = lower - upper
            eye_height = float(np.linalg.norm(opening, axis=1).max())
            # An eye already shut (or a closed-eye photo) has nothing to cover.
            if eye_height < 1.0:
                continue
            # Where the lid's lower edge has got to. LID_CLOSE stops just short
            # of the lower lash line, so the two lash lines do not merge into
            # one thick band.
            edge = upper + LID_CLOSE * amount * opening

            # Work in a small box around the eye, not the whole canvas: masks,
            # blurs and blends then touch a few hundred pixels instead of the
            # whole frame. The pad leaves room for the blur to spread.
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
                """Box-relative points in OpenCV's 4-bit fixed point (shift=4)."""
                return np.round((pts - box) * 16.0).astype(np.int32)

            # The lid starts a little above the lash line so its top edge
            # blends into the existing upper lid instead of drawing a seam.
            top = upper - 0.35 * opening
            lid = np.zeros(shape, dtype=np.uint8)
            cv2.fillPoly(
                lid, [fixed(np.vstack([top, edge[::-1]]))], (255,), lineType=cv2.LINE_AA, shift=4
            )
            # Blur radius scales with the eye so big and small faces get the
            # same relative softness. Kernel size (0, 0) tells OpenCV to derive
            # the kernel from sigma. The blurred 0..1 mask is the lid's alpha.
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

            # The paint layer: flat lid colour, darkened along the lash line
            # (to 28% brightness at full lash strength).
            colour = self._lid_rgb[side]
            layer = np.empty(shape + (3,), dtype=np.float32)
            layer[:] = colour
            layer = layer * (1 - 0.72 * lash_alpha[:, :, None])

            # Coverage is the lid or the lash, whichever is stronger; the lash
            # spills slightly past the lid edge. [:, :, None] adds a channel
            # axis so one alpha value applies to R, G and B.
            a = np.clip(np.maximum(alpha, lash_alpha * 0.9), 0.0, 1.0)[:, :, None]
            region = canvas[by0:by1, bx0:bx1].astype(np.float32)
            canvas[by0:by1, bx0:bx1] = np.clip(
                region * (1 - a) + layer * a, 0, 255
            ).astype(np.uint8)

    def _paint_mouth_interior(
        self, roi: np.ndarray, dst: np.ndarray, tongue_out: float
    ) -> None:
        """
        Fill the parted lips with a cavity, teeth and tongue, in place.

        Only used for closed-mouth photos (``_procedural_interior``). ``roi`` is
        the warped region from ``render``; ``dst`` are this frame's control
        point positions; ``tongue_out`` is the ARKit ``tongueOut`` weight.
        Layers are built back to front (cavity, tongue, teeth), shaded, then
        blended into ``roi`` through a soft mouth-shaped mask.
        """
        import cv2

        # Positions relative to the ROI, which is the canvas here.
        x0, y0, _, _ = self.roi
        origin = np.array([x0, y0])
        upper = dst[list(INNER_UPPER)] - origin
        lower = dst[list(INNER_LOWER)] - origin
        # Opening at each chain position, measured along the face's vertical.
        gaps = np.maximum(0.0, (lower - upper) @ self.ey)
        gap = float(gaps.max())
        # Under 1.5 px the opening is a line; painting it would only smudge.
        if gap < 1.5:
            return

        # Same small-box trick as the eyelids: a bounding box round the lips.
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
            """Box-relative points in OpenCV's 4-bit fixed point (shift=4)."""
            return np.round(points * 16.0).astype(np.int32)

        # The opening: inner upper lip left to right, then inner lower lip
        # right to left, which closes the outline.
        mouth = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(
            mouth, [poly(np.vstack([upper_b, lower_b[::-1]]))], (255,), lineType=cv2.LINE_AA, shift=4
        )

        # Bottom layer: the dark cavity colour everywhere in the box.
        layer = np.empty(shape + (3,), dtype=np.float32)
        layer[:] = self._cavity_rgb

        # Approximate u (-1..1) along the chain, by index rather than by
        # exact position; it only shapes the teeth's taper toward the corners.
        chain_u = np.linspace(-1.0, 1.0, len(upper_b))
        light = self._light

        # Tongue first: it sits behind the teeth.
        # It shows on a wide enough opening, or whenever tongueOut asks for it.
        if gap > 0.22 * self.half_width or tongue_out > 0.05:
            # Index 5 is the middle of each 11-point inner-lip chain. The
            # tongue rests low in the mouth and rises forward with tongueOut.
            centre = 0.5 * (upper_b[5] + lower_b[5]) + self.ey * gap * (0.22 - 0.3 * tongue_out)
            tongue = np.zeros(shape, dtype=np.uint8)
            cv2.ellipse(
                tongue,
                (int(round(centre[0])), int(round(centre[1]))),
                (max(1, int(0.5 * self.half_width)), max(1, int(gap * 0.34))),
                # Rotate the ellipse with the head: the angle of the mouth axis.
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
        # Their visible height is half the opening, capped at a tooth length
        # that shortens toward the corners where the teeth curve away.
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

        # A slight blur takes the "drawn" look off the layer, and the blurred
        # mask feathers its edge into the real lips.
        softness = max(0.6, self.half_width / 45.0)
        layer = cv2.GaussianBlur(layer, (0, 0), softness)
        alpha = cv2.GaussianBlur(mouth.astype(np.float32) / 255.0, (0, 0), softness * 0.8)
        alpha = alpha[:, :, None]

        region = roi[by0:by1, bx0:bx1].astype(np.float32)
        roi[by0:by1, bx0:bx1] = np.clip(
            region * (1 - alpha) + layer * alpha, 0, 255
        ).astype(np.uint8)
