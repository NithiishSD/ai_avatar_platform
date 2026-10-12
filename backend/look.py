"""
Lighting, colour grade and soft-focus controls for the avatar's photo (R-44, N-15).

``contracts.AppearanceSpec`` lets a client say how the picture should look: brighter, warmer, more
contrast, auto-corrected lighting, smoother skin, a blurred background, and so on. This module turns
that spec into pixels. It works on the *still photo*, once, before the face is animated, for the same
reason the background is replaced once (``render_engine.apply_background``): the animation moves
pixels only near the eyes, brows and mouth, so a look baked into the photo carries through every
frame at no per-frame cost, and a live session pays it once when it opens.

Where it sits: ``render_engine.render_job`` and ``live_engine.LiveSession.open`` call
:func:`apply_look` after the face has been found and before ``PortraitAnimator`` is built. Nothing
here loads a model; the background blur takes a ``foreground`` function from the caller, which is the
person segmenter the background replacement already uses.

Concepts used here, explained once:

**Float working space.** Every step runs on float32 numbers in 0..1 and the result is clipped back
to 0..255 once at the end. Doing it in 8 bits would round after each step, and a brightness cut
followed by a boost would leave visible banding.

**Gamma.** Brightness is not linear to the eye: halving the light does not look half as bright.
``out = in ** (1 / gamma)`` lifts the mid-tones (gamma above 1) or deepens them (below 1) while
leaving pure black and pure white where they are, which is why it is the gentle way to fix exposure.

**Illumination correction.** A picture is the scene multiplied by the light that fell on it. The
light varies slowly across the frame (a window on one side), the scene varies quickly. A heavy blur
of the picture is therefore an estimate of the light, and dividing by it evens the light out.

**CLAHE** (contrast-limited adaptive histogram equalisation). Plain histogram equalisation stretches
the whole picture's brightness range and over-amplifies noise. CLAHE does it in small tiles and caps
how much any tile may stretch, so a face lit hard from one side is evened out locally without the
sky-high contrast of the plain version. It is run on the *lightness* channel of CIELAB only, so the
colours are not shifted.

**Grey-world white balance.** Assume the picture averages to neutral grey; if its red, green and blue
averages differ, that difference is the colour cast of the light, and scaling the channels to equal
averages removes it. The gain is capped (``MAX_WB_GAIN``) so a photo of a red wall is not turned
cyan.

**Bilateral filter.** An ordinary blur smears across edges. A bilateral filter averages a pixel only
with neighbours that are also *close in colour*, so skin texture is smoothed while the edge of an eye,
a lip or the jaw stays sharp. It is the standard cheap skin smoother.

**Unsharp mask.** Sharpen by adding back the difference between the picture and a blurred copy of
it: the blurred copy keeps the broad shapes, the difference is the fine detail, and adding a multiple
of that difference makes detail stand out.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from contracts import AppearanceSpec

# How strongly each unit of a -1..1 control acts. Chosen so the ends of the slider are strong but
# still natural: brightness +-1 moves about 64 grey levels, warmth +-1 shifts red and blue by 20%.
BRIGHTNESS_RANGE = 0.25
WARMTH_RANGE = 0.20
TINT_RANGE = 0.15
# The most any one channel is scaled by the automatic white balance.
MAX_WB_GAIN = 1.25
# Illumination correction: the blur that estimates "the light" is a quarter of the picture wide, and
# 70% of the correction is applied (exponent 0.7), so some facial shading survives.
ILLUMINATION_SIGMA = 0.25
ILLUMINATION_STRENGTH = 0.7
# CLAHE settings: a clip limit of 2 and 8 x 8 tiles are the usual defaults for portraits.
CLAHE_CLIP = 2.0
CLAHE_TILES = (8, 8)
# Rec. 709 luma weights: how much each of R, G and B contributes to perceived lightness.
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

ForegroundFn = Callable[[np.ndarray], np.ndarray]


def apply_look(
    image: np.ndarray,
    spec: Optional[AppearanceSpec],
    face_points: Optional[Sequence[Tuple[int, int]]] = None,
    foreground: Optional[ForegroundFn] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return ``(graded image, warnings)`` for an RGB uint8 photo.

    ``spec`` None or empty returns the very same array (no copy, no work). ``face_points`` are the
    face landmarks in pixels, used to confine skin smoothing to the face; ``foreground`` maps an RGB
    image to a 0..1 person mask, used for the background blur. A control that needs one of them and
    does not get it is skipped *with a warning*, never silently (golden rule 1).

    The order is fixed: background blur, automatic lighting, gamma, brightness, contrast, saturation,
    warmth, tint, skin smoothing, sharpness, vignette. Light and colour come before the detail steps
    so that smoothing and sharpening act on the final tones.
    """
    if spec is None or spec.is_empty():
        return image, []

    import cv2

    warnings: List[str] = []
    work = image.astype(np.float32) / 255.0

    # --- background blur: needs the person mask, so it runs on the untouched photo ---------------
    if spec.background_blur:
        if foreground is None:
            warnings.append("backgroundBlur was skipped: no person segmenter is available")
        else:
            mask = np.clip(foreground(image), 0.0, 1.0).astype(np.float32)
            # Sigma grows with the picture, so the same setting looks the same at 512 px and 1080p.
            sigma = 2.0 + spec.background_blur * 0.035 * max(image.shape[:2])
            blurred = cv2.GaussianBlur(work, (0, 0), sigmaX=sigma)
            m = mask[..., None]
            work = work * m + blurred * (1.0 - m)

    # --- automatic lighting correction (R-44) -----------------------------------------------------
    if spec.auto_lighting:
        work = _auto_lighting(work)

    # --- tone: gamma, brightness, contrast ---------------------------------------------------------
    if spec.gamma is not None and spec.gamma != 1.0:
        work = np.power(np.clip(work, 0.0, 1.0), 1.0 / spec.gamma)
    if spec.brightness:
        work = work + spec.brightness * BRIGHTNESS_RANGE
    if spec.contrast:
        # Pivot on the picture's own mean lightness, so contrast does not also change brightness.
        pivot = float((work @ LUMA).mean())
        work = (work - pivot) * (1.0 + spec.contrast) + pivot

    # --- colour: saturation, warmth, tint ----------------------------------------------------------
    if spec.saturation:
        grey = (work @ LUMA)[..., None]
        work = grey + (work - grey) * (1.0 + spec.saturation)
    if spec.warmth:
        work = work * np.array([1.0 + spec.warmth * WARMTH_RANGE, 1.0, 1.0 - spec.warmth * WARMTH_RANGE], dtype=np.float32)
    if spec.tint:
        # Positive tint pulls green down (towards magenta) and nudges red and blue up with it.
        work = work * np.array([1.0 + spec.tint * 0.05, 1.0 - spec.tint * TINT_RANGE, 1.0 + spec.tint * 0.05], dtype=np.float32)

    work = np.clip(work, 0.0, 1.0)

    # --- detail: skin smoothing, sharpness ---------------------------------------------------------
    if spec.skin_smoothing:
        if not face_points:
            warnings.append("skinSmoothing was skipped: no face landmarks were available to confine it to the face")
        else:
            work = _smooth_skin(work, face_points, float(spec.skin_smoothing))
    if spec.sharpness:
        soft = cv2.GaussianBlur(work, (0, 0), sigmaX=1.5)
        work = np.clip(work + spec.sharpness * 1.5 * (work - soft), 0.0, 1.0)

    # --- vignette ------------------------------------------------------------------------------------
    if spec.vignette:
        work = work * _vignette_gain(work.shape[0], work.shape[1], float(spec.vignette))[..., None]

    return (np.clip(work, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), warnings


def _auto_lighting(work: np.ndarray) -> np.ndarray:
    """
    Even out the light, then add local contrast, then remove a colour cast. ``work`` is float32 RGB 0..1.

    Three steps, because they fix three different things: a lamp on one side of the face (the
    illumination step), a flat or muddy picture (CLAHE) and a tinted light (white balance).
    """
    import cv2

    # CIELAB wants 8-bit input here; the conversion is the only place precision drops to 8 bits.
    lab = cv2.cvtColor((work * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lightness = lab[..., 0].astype(np.float32)
    # 1. Illumination: the slowly varying part of the lightness is the light, the fast part is the
    #    scene. Dividing the picture by its blurred self evens the light out; only half of the
    #    correction is applied (the exponent), so the real shading that gives a face its shape stays.
    sigma = ILLUMINATION_SIGMA * max(lightness.shape)
    light = np.maximum(cv2.GaussianBlur(lightness, (0, 0), sigmaX=sigma), 1.0)
    lightness = np.clip(lightness * np.power(lightness.mean() / light, ILLUMINATION_STRENGTH), 0.0, 255.0)
    lab[..., 0] = (lightness + 0.5).astype(np.uint8)
    # 2. Local contrast on the evened-out lightness.
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES)
    lab[..., 0] = clahe.apply(lab[..., 0])
    equalised = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    means = equalised.reshape(-1, 3).mean(axis=0)
    # Each channel is scaled toward the overall mean; the cap stops an honestly red scene being "fixed".
    gains = np.clip(means.mean() / np.maximum(means, 1e-4), 1.0 / MAX_WB_GAIN, MAX_WB_GAIN)
    return equalised * gains.astype(np.float32)


def _face_mask(shape: Tuple[int, int], face_points: Sequence[Tuple[int, int]]) -> np.ndarray:
    """A feathered 0..1 mask of the face: the convex hull of the landmarks, softened at the edge."""
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    hull = np.asarray(cv2.convexHull(np.asarray(face_points, dtype=np.int32)), dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, hull, (255,))
    # A feather of a few percent of the face width, so the smoothing fades out instead of ending in a line.
    width = max(8, int(0.04 * (hull[:, 0, 0].max() - hull[:, 0, 0].min())))
    width += 1 - width % 2  # a Gaussian kernel must have an odd size
    return cv2.GaussianBlur(mask, (width * 2 + 1, width * 2 + 1), 0).astype(np.float32) / 255.0


def _smooth_skin(work: np.ndarray, face_points: Sequence[Tuple[int, int]], amount: float) -> np.ndarray:
    """Bilateral-filter the face and fade the result in by ``amount`` (0..1)."""
    import cv2

    # Strength rises with the setting; d=0 lets OpenCV derive the window from sigmaSpace. Two passes
    # above half strength: one pass leaves the fine grain of pores, a second takes it out while the
    # edges (eyelids, lips, jaw) stay, because they are far apart in colour.
    sigma_color, sigma_space = 0.05 + 0.13 * amount, 3.0 + 7.0 * amount
    smooth = cv2.bilateralFilter(work, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    if amount > 0.5:
        smooth = cv2.bilateralFilter(smooth, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    mask = (_face_mask((int(work.shape[0]), int(work.shape[1])), face_points) * min(1.0, amount * 1.2))[..., None]
    return work * (1.0 - mask) + smooth * mask


def _vignette_gain(height: int, width: int, amount: float) -> np.ndarray:
    """Per-pixel brightness factor: 1 in the centre, falling to ``1 - 0.6 * amount`` in the corners."""
    ys = (np.arange(height, dtype=np.float32) - (height - 1) / 2.0) / (height / 2.0)
    xs = (np.arange(width, dtype=np.float32) - (width - 1) / 2.0) / (width / 2.0)
    r2 = (ys[:, None] ** 2 + xs[None, :] ** 2) / 2.0  # 0 at the centre, 1 at the corners
    return 1.0 - 0.6 * amount * r2


def expression_offsets(spec: Optional[AppearanceSpec]) -> Optional[Dict[str, float]]:
    """The standing face-shape offsets of a spec as ``{ARKit name: weight}``, or None when there are none."""
    if spec is None or spec.expression is None:
        return None
    values = {k: float(v) for k, v in spec.expression.model_dump(exclude_none=True).items() if v}
    return values or None


def head_offset(spec: Optional[AppearanceSpec]) -> Optional[Tuple[float, float, float, float]]:
    """``(roll, shift_x, shift_y, scale)`` in the renderer's pose units, or None when all are zero."""
    if spec is None:
        return None
    offset = (
        float(spec.head_roll or 0.0), float(spec.head_shift_x or 0.0),
        float(spec.head_shift_y or 0.0), float(spec.head_scale or 0.0),
    )
    return offset if any(offset) else None


def describe(spec: Optional[AppearanceSpec]) -> Optional[Dict[str, Any]]:
    """The fields the client actually set, in wire names, for the render result and the manifest."""
    if spec is None or spec.is_empty():
        return None
    return spec.model_dump(exclude_none=True, by_alias=True)

