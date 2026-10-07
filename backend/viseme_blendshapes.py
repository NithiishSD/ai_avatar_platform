"""
Viseme and emotion -> ARKit blendshape tables (tasks G2-01 and G3-02).

This is where Developer 1's output meets Developer 2's face. The aligner emits
one of 15 visemes per phoneme; MediaPipe describes a face as 52 ARKit
blendshape weights. The table below is the bridge, and ARKit weights were
chosen as the intermediate on purpose:

* the renderer in ``face_warp.py`` consumes them,
* the photo's own expression arrives from MediaPipe in the same vocabulary, so
  "what the mouth is doing now" and "what it should do" are comparable, and
* any future 3-D avatar (three.js, a FLAME rig) takes ARKit weights directly.

Weights are hand-tuned against the standard Oculus/ARKit viseme references and
are deliberately moderate. An over-driven jaw on a warped photograph reads as
a puppet; slightly under-driven reads as calm speech.

Left/right follow ARKit: they are the *subject's* left and right, so ``Left``
shapes appear on the right-hand side of the image.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

CANONICAL_VISEMES: Tuple[str, ...] = (
    "viseme_sil",
    "viseme_PP",
    "viseme_FF",
    "viseme_TH",
    "viseme_DD",
    "viseme_kk",
    "viseme_CH",
    "viseme_SS",
    "viseme_nn",
    "viseme_RR",
    "viseme_aa",
    "viseme_E",
    "viseme_I",
    "viseme_O",
    "viseme_U",
)

# Names that appear in older payloads -- including the roadmap's own example
# job -- but are not one of the 15. Mapping them keeps those payloads
# renderable; anything not listed here is reported, never guessed.
VISEME_ALIASES: Dict[str, str] = {
    "viseme_L": "viseme_nn",
    "viseme_A": "viseme_aa",
    "viseme_AA": "viseme_aa",
    "viseme_KK": "viseme_kk",
    "viseme_NN": "viseme_nn",
    "viseme_e": "viseme_E",
    "viseme_i": "viseme_I",
    "viseme_o": "viseme_O",
    "viseme_u": "viseme_U",
    "sil": "viseme_sil",
    "viseme_SIL": "viseme_sil",
}

# Blendshapes that exist as a Left/Right pair.
_PAIRED = {
    "mouthSmile",
    "mouthFrown",
    "mouthStretch",
    "mouthDimple",
    "mouthPress",
    "mouthUpperUp",
    "mouthLowerDown",
    "browDown",
    "browOuterUp",
    "cheekSquint",
    "eyeBlink",
    "eyeSquint",
    "eyeWide",
    "noseSneer",
}


def _expand(shapes: Mapping[str, float]) -> Dict[str, float]:
    """Expand symmetric shorthand (``mouthSmile``) into its Left/Right pair."""
    expanded: Dict[str, float] = {}
    for name, weight in shapes.items():
        if name in _PAIRED:
            expanded[f"{name}Left"] = weight
            expanded[f"{name}Right"] = weight
        else:
            expanded[name] = weight
    return expanded


# fmt: off
_VISEME_SHORTHAND: Dict[str, Dict[str, float]] = {
    # Rest. Lips together, jaw closed.
    "viseme_sil": {},
    # p, b, m: lips pressed shut. The closure is the whole signal, so nothing
    # here opens the jaw.
    "viseme_PP": {"mouthPress": 0.45, "mouthRollLower": 0.20, "mouthRollUpper": 0.20},
    # f, v: lower lip tucks under the upper teeth.
    "viseme_FF": {"jawOpen": 0.08, "mouthRollLower": 0.50, "mouthUpperUp": 0.22},
    # th: teeth parted, tongue tip between them.
    "viseme_TH": {"jawOpen": 0.18, "tongueOut": 0.30, "mouthUpperUp": 0.12, "mouthLowerDown": 0.15},
    # t, d: tongue behind the teeth, lips parted.
    "viseme_DD": {"jawOpen": 0.18, "mouthUpperUp": 0.12, "mouthLowerDown": 0.18, "mouthStretch": 0.10},
    # k, g: back-of-mouth closure, jaw slightly dropped.
    "viseme_kk": {"jawOpen": 0.25, "mouthLowerDown": 0.12, "mouthStretch": 0.08},
    # ch, j, sh: lips pushed forward into a flared funnel.
    "viseme_CH": {"jawOpen": 0.15, "mouthFunnel": 0.55, "mouthPucker": 0.20, "mouthUpperUp": 0.10, "mouthLowerDown": 0.15},
    # s, z: teeth together, lips drawn back.
    "viseme_SS": {"jawOpen": 0.06, "mouthStretch": 0.30, "mouthSmile": 0.15, "mouthUpperUp": 0.15, "mouthLowerDown": 0.20},
    # n, l: like DD with a softer lip opening.
    "viseme_nn": {"jawOpen": 0.16, "mouthUpperUp": 0.10, "mouthLowerDown": 0.14},
    # r: rounded, slightly protruded.
    "viseme_RR": {"jawOpen": 0.14, "mouthFunnel": 0.30, "mouthPucker": 0.30},
    # Open vowel: the widest jaw in the set.
    "viseme_aa": {"jawOpen": 0.65, "mouthLowerDown": 0.25, "mouthUpperUp": 0.10, "mouthStretch": 0.05},
    # Mid front vowel.
    "viseme_E": {"jawOpen": 0.35, "mouthStretch": 0.35, "mouthSmile": 0.15, "mouthLowerDown": 0.20, "mouthUpperUp": 0.12},
    # Close front vowel: wide, nearly closed.
    "viseme_I": {"jawOpen": 0.18, "mouthStretch": 0.45, "mouthSmile": 0.30, "mouthUpperUp": 0.12, "mouthLowerDown": 0.12},
    # Rounded open vowel.
    "viseme_O": {"jawOpen": 0.45, "mouthFunnel": 0.60, "mouthPucker": 0.25},
    # Rounded close vowel: tight pucker.
    "viseme_U": {"jawOpen": 0.20, "mouthFunnel": 0.35, "mouthPucker": 0.70},
}

# Facial expression per named emotion at weight 1.0 (task G3-02). These only
# touch shapes that do not fight the visemes for the mouth opening: brows,
# eyes, cheeks, and the mouth *corners*.
_EMOTION_SHORTHAND: Dict[str, Dict[str, float]] = {
    "joy":        {"mouthSmile": 0.55, "cheekSquint": 0.40, "eyeSquint": 0.15, "browOuterUp": 0.10},
    "excitement": {"mouthSmile": 0.45, "eyeWide": 0.35, "browInnerUp": 0.30, "browOuterUp": 0.35},
    "sorrow":     {"mouthFrown": 0.50, "browInnerUp": 0.60, "eyeSquint": 0.10},
    "anger":      {"browDown": 0.70, "mouthFrown": 0.25, "mouthPress": 0.20, "eyeSquint": 0.30, "noseSneer": 0.30},
    "authority":  {"browDown": 0.25, "mouthPress": 0.10},
    "calm":       {"mouthSmile": 0.15, "eyeSquint": 0.05},
}

# The Phase 0 ``happy`` field, used only when no named emotion is present.
_HAPPY_SHORTHAND: Dict[str, float] = {"mouthSmile": 0.50, "cheekSquint": 0.30}
# fmt: on

VISEME_BLENDSHAPES: Dict[str, Dict[str, float]] = {
    viseme: _expand(shapes) for viseme, shapes in _VISEME_SHORTHAND.items()
}
EMOTION_BLENDSHAPES: Dict[str, Dict[str, float]] = {
    emotion: _expand(shapes) for emotion, shapes in _EMOTION_SHORTHAND.items()
}

# Shapes that part the lips. A bilabial closure has to override these, and the
# speech-energy gate scales them.
OPENING_SHAPES: Tuple[str, ...] = (
    "jawOpen",
    "mouthFunnel",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "tongueOut",
)

# Visemes whose defining feature is the lips meeting, with how completely.
CLOSURE_VISEMES: Dict[str, float] = {"viseme_PP": 1.0, "viseme_FF": 0.6}


def resolve_viseme(name: str) -> Tuple[str, bool]:
    """
    Map a viseme name onto the canonical 15.

    Returns ``(canonical, known)``. An unknown name resolves to the rest pose
    and ``known=False`` so the caller can count and report it: a payload full
    of names this table has never heard of should produce a visible warning,
    not a video of someone mumbling.
    """
    if name in VISEME_BLENDSHAPES:
        return name, True
    alias = VISEME_ALIASES.get(name)
    if alias is not None:
        return alias, True
    return "viseme_sil", False


def viseme_weights(name: str) -> Dict[str, float]:
    """ARKit weights for a viseme name (aliases resolved, unknown -> rest)."""
    canonical, _ = resolve_viseme(name)
    return dict(VISEME_BLENDSHAPES[canonical])


def emotion_weights(emotion_vector: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """
    Facial-expression weights for an ``EmotionVector`` payload.

    Named emotions (``joy``, ``sorrow``, ...) drive the face when present.
    ``happy`` is the Phase 0 field that every payload carries; it is derived
    from the named emotions on the audio side, so using both would count the
    same smile twice. It is therefore used only when no named emotion is set.
    """
    if not emotion_vector:
        return {}

    def value(key: str) -> float:
        raw = emotion_vector.get(key)
        try:
            return max(0.0, min(1.0, float(raw))) if raw is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    weights: Dict[str, float] = {}
    named_total = 0.0
    for emotion, shapes in EMOTION_BLENDSHAPES.items():
        amount = value(emotion)
        if amount <= 0:
            continue
        named_total += amount
        for shape, weight in shapes.items():
            weights[shape] = weights.get(shape, 0.0) + weight * amount

    if named_total <= 0:
        happy = value("happy")
        if happy > 0:
            for shape, weight in _expand(_HAPPY_SHORTHAND).items():
                weights[shape] = weight * happy

    return {shape: min(1.0, weight) for shape, weight in weights.items()}
