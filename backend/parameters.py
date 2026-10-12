"""
The customisation parameters the API accepts, and what is actually known about each (N-15, T8.7).

The target asks for "50+ appearance parameters". This catalogue is the honest answer to how many
there are: every field a client can set that is meant to change the avatar's look or voice, with
the type, range and default read straight from the request models (so they cannot drift), plus a
``status`` saying what was found when the parameter was *measured* (``scripts/measure_parameters.py``,
8 Oct 2026, CPU):

* ``measured``      -- changing it moved the output as asked, past a stated threshold.
* ``no-effect``     -- accepted and validated, but the output did not change. Reported, not hidden.
* ``limited``       -- works only under a condition, named in ``note``.
* ``unavailable``   -- the setting needs a model this stack cannot run.
* ``not-measured``  -- accepted; no automated check exists (a person has to judge it, M-08).

``test_parameters.py`` fails if a model field is added without a row here, or a row names a field
that no longer exists.

Where it sits: ``app.py`` serves :func:`catalogue` as a read-only endpoint. Nothing in the render
path reads this module; it only describes the request models in ``contracts.py``.

Why the schema is read, not copied: Pydantic can describe any model as JSON Schema
(``model_json_schema``). Pulling type, range and default from there means a change to a field's
bounds in ``contracts.py`` shows up here with no second edit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import contracts

# Short aliases for the five statuses, so each row in the table below fits on one line.
M, NO, LIM, UNAV, NM = "measured", "no-effect", "limited", "unavailable", "not-measured"

# (request model, field, group, status, note)
#   group: voice | expression | appearance | render | generation
# The request model is the class itself, not its name, so a renamed class fails at import time
# instead of producing a stale row. The note records what the measurement found, in plain words.
_ROWS: List[Tuple[Any, str, str, str, str]] = [
    (contracts.AudioSynthesisRequest, "mode", "voice", NM, "chooses the engine (fast, clone, high_quality, dialogue, multilingual); each engine has its own tests, not one before/after number"),
    (contracts.AudioSynthesisRequest, "language", "voice", M, "es vs en changes the speech; 1,100+ languages listed at /audio/languages, but only some were scored for lip sync"),
    (contracts.AudioSynthesisRequest, "quality", "voice", LIM, "'fast' = 'balanced' on Kokoro (no change); 'high' routes to Higgs TTS 2, which cannot run on this stack"),
    (contracts.AudioSynthesisRequest, "style", "voice", LIM, "only 'dialogue' changes the audio (it selects Bark); 'expressive' and 'narration' did not change the output"),
    (contracts.AudioSynthesisRequest, "speed", "voice", M, "1.5 shortened 4.98 s to 3.32 s; 0.7 lengthened it to 7.11 s"),
    (contracts.AudioSynthesisRequest, "pitch", "voice", M, "1.5 raised median F0 by 38-48% across runs; 0.7 lowered it by 20-25%"),
    (contracts.AudioSynthesisRequest, "emotion", "voice", M, "all six presets changed duration, loudness and pitch in their documented directions"),
    (contracts.AudioSynthesisRequest, "emotionIntensity", "voice", M, "0.3 sits between neutral and full strength (loudness)"),
    (contracts.AudioSynthesisRequest, "emotionVector", "voice", NM, "a blend of the six presets; each preset is measured, the blend itself is not"),
    (contracts.AudioSynthesisRequest, "speakerWav", "voice", NM, "clone reference; similarity is measured (N-02, 62%), not a before/after of this field"),
    (contracts.AudioSynthesisRequest, "voice", "voice", NM, "Kokoro speaker (2 female, 2 male) so the voice can match the face; a unit test shows the choice reaches Kokoro, the audible difference is not measured by the script"),
    (contracts.AudioSynthesisRequest, "cloneEngine", "voice", NM, "xtts-v2 vs openvoice-v2 measured separately in T2.1 / T3.3"),
    (contracts.EmotionVector, "happy", "expression", M, "smile at full strength changed the picture; ignored when a named emotion is set"),
    (contracts.EmotionVector, "neutral", "expression", NO, "accepted but never read by the face rig: setting it to 0 changed nothing"),
    (contracts.EmotionVector, "eyeblinkRate", "expression", M, "0 gave no blinks, 5 gave three times the baseline"),
    (contracts.EmotionVector, "joy", "expression", M, "changed the picture"),
    (contracts.EmotionVector, "anger", "expression", M, "changed the picture"),
    (contracts.EmotionVector, "sorrow", "expression", M, "changed the picture"),
    (contracts.EmotionVector, "authority", "expression", M, "changed the picture"),
    (contracts.EmotionVector, "calm", "expression", M, "changed the picture"),
    (contracts.EmotionVector, "excitement", "expression", M, "changed the picture"),
    (contracts.BackgroundSpec, "color", "appearance", M, "a #RRGGBB background replaced the old one"),
    (contracts.BackgroundSpec, "imageUrl", "appearance", M, "an image under outputs/ or inputs/ replaced the old background"),
    (contracts.AvatarRenderJob, "renderQuality", "render", M, "PREVIEW caps at 512 px; 1080P_HQ fits 1920x1080, and a photo that would give less than 720p is first enlarged with Real-ESRGAN (T8.8: a 512 px photo renders at 1080x1080, its added detail synthesised)"),
    (contracts.AvatarRenderJob, "motionIntensity", "appearance", M, "head tilt / nod / sway and breathing while talking (I-01); 0 still, 1 natural, 2 lively: head wander (std of face position and roll across 14 frames) 2.8 -> 11.8 from 0 to 2.0"),
    (contracts.AvatarRenderJob, "targetFps", "render", M, "12 fps gave 60 frames against 125 at 25 fps"),
    (contracts.AvatarGenerateRequest, "age", "appearance", NM, "Stable Diffusion prompt word; SD 1.5 follows it only approximately"),
    (contracts.AvatarGenerateRequest, "presentation", "appearance", NM, "Stable Diffusion prompt word; not measured"),
    (contracts.AvatarGenerateRequest, "hair", "appearance", NM, "prompt word; a 'long-dark' request produced short hair once (T3.2), so it is not guaranteed"),
    (contracts.AvatarGenerateRequest, "glasses", "appearance", NM, "prompt word; needs a person to judge (M-08)"),
    (contracts.AvatarGenerateRequest, "seed", "generation", NM, "picks one of many faces; reproducibility not measured"),
    (contracts.AvatarGenerateRequest, "attempts", "generation", NM, "how many seeds are tried before the quality gate gives up"),
    (contracts.AvatarGenerateRequest, "steps", "generation", NM, "diffusion steps: speed against detail"),
]

# Lighting, grade, head pose and standing expressions (T9.7): one row per field of AppearanceSpec and
# ExpressionOffsets, with what ``scripts/measure_parameters.py`` found on 11 Oct 2026 (CPU, the demo face).
# The notes are the measured numbers: setting, the quantity read from the rendered frames, before -> after.
_APPEARANCE_NOTES: Dict[str, str] = {
    "brightness": "+0.6: mean lightness 104 -> 142; -0.6: mean lightness 104 -> 68.6",
    "contrast": "+0.7: lightness spread 48.5 -> 68.3",
    "gamma": "1.7: mean lightness 104 -> 145",
    "saturation": "-1.0: channel spread 24.6 -> 0.0444",
    "warmth": "+0.7: red minus blue 13.5 -> 42.2",
    "tint": "+0.7: mean green 102 -> 90.7",
    "sharpness": "1.0: texture (Laplacian var) 68.5 -> 333",
    "skinSmoothing": "1.0: cheek texture 22.3 -> 13.9",
    "vignette": "1.0: corner / centre lightness 1.03 -> 0.624",
    "backgroundBlur": "1.0 over a noise background: background texture 38.2 -> 4.42",
    "autoLighting": "lamp on one side: left/right lightness gap 51.2 -> 35.7; on the demo photo: pixel diff 0 -> 16.4",
    "headRoll": "8.0: face roll 0.793 -> -6.42",
    "headShiftX": "0.06: face cx 354 -> 370",
    "headShiftY": "0.06: face cy 324 -> 340",
    "headScale": "0.08: face height 270 -> 292",
    "jawOpen": "0.8: mouthOpen 8.5 -> 34.1",
    "mouthClose": "0.8: mouthOpen 8.5 -> 4.75",
    "mouthFunnel": "0.8: mouthWidth 73.4 -> 67.1",
    "mouthPucker": "0.8: mouthWidth 73.4 -> 60.6",
    "mouthRollLower": "0.8: mouthOpen 8.5 -> 7.38",
    "mouthRollUpper": "0.8: mouthOpen 8.5 -> 7.38",
    "mouthSmile": "0.8: cornerLift -3.12 -> 4.94",
    "mouthFrown": "0.8: cornerLift -3.12 -> -9.62",
    "mouthStretch": "0.8: mouthWidth 73.4 -> 82.2",
    "mouthDimple": "0.8: mouthWidth 73.4 -> 77.4",
    "mouthPress": "0.8: mouthOpen 8.5 -> 6.62",
    "mouthUpperUp": "0.8: mouthOpen 8.5 -> 15.1",
    "mouthLowerDown": "0.8: mouthOpen 8.5 -> 13.4",
    "tongueOut": "0.8: pixel diff 0 -> 0.455",
    "eyeBlinkLeft": "0.8: eyeL 15 -> 8.62",
    "eyeBlinkRight": "0.8: eyeR 14.8 -> 7.5",
    "eyeSquintLeft": "0.8: eyeL 15 -> 14.1",
    "eyeSquintRight": "0.8: eyeR 14.8 -> 12.9",
    "eyeWideLeft": "0.8: eyeL 15 -> 16.4",
    "eyeWideRight": "0.8: eyeR 14.8 -> 15.6",
    "browInnerUp": "0.8: browInnerL 17.1 -> 25.6; 0.8 (right): browInnerR 16.9 -> 25.2",
    "browOuterUpLeft": "0.8: browOuterL 13.4 -> 17.8",
    "browOuterUpRight": "0.8: browOuterR 13.4 -> 17",
    "browDownLeft": "0.8: browMidL 24 -> 19.4",
    "browDownRight": "0.8: browMidR 24.4 -> 19.4"
}

_ROWS += [
    (contracts.AppearanceSpec, alias, "appearance", M, _APPEARANCE_NOTES[alias])
    for alias in ("autoLighting", "brightness", "contrast", "gamma", "saturation", "warmth", "tint", "sharpness",
                  "skinSmoothing", "vignette", "backgroundBlur", "headRoll", "headShiftX", "headShiftY", "headScale")
] + [
    (contracts.ExpressionOffsets, shape, "expression", M, _APPEARANCE_NOTES[shape])
    for shape in contracts.EXPRESSION_SHAPES
]

# Fields on the request models that are plumbing, not customisation. Listing them here (rather than
# silently skipping) keeps the completeness test honest.
NOT_CUSTOMISATION: Dict[Any, set] = {
    contracts.AudioSynthesisRequest: {"text", "returnAlignment", "auditQuality", "outputFilename"},
    contracts.AvatarGenerateRequest: {"avatarId", "overwrite"},
    contracts.AvatarRenderJob: {"jobId", "avatarId", "audioUrl", "sampleRate", "durationSeconds", "phonemeTimestamps", "emotionVector", "background", "appearance"},
    # ``expression`` is a nested object; its fields are the rows of ExpressionOffsets.
    contracts.AppearanceSpec: {"expression"},
}
# Nested models: their fields are reported with a dotted name ("emotionVector.happy"), because a
# client sets them inside that parent object, not at the top level of the request.
_PARENT = {
    contracts.EmotionVector: "emotionVector",
    contracts.BackgroundSpec: "background",
    contracts.AppearanceSpec: "appearance",
    contracts.ExpressionOffsets: "appearance.expression",
}
# N-15's "50+ appearance parameters", the number the counts are compared against.
TARGET = 50


def _schema_info(model: Any, field: str) -> Dict[str, Any]:
    """The client-facing schema facts for one field: type, enum, bounds, pattern and default.

    ``by_alias=True`` because clients send the alias (camelCase), so that is the key to look up.
    """
    prop = model.model_json_schema(by_alias=True)["properties"][field]
    # Only the keys a client needs to build a form; titles and descriptions are left out.
    keep = ("type", "enum", "minimum", "maximum", "exclusiveMinimum", "minLength", "pattern", "default", "anyOf")
    info = {key: prop[key] for key in keep if key in prop}
    # ``anyOf`` is how Optional[...] is written; unwrap to the non-null branch so clients see one type.
    branches = [b for b in info.pop("anyOf", []) if b.get("type") != "null"]
    if branches:
        info.update({k: v for k, v in branches[0].items() if k in keep})
    if "$ref" in prop:  # an enum defined elsewhere in the schema
        info["ref"] = prop["$ref"].rsplit("/", 1)[-1]
    return info


def catalogue() -> Dict[str, Any]:
    """Every parameter with its schema and measured status, and the count against the target.

    Returns a JSON-ready dict: ``parameters`` (one row per field), ``counts`` per status and per
    visual group, ``target`` (N-15, never marked met here) and ``measuredBy`` (where the numbers came from).
    """
    rows = []
    # Each row merges the hand-written facts (group, status, note) with the schema-derived ones.
    for model, field, group, status, note in _ROWS:
        name = f"{_PARENT[model]}.{field}" if model in _PARENT else field
        rows.append({"name": name, "group": group, "status": status, "note": note, "request": model.__name__,
                     **_schema_info(model, field)})
    # A tiny counting helper; the noqa silences ruff's "do not assign a lambda" rule for this one line.
    count = lambda pred: sum(1 for r in rows if pred(r))  # noqa: E731
    visual = ("appearance", "expression")
    return {
        "parameters": rows,
        "counts": {
            "listed": len(rows),
            "measuredWorking": count(lambda r: r["status"] == M),
            "noEffect": count(lambda r: r["status"] == NO),
            "limited": count(lambda r: r["status"] == LIM),
            "notMeasured": count(lambda r: r["status"] == NM),
            "visualListed": count(lambda r: r["group"] in visual),
            "visualMeasuredWorking": count(lambda r: r["group"] in visual and r["status"] == M),
        },
        # "met" is derived from the evidence, never set by hand: only parameters whose effect was
        # measured count, and a row downgraded to not-measured takes the target back under 50.
        "target": {"appearanceParameters": TARGET,
                   "met": count(lambda r: r["group"] in visual and r["status"] == M) >= TARGET,
                   "note": "N-15 asks for 50+ appearance parameters with real-time preview; the count is the visual "
                           "rows (appearance and expression groups) whose effect was measured on rendered output. "
                           "Preview: a live session applies and updates them (T9.8)"},
        "measuredBy": "scripts/measure_parameters.py (8 Oct 2026 for voice, background, emotion; 11 Oct 2026 for "
                      "appearance); see docs/08-TESTING.md N-15",
    }


def unlisted_fields() -> List[str]:
    """Request-model fields that are neither catalogued nor declared plumbing; the completeness test expects none.

    Returns ``"Model.field"`` strings, empty when the catalogue is complete.
    """
    # (model, field) pairs, so the same field name on two models is tracked separately.
    covered = {(m, f) for m, f, *_ in _ROWS}
    missing = []
    models: List[Any] = [contracts.AudioSynthesisRequest, contracts.AvatarGenerateRequest, contracts.AvatarRenderJob,
                         contracts.EmotionVector, contracts.BackgroundSpec, contracts.AppearanceSpec,
                         contracts.ExpressionOffsets]
    for model in models:
        skip: Optional[set] = NOT_CUSTOMISATION.get(model)
        for name, info in model.model_fields.items():
            # Rows use the wire name; a field without an alias is sent under its Python name.
            alias = info.alias or name
            if (model, alias) not in covered and alias not in (skip or set()):
                missing.append(f"{model.__name__}.{alias}")
    return missing
