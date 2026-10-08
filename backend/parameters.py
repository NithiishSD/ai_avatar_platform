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
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import contracts

M, NO, LIM, UNAV, NM = "measured", "no-effect", "limited", "unavailable", "not-measured"

# (request model, field, group, status, note)
#   group: voice | expression | appearance | render | generation
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
    (contracts.AvatarRenderJob, "renderQuality", "render", LIM, "PREVIEW caps at 512 px, HD at 1920x1080, but a photo is never enlarged: on a 512 px photo 1080P_HQ changes nothing"),
    (contracts.AvatarRenderJob, "targetFps", "render", M, "12 fps gave 60 frames against 125 at 25 fps"),
    (contracts.AvatarGenerateRequest, "age", "appearance", NM, "Stable Diffusion prompt word; SD 1.5 follows it only approximately"),
    (contracts.AvatarGenerateRequest, "presentation", "appearance", NM, "Stable Diffusion prompt word; not measured"),
    (contracts.AvatarGenerateRequest, "hair", "appearance", NM, "prompt word; a 'long-dark' request produced short hair once (T3.2), so it is not guaranteed"),
    (contracts.AvatarGenerateRequest, "glasses", "appearance", NM, "prompt word; needs a person to judge (M-08)"),
    (contracts.AvatarGenerateRequest, "seed", "generation", NM, "picks one of many faces; reproducibility not measured"),
    (contracts.AvatarGenerateRequest, "attempts", "generation", NM, "how many seeds are tried before the quality gate gives up"),
    (contracts.AvatarGenerateRequest, "steps", "generation", NM, "diffusion steps: speed against detail"),
]

# Fields on the request models that are plumbing, not customisation. Listing them here (rather than
# silently skipping) keeps the completeness test honest.
NOT_CUSTOMISATION: Dict[Any, set] = {
    contracts.AudioSynthesisRequest: {"text", "returnAlignment", "auditQuality", "outputFilename"},
    contracts.AvatarGenerateRequest: {"avatarId", "overwrite"},
    contracts.AvatarRenderJob: {"jobId", "avatarId", "audioUrl", "sampleRate", "durationSeconds", "phonemeTimestamps", "emotionVector", "background"},
}
_PARENT = {contracts.EmotionVector: "emotionVector", contracts.BackgroundSpec: "background"}
TARGET = 50


def _schema_info(model: Any, field: str) -> Dict[str, Any]:
    prop = model.model_json_schema(by_alias=True)["properties"][field]
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
    """Every parameter with its schema and measured status, and the count against the target."""
    rows = []
    for model, field, group, status, note in _ROWS:
        name = f"{_PARENT[model]}.{field}" if model in _PARENT else field
        rows.append({"name": name, "group": group, "status": status, "note": note, "request": model.__name__,
                     **_schema_info(model, field)})
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
        "target": {"appearanceParameters": TARGET, "met": False,
                   "note": "N-15 asks for 50+ appearance parameters; only the 'visual' counts above describe the avatar's look, and fewer than that are verified"},
        "measuredBy": "scripts/measure_parameters.py (8 Oct 2026, CPU); see docs/08-TESTING.md N-15",
    }


def unlisted_fields() -> List[str]:
    """Request-model fields that are neither catalogued nor declared plumbing; the completeness test expects none."""
    covered = {(m, f) for m, f, *_ in _ROWS}
    missing = []
    models: List[Any] = [contracts.AudioSynthesisRequest, contracts.AvatarGenerateRequest, contracts.AvatarRenderJob,
                         contracts.EmotionVector, contracts.BackgroundSpec]
    for model in models:
        skip: Optional[set] = NOT_CUSTOMISATION.get(model)
        for name, info in model.model_fields.items():
            alias = info.alias or name
            if (model, alias) not in covered and alias not in (skip or set()):
                missing.append(f"{model.__name__}.{alias}")
    return missing
