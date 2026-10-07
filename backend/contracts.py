"""
The frozen contract between the audio side and the vision side.

Every shape that crosses a boundary in this project is defined here: what the
API accepts, what it returns, and what the render worker is handed. Golden
rule 4 says ``AvatarRenderJob`` is the *only* interface between audio and
vision, so changing anything in this file means updating both sides, updating
the tests, and recording it in ``docs/context.md``.

Concepts used throughout, explained once here:

**Pydantic** is a validation library. You declare a class with typed fields and
Pydantic builds a validator from the annotations: ``Model(**data)`` either
returns a fully checked object or raises. Compared with a plain ``dataclass``,
you get coercion ("24000" becomes 24000), constraint checks, and JSON
schema generation for free. FastAPI reads these classes to validate requests
and to produce the ``/docs`` page.

**How to say this in an interview:** "We validate at the trust boundary with
Pydantic models, so by the time a payload reaches business logic it is already
well-formed and typed - no defensive checks scattered through the code."
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RenderQuality(str, Enum):
    """
    Output quality tiers the renderer understands.

    Inheriting from ``str`` as well as ``Enum`` makes this a *string enum*: the
    member compares equal to its value, so ``RenderQuality.PREVIEW == "PREVIEW"``
    is True and ``json.dumps`` can serialise it without a custom encoder. A
    plain ``Enum`` would need ``.value`` everywhere.

    Why an enum at all instead of a free string: the set of valid values lives
    in one place, and Pydantic rejects anything outside it at the boundary.
    """

    PREVIEW = "PREVIEW"
    # The member name may differ from its wire value. Callers send "1080P_HQ";
    # Python cannot start an identifier with a digit, hence the HD_ prefix.
    HD_1080P = "1080P_HQ"


class JobStatus(str, Enum):
    """
    The four states a queued job moves through.

    Both queue backends (in-memory and Celery) report these same four, so the
    frontend polls one vocabulary regardless of which backend is configured.
    COMPLETED and FAILED are terminal.
    """

    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SynthesisMode(str, Enum):
    """
    What the caller wants from speech synthesis; the router maps this to a model.

    This is *intent*, not a model name, on purpose: the caller asks for "fast"
    or "clone" and ``voice_engine.VoiceEngineRouter`` decides which of the five
    engines serves it. That indirection is what lets a model be swapped without
    touching the API contract.
    """

    FAST = "fast"
    CLONE = "clone"
    HIGH_QUALITY = "high_quality"
    DIALOGUE = "dialogue"
    # Phase 3: force the MMS-TTS path for one of its 1000+ languages.
    MULTILINGUAL = "multilingual"


class PhonemeTimestamp(BaseModel):
    """
    One sound, and the window of time the mouth should be making it.

    This is the unit the whole lip-sync pipeline is built on: the aligner emits
    a list of these, and ``face_animation`` turns them into per-frame blendshape
    weights. ``phoneme`` is the linguistic sound (ARPAbet/IPA); ``viseme`` is
    its *visible* mouth shape, and many phonemes share one viseme because /p/,
    /b/ and /m/ look identical from outside.
    """

    # ConfigDict is Pydantic v2's replacement for the old inner `class Config`.
    #   populate_by_name=True  -> accept EITHER the field name (start_ms) or its
    #                             alias (startMs). Python code can use snake_case
    #                             while JSON stays camelCase.
    #   extra="forbid"         -> reject unknown keys instead of ignoring them.
    #                             A typo'd "startMS" becomes a 422 error rather
    #                             than a silently dropped field, which is the
    #                             whole point of validating at the boundary.
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # Field() attaches constraints to a type. min_length=1 on a str rejects "".
    phoneme: str = Field(min_length=1)
    viseme: str = Field(min_length=1)
    # ge = "greater than or equal", gt = "greater than". So a phoneme may start
    # at 0 ms but cannot end at 0 ms, since a zero-length sound is meaningless.
    start_ms: int = Field(alias="startMs", ge=0)
    end_ms: int = Field(alias="endMs", gt=0)

    # A *model* validator sees the finished object, so it can compare two fields
    # against each other. A field_validator only ever sees one value, which is
    # why this cannot be expressed as a constraint on end_ms alone.
    # mode="after" means "run once the individual fields have been validated and
    # coerced" - so self.end_ms is guaranteed to already be an int here.
    @model_validator(mode="after")
    def validate_interval(self):
        if self.end_ms <= self.start_ms:
            # Raising ValueError (not returning False) is how a Pydantic
            # validator fails; FastAPI turns it into a 422 with this message.
            raise ValueError("endMs must be greater than startMs")
        # An "after" validator must return the instance, not None.
        return self


class EmotionVector(BaseModel):
    """
    Facial expression weights sent to the renderer.

    ``happy`` / ``neutral`` / ``eyeblinkRate`` are the Phase 0 frozen fields and
    stay required. Phase 3 adds the roadmap's named emotions as *optional*
    fields, so a Phase 0 payload validates unchanged while an emotion-aware
    producer can send the full vector.

    This is the standard way to extend a frozen contract without breaking it:
    new fields are optional with a default, so old callers stay valid. Making
    any of the Phase 3 emotions required would have been a breaking change.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # 0..1 weights, so the renderer can blend them without rescaling.
    happy: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    # Blinks per second, not a 0..1 weight - hence the different ceiling.
    eyeblink_rate: float = Field(alias="eyeblinkRate", ge=0, le=10)

    # Phase 3 emotion prosody vector (optional, defaults preserve Phase 0).
    # Optional[float] means "float or None"; default=None is what makes a field
    # genuinely optional. Note that Optional alone does NOT imply a default -
    # without default=None these would still be required, just nullable.
    joy: Optional[float] = Field(default=None, ge=0, le=1)
    anger: Optional[float] = Field(default=None, ge=0, le=1)
    sorrow: Optional[float] = Field(default=None, ge=0, le=1)
    authority: Optional[float] = Field(default=None, ge=0, le=1)
    calm: Optional[float] = Field(default=None, ge=0, le=1)
    excitement: Optional[float] = Field(default=None, ge=0, le=1)


class AvatarRenderJob(BaseModel):
    """
    **The frozen contract.** Everything the renderer needs to make one video.

    Audio synthesis produces it, the queue carries it, the render worker
    consumes it. Nothing else crosses that line. Because both developers build
    against this shape, it was frozen at Gate 0 and mock payloads were flowed
    end to end before either side was real - which is why the two halves fitted
    together when they met.

    **How to say this in an interview:** "We froze the interface between the two
    workstreams first and passed mock payloads through it, so each side could be
    built and tested independently against a contract instead of against the
    other team's half-finished code."
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    job_id: str = Field(alias="jobId", min_length=1)
    avatar_id: str = Field(alias="avatarId", min_length=1)
    audio_url: str = Field(alias="audioUrl", min_length=1)
    # Audible range bounds: 8 kHz is telephone quality, 192 kHz is studio.
    # Anything outside this is a mistake rather than a choice.
    sample_rate: int = Field(alias="sampleRate", ge=8000, le=192000)
    # gt=0 rejects a zero-length video; le=3600 caps one job at an hour so a
    # bad request cannot occupy the single GPU indefinitely.
    duration_seconds: float = Field(alias="durationSeconds", gt=0, le=3600)
    # min_length=1 on a list means "not empty". A render job with no phonemes
    # would produce a video of a motionless face, so it is rejected as a bug.
    phoneme_timestamps: List[PhonemeTimestamp] = Field(alias="phonemeTimestamps", min_length=1)
    # Nesting models composes their validation: an invalid EmotionVector makes
    # the whole job invalid, and the error path points at the exact inner field.
    emotion_vector: EmotionVector = Field(alias="emotionVector")
    render_quality: RenderQuality = Field(alias="renderQuality")
    target_fps: int = Field(alias="targetFps", ge=1, le=120)

    # A field_validator runs on one field. The @classmethod is required by
    # Pydantic v2 (the validator belongs to the class, not an instance - it runs
    # before any instance exists), and it must come *below* @field_validator.
    @field_validator("audio_url")
    @classmethod
    def validate_audio_url(cls, value: str) -> str:
        if "://" not in value:
            raise ValueError("audioUrl must be an absolute storage or HTTP URL")
        # A field validator returns the (possibly transformed) value. Returning
        # None here would silently blank the field.
        return value

    @model_validator(mode="after")
    def validate_timestamps(self):
        """
        Two invariants that need the whole object: ordering, and fitting inside
        the stated duration.

        Catching these here rather than in the renderer means a malformed
        timeline is a 422 at the API edge, not a corrupt video twenty seconds
        into a GPU job.
        """
        # -1 rather than 0: a first phoneme legitimately starts at 0 ms, and
        # starting the comparison at 0 would reject it.
        previous_start = -1
        duration_ms = self.duration_seconds * 1000
        for timestamp in self.phoneme_timestamps:
            if timestamp.start_ms < previous_start:
                raise ValueError("phonemeTimestamps must be ordered by startMs")
            if timestamp.end_ms > duration_ms:
                raise ValueError("phoneme timestamp exceeds durationSeconds")
            previous_start = timestamp.start_ms
        return self


class RenderJobResponse(BaseModel):
    """What ``GET``/``POST /api/v1/avatar/render-job`` reports back."""

    # No extra="forbid" on responses: we construct these ourselves, so there is
    # no untrusted input to guard against, and forbidding extras on an outbound
    # model only makes it harder to evolve.
    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(alias="jobId")
    status: JobStatus
    # Populated once a worker picks the job up. Absent (not null) until then,
    # so the Phase 0 response shape is unchanged for a job that is only queued.
    engine: Optional[str] = None
    progress: Optional[float] = None
    video_url: Optional[str] = Field(default=None, alias="videoUrl")
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class AudioSynthesisRequest(BaseModel):
    """
    Everything a caller can ask of the speech side, in one request body.

    Note how much of the behaviour is expressed as *constraints* rather than as
    code: the ``pattern`` on ``quality`` and the 0.5-2.0 bounds on speed and
    pitch mean the engine never has to check them. ``description`` strings are
    not comments - FastAPI publishes them in the OpenAPI schema and the ``/docs``
    UI, so they are the API's documentation.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    text: str = Field(min_length=1)
    # A default makes the field optional for the caller. Using the enum member
    # (not the string "fast") keeps the default type-checked.
    mode: SynthesisMode = SynthesisMode.FAST
    # 16 chars is generous for a language tag: "en", "hin", "zh-Hant-TW".
    language: str = Field(default="en", min_length=2, max_length=16)
    # A regex constraint where an Enum would also have worked. Enum is usually
    # better (it is discoverable in the schema); this predates the others.
    quality: str = Field(default="balanced", pattern="^(fast|balanced|high)$")
    style: Optional[str] = Field(
        default=None,
        description="Optional style hint: 'dialogue', 'expressive', 'narration'",
    )
    speed: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=2.0,
        description="Playback rhythm/speed multiplier (0.5x to 2.0x)",
    )
    pitch: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=2.0,
        description="Pitch shift multiplier (0.5x to 2.0x)",
    )
    # None vs False matters here: None means "caller said nothing, use the
    # engine default", while False is an explicit opt-out. Alignment is off by
    # default because it loads a second model and costs time.
    return_alignment: bool = Field(
        default=False,
        alias="returnAlignment",
        description="Whether to generate millisecond phoneme/viseme timestamps",
    )
    emotion: Optional[str] = Field(
        default=None,
        description="Named emotion preset: neutral, joy, anger, sorrow, authority, calm, excitement",
    )
    emotion_intensity: float = Field(
        default=1.0,
        alias="emotionIntensity",
        ge=0.0,
        le=1.0,
        description="Strength of the named emotion; the remainder stays neutral",
    )
    # Dict[str, float] rather than EmotionVector: this is a *blend request*
    # keyed by preset name, which the emotion engine resolves into the frozen
    # EmotionVector shape. Two different ideas, deliberately two types.
    emotion_vector: Optional[Dict[str, float]] = Field(
        default=None,
        alias="emotionVector",
        description="Blend of named emotions, e.g. {'joy': 0.6, 'authority': 0.4}. Overrides 'emotion'.",
    )
    audit_quality: bool = Field(
        default=False,
        alias="auditQuality",
        description="Run the MOS/PESQ speech quality auditor on the generated audio",
    )
    speaker_wav: Optional[str] = Field(default=None, alias="speakerWav")
    # Which engine clones in mode="clone". Optional, defaulting to XTTS-v2, so
    # existing callers are unchanged; the pattern rejects anything else at the
    # edge. Explicit on purpose: the router never switches cloners on its own.
    clone_engine: Optional[str] = Field(
        default=None,
        alias="cloneEngine",
        pattern="^(xtts-v2|openvoice-v2)$",
        description="Cloning engine for mode='clone': 'xtts-v2' (default) or 'openvoice-v2'",
    )
    output_filename: str = Field(default="speech.wav", alias="outputFilename", min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        # min_length=1 already rejects "", but not "   ". Whitespace-only text
        # would reach the model and produce a silent clip, so catch it here.
        if not value.strip():
            raise ValueError("text must not be empty")
        # Returns the original value, not the stripped one: leading space can be
        # meaningful to a TTS model's prosody, so we validate without mutating.
        return value


class AudioSynthesisResponse(BaseModel):
    """
    The synchronous result of one synthesis call.

    ``model`` is here for golden rule 1: the response always states which engine
    actually produced the audio, so a fallback can never masquerade as the model
    that was asked for.
    """

    model_config = ConfigDict(populate_by_name=True)

    output_path: str = Field(alias="outputPath")
    sample_rate: int = Field(alias="sampleRate")
    duration_seconds: float = Field(alias="durationSeconds")
    latency_ms: float = Field(alias="latencyMs")
    model: str
    mode: SynthesisMode
    phoneme_timestamps: Optional[List[PhonemeTimestamp]] = Field(default=None, alias="phonemeTimestamps")
    emotion: Optional[Dict[str, Any]] = Field(default=None)
    quality_report: Optional[Dict[str, Any]] = Field(default=None, alias="qualityReport")
    language: Optional[Dict[str, Any]] = Field(default=None)


class SynthesisJobResponse(BaseModel):
    """
    The *asynchronous* view of the same work, polled by task id.

    Heavy work goes through the queue rather than the request (a 6 GB card
    cannot hold a model per HTTP worker), so the API hands back a task id
    immediately and the client polls this shape until it is terminal.
    """

    model_config = ConfigDict(populate_by_name=True)

    task_id: str = Field(alias="taskId")
    status: str
    model_used: Optional[str] = Field(default=None, alias="modelUsed")
    output_path: Optional[str] = Field(default=None, alias="outputPath")
    duration_seconds: Optional[float] = Field(default=None, alias="durationSeconds")
    phoneme_timestamps: Optional[List[PhonemeTimestamp]] = Field(default=None, alias="phonemeTimestamps")
    # "mms_fa" = measured from the audio; "acoustic-fallback" = estimated.
    # This field exists because a guessed timeline used to be indistinguishable
    # from a measured one - the exact failure golden rule 1 is written against.
    alignment_method: Optional[str] = Field(default=None, alias="alignmentMethod")
    emotion: Optional[Dict[str, Any]] = Field(default=None)
    quality_report: Optional[Dict[str, Any]] = Field(default=None, alias="qualityReport")
    language: Optional[Dict[str, Any]] = Field(default=None)
    latency_ms: Optional[float] = Field(default=None, alias="latencyMs")


class AlignmentRequest(BaseModel):
    """Align existing audio against its transcript (no synthesis involved)."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    audio_path: str = Field(alias="audioPath", min_length=1)
    transcript: str = Field(min_length=1)
    language: str = Field(default="en", min_length=2, max_length=16)
    sample_rate: int = Field(default=24000, alias="sampleRate", ge=8000, le=192000)


class AlignmentResponse(BaseModel):
    """Phoneme timeline for one clip, plus the totals the UI displays."""

    model_config = ConfigDict(populate_by_name=True)

    phoneme_timestamps: List[PhonemeTimestamp] = Field(alias="phonemeTimestamps")
    duration_seconds: float = Field(alias="durationSeconds")
    # Derived from the list above, returned anyway so a client can show a count
    # without walking the array.
    phoneme_count: int = Field(alias="phonemeCount")

# ---------------------------------------------------------------------------
# Phase 3 - multilingual synthesis, emotion prosody, quality auditing
# ---------------------------------------------------------------------------


class LanguageEntry(BaseModel):
    """One entry of the MMS-TTS language catalogue."""

    model_config = ConfigDict(populate_by_name=True)

    # What the caller typed, echoed back so they can match request to result
    # when the resolver normalised it (e.g. "hindi" -> "hin").
    requested: str
    iso3: str
    name: str
    # Which engines can actually speak it. Two separate booleans rather than one
    # "supported" flag, because the answer differs per engine and the UI needs
    # to explain *why* a language is unavailable.
    mms_supported: bool = Field(alias="mmsSupported")
    mms_model: Optional[str] = Field(default=None, alias="mmsModel")
    xtts_supported: bool = Field(alias="xttsSupported")
    is_english: bool = Field(alias="isEnglish")


class LanguagesResponse(BaseModel):
    """A page of the language catalogue."""

    model_config = ConfigDict(populate_by_name=True)

    # total = how many matched; returned = how many are in this response.
    # Both are needed for a client to know the list was truncated.
    total: int
    returned: int
    query: str = ""
    source: str
    languages: List[LanguageEntry]


class EmotionPresetEntry(BaseModel):
    """One named emotion, and what it does to voice and face."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    label: str
    description: str
    # prosody drives the *voice* (speed, pitch, energy); render_hint drives the
    # *face* (brow, cheek, blink). One preset, two consumers.
    prosody: Dict[str, float]
    render_hint: Dict[str, float] = Field(alias="renderHint")


class EmotionPresetsResponse(BaseModel):
    """The whole preset catalogue, so the UI never hardcodes emotion names."""

    model_config = ConfigDict(populate_by_name=True)

    presets: List[EmotionPresetEntry]
    default: str = "neutral"


class QualityAuditRequest(BaseModel):
    """Ask the SQUIM auditor to score a clip."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    audio_path: str = Field(alias="audioPath", min_length=1)
    reference_path: Optional[str] = Field(
        default=None,
        alias="referencePath",
        description=(
            "Any clean, non-matching speech clip. SQUIM's subjective MOS head "
            "needs one; without it the MOS is self-referenced and biased upward."
        ),
    )


class QualityAuditResponse(BaseModel):
    """
    The auditor's verdict.

    ``report`` is a loose dict rather than a typed model on purpose: it carries
    the metric *and its method* (golden rule 2), and which metrics are present
    depends on whether a reference was supplied.
    """

    model_config = ConfigDict(populate_by_name=True)

    audio_path: str = Field(alias="audioPath")
    report: Dict[str, Any]


class VoiceSimilarityRequest(BaseModel):
    """Compare a cloned clip against the reference it was cloned from."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    reference_path: str = Field(alias="referencePath", min_length=1)
    generated_path: str = Field(alias="generatedPath", min_length=1)


class VoiceSimilarityResponse(BaseModel):
    """
    ECAPA-TDNN similarity, with its admissibility.

    The report says whether the reference was a consented human recording,
    because only then may the number be reported against the >85% target - a
    score against a synthetic reference is a pipeline test, not evidence.
    """

    model_config = ConfigDict(populate_by_name=True)

    report: Dict[str, Any]


# ---------------------------------------------------------------------------
# Vision - avatar faces, face analysis, lip-sync metric
# ---------------------------------------------------------------------------


class AvatarFaceEntry(BaseModel):
    """One image in the avatar store and whether it may be animated."""

    model_config = ConfigDict(populate_by_name=True)

    avatar_id: str = Field(alias="avatarId")
    filename: str
    image_url: str = Field(alias="imageUrl")
    width: int
    height: int
    # usable is the *consent and quality* verdict, not "the file exists".
    # usability_reason carries the explanation so the UI can say why not.
    usable: bool
    usability_reason: str = Field(alias="usabilityReason")
    # default_factory, not default={}: a mutable default would be shared by
    # every instance of the model, so one avatar's provenance would leak into
    # the next. The factory builds a fresh dict per instance. This is the single
    # most common mutable-default bug in Python.
    provenance: Dict[str, Any] = Field(default_factory=dict)


class AvatarFacesResponse(BaseModel):
    """The avatar list, plus the two things the UI needs to render its form."""

    model_config = ConfigDict(populate_by_name=True)

    avatars: List[AvatarFaceEntry]
    # Served rather than hardcoded in the frontend: the valid consent bases are
    # a policy decision, and the UI must not be able to drift from the backend.
    consent_bases: List[str] = Field(alias="consentBases")
    # engine name -> are its weights actually on disk. Lets the UI disable an
    # engine instead of offering it and failing at render time.
    render_engines: Dict[str, bool] = Field(alias="renderEngines")


class FaceQualityResponse(BaseModel):
    """The quality gate's verdict, in the uploader's language."""

    model_config = ConfigDict(populate_by_name=True)

    passed: bool
    face_count: int = Field(alias="faceCount")
    # Errors block registration; warnings do not. Both are lists of dicts
    # ({code, message}) so the UI can act on the code and show the message.
    errors: List[Dict[str, str]] = Field(default_factory=list)
    warnings: List[Dict[str, str]] = Field(default_factory=list)


class FaceAnalysisResponse(BaseModel):
    """Landmarks, pose and blendshapes for one photo."""

    model_config = ConfigDict(populate_by_name=True)

    quality: FaceQualityResponse
    # Null when no face was found. ``landmarks`` inside it is only present
    # when the request asked for them: 478 points is about 30 kB of JSON.
    analysis: Optional[Dict[str, Any]] = None
    embedded_notices: Dict[str, str] = Field(default_factory=dict, alias="embeddedNotices")


class AvatarRegisterResponse(BaseModel):
    """Result of registering a photo: the stored entry, and why it passed."""

    model_config = ConfigDict(populate_by_name=True)

    avatar: AvatarFaceEntry
    # Optional because a synthetic avatar is registered by the generator, which
    # has already run the gate and does not re-report it.
    quality: Optional[FaceQualityResponse] = None


class LipSyncScoreResponse(BaseModel):
    """
    SyncNet's verdict on a rendered clip.

    ``score`` stays a dict so it can carry LSE-C, LSE-D, the best offset *and*
    the method string together - a bare float would be a number with no stated
    method, which golden rule 2 forbids.
    """

    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(alias="jobId")
    score: Dict[str, Any]
