from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RenderQuality(str, Enum):
    PREVIEW = "PREVIEW"
    HD_1080P = "1080P_HQ"


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SynthesisMode(str, Enum):
    FAST = "fast"
    CLONE = "clone"
    HIGH_QUALITY = "high_quality"
    DIALOGUE = "dialogue"
    # Phase 3: force the MMS-TTS path for one of its 1000+ languages.
    MULTILINGUAL = "multilingual"


class PhonemeTimestamp(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    phoneme: str = Field(min_length=1)
    viseme: str = Field(min_length=1)
    start_ms: int = Field(alias="startMs", ge=0)
    end_ms: int = Field(alias="endMs", gt=0)

    @model_validator(mode="after")
    def validate_interval(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("endMs must be greater than startMs")
        return self


class EmotionVector(BaseModel):
    """
    Facial expression weights sent to the renderer.

    ``happy`` / ``neutral`` / ``eyeblinkRate`` are the Phase 0 frozen fields and
    stay required. Phase 3 adds the roadmap's named emotions as *optional*
    fields, so a Phase 0 payload validates unchanged while an emotion-aware
    producer can send the full vector.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    happy: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    eyeblink_rate: float = Field(alias="eyeblinkRate", ge=0, le=10)

    # Phase 3 emotion prosody vector (optional, defaults preserve Phase 0).
    joy: Optional[float] = Field(default=None, ge=0, le=1)
    anger: Optional[float] = Field(default=None, ge=0, le=1)
    sorrow: Optional[float] = Field(default=None, ge=0, le=1)
    authority: Optional[float] = Field(default=None, ge=0, le=1)
    calm: Optional[float] = Field(default=None, ge=0, le=1)
    excitement: Optional[float] = Field(default=None, ge=0, le=1)


class AvatarRenderJob(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    job_id: str = Field(alias="jobId", min_length=1)
    avatar_id: str = Field(alias="avatarId", min_length=1)
    audio_url: str = Field(alias="audioUrl", min_length=1)
    sample_rate: int = Field(alias="sampleRate", ge=8000, le=192000)
    duration_seconds: float = Field(alias="durationSeconds", gt=0, le=3600)
    phoneme_timestamps: List[PhonemeTimestamp] = Field(alias="phonemeTimestamps", min_length=1)
    emotion_vector: EmotionVector = Field(alias="emotionVector")
    render_quality: RenderQuality = Field(alias="renderQuality")
    target_fps: int = Field(alias="targetFps", ge=1, le=120)

    @field_validator("audio_url")
    @classmethod
    def validate_audio_url(cls, value: str) -> str:
        if "://" not in value:
            raise ValueError("audioUrl must be an absolute storage or HTTP URL")
        return value

    @model_validator(mode="after")
    def validate_timestamps(self):
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
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    text: str = Field(min_length=1)
    mode: SynthesisMode = SynthesisMode.FAST
    language: str = Field(default="en", min_length=2, max_length=16)
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
    output_filename: str = Field(default="speech.wav", alias="outputFilename", min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value


class AudioSynthesisResponse(BaseModel):
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
    model_config = ConfigDict(populate_by_name=True)

    task_id: str = Field(alias="taskId")
    status: str
    model_used: Optional[str] = Field(default=None, alias="modelUsed")
    output_path: Optional[str] = Field(default=None, alias="outputPath")
    duration_seconds: Optional[float] = Field(default=None, alias="durationSeconds")
    phoneme_timestamps: Optional[List[PhonemeTimestamp]] = Field(default=None, alias="phonemeTimestamps")
    # "mms_fa" = measured from the audio; "acoustic-fallback" = estimated.
    alignment_method: Optional[str] = Field(default=None, alias="alignmentMethod")
    emotion: Optional[Dict[str, Any]] = Field(default=None)
    quality_report: Optional[Dict[str, Any]] = Field(default=None, alias="qualityReport")
    language: Optional[Dict[str, Any]] = Field(default=None)
    latency_ms: Optional[float] = Field(default=None, alias="latencyMs")


class AlignmentRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    audio_path: str = Field(alias="audioPath", min_length=1)
    transcript: str = Field(min_length=1)
    language: str = Field(default="en", min_length=2, max_length=16)
    sample_rate: int = Field(default=24000, alias="sampleRate", ge=8000, le=192000)


class AlignmentResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    phoneme_timestamps: List[PhonemeTimestamp] = Field(alias="phonemeTimestamps")
    duration_seconds: float = Field(alias="durationSeconds")
    phoneme_count: int = Field(alias="phonemeCount")

# ---------------------------------------------------------------------------
# Phase 3 - multilingual synthesis, emotion prosody, quality auditing
# ---------------------------------------------------------------------------


class LanguageEntry(BaseModel):
    """One entry of the MMS-TTS language catalogue."""

    model_config = ConfigDict(populate_by_name=True)

    requested: str
    iso3: str
    name: str
    mms_supported: bool = Field(alias="mmsSupported")
    mms_model: Optional[str] = Field(default=None, alias="mmsModel")
    xtts_supported: bool = Field(alias="xttsSupported")
    is_english: bool = Field(alias="isEnglish")


class LanguagesResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    total: int
    returned: int
    query: str = ""
    source: str
    languages: List[LanguageEntry]


class EmotionPresetEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    label: str
    description: str
    prosody: Dict[str, float]
    render_hint: Dict[str, float] = Field(alias="renderHint")


class EmotionPresetsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    presets: List[EmotionPresetEntry]
    default: str = "neutral"


class QualityAuditRequest(BaseModel):
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
    model_config = ConfigDict(populate_by_name=True)

    audio_path: str = Field(alias="audioPath")
    report: Dict[str, Any]


class VoiceSimilarityRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    reference_path: str = Field(alias="referencePath", min_length=1)
    generated_path: str = Field(alias="generatedPath", min_length=1)


class VoiceSimilarityResponse(BaseModel):
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
    usable: bool
    usability_reason: str = Field(alias="usabilityReason")
    provenance: Dict[str, Any] = Field(default_factory=dict)


class AvatarFacesResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    avatars: List[AvatarFaceEntry]
    consent_bases: List[str] = Field(alias="consentBases")
    render_engines: Dict[str, bool] = Field(alias="renderEngines")


class FaceQualityResponse(BaseModel):
    """The quality gate's verdict, in the uploader's language."""

    model_config = ConfigDict(populate_by_name=True)

    passed: bool
    face_count: int = Field(alias="faceCount")
    errors: List[Dict[str, str]] = Field(default_factory=list)
    warnings: List[Dict[str, str]] = Field(default_factory=list)


class FaceAnalysisResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    quality: FaceQualityResponse
    # Null when no face was found. ``landmarks`` inside it is only present
    # when the request asked for them: 478 points is about 30 kB of JSON.
    analysis: Optional[Dict[str, Any]] = None
    embedded_notices: Dict[str, str] = Field(default_factory=dict, alias="embeddedNotices")


class AvatarRegisterResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    avatar: AvatarFaceEntry
    quality: Optional[FaceQualityResponse] = None


class LipSyncScoreResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(alias="jobId")
    score: Dict[str, Any]
