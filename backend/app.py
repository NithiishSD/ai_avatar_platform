import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[1]
dotenv_path = project_root / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    load_dotenv()

from celery_app import celery, get_router, run_render, synthesize_audio
from contracts import (
    AudioSynthesisRequest,
    AvatarFaceEntry,
    AvatarFacesResponse,
    AvatarGenerateRequest,
    AvatarGenerateResponse,
    AvatarRegisterResponse,
    FaceAnalysisResponse,
    LipSyncScoreResponse,
    AvatarRenderJob,
    RenderJobResponse,
    SynthesisJobResponse,
    AlignmentRequest,
    AlignmentResponse,
    EmotionPresetEntry,
    EmotionPresetsResponse,
    LanguageEntry,
    LanguagesResponse,
    QualityAuditRequest,
    QualityAuditResponse,
    VoiceSimilarityRequest,
    VoiceSimilarityResponse,
)
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import avatar_generator
import avatar_store
import language_registry
from generation_jobs import GenerationJobs
from face_engine import FaceEngineUnavailable
import provenance
import render_engine
from audio_utils import list_voice_samples, SUPPORTED_EXTENSIONS
from alignment_engine import ForcedAligner, PhonemeToVisemeMapper
from emotion_engine import preset_catalogue
from job_queue import CeleryJobQueue, InMemoryJobQueue
from model_registry import (
    audit_summary,
    audit_vision_weights,
    log_vision_audit,
    log_weight_audit,
    vision_audit_summary,
)
from quality_auditor import SpeechQualityAuditor
from security import API_KEY_HEADER, SecurityGate
from voice_engine import CLONE_ENGINES, ModelWeightsMissing, VoiceConsentRequired

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Audit model weights before serving a single request.

    Printed as well as logged: the router's fallbacks are silent by design, so
    a missing checkpoint has to be impossible to miss in the server output.
    """
    statuses = log_weight_audit()
    missing = [s for s in statuses if not s.present]
    print("=" * 60)
    print(f"[Model Weights] {len(statuses) - len(missing)}/{len(statuses)} available")
    # `entry`, not `status`: assigning a name anywhere in a function makes it
    # local to the whole function, which would shadow FastAPI's `status`
    # module imported above.
    for entry in statuses:
        mark = "OK  " if entry.present else "MISS"
        print(f"  {mark}  {entry.key:<13} {entry.size_label:>8}  {entry.detail}")
    if missing:
        print(
            f"[Model Weights] {len(missing)} model(s) will fall back: "
            f"{', '.join(s.key for s in missing)}"
        )
        print("[Model Weights] Fetch them with: python scripts/fetch_models.py")
    vision = log_vision_audit()
    vision_missing = [s for s in vision if not s.present]
    print(f"[Vision Weights] {len(vision) - len(vision_missing)}/{len(vision)} available")
    for entry in vision:
        mark = "OK  " if entry.present else "MISS"
        print(f"  {mark}  {entry.key:<21} {entry.size_label:>8}  {entry.detail}")
    print("=" * 60)
    yield


app = FastAPI(
    title="AI Avatar Platform API",
    version="1.0.0",
    description="Developer 1 audio and avatar render-job service.",
    lifespan=lifespan,
)
# Vite picks the next free port when 5173 is taken, so pinning a single port
# breaks the dev server silently. Any loopback port is allowed in development;
# set CORS_ORIGINS to an explicit comma-separated list for deployment.
_cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or [],
    allow_origin_regex=None if _cors_origins else r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API-key auth + per-identity rate limiting. Both are configured from .env and
# auth is off by default so local development needs no key.
security_gate = SecurityGate()


@app.middleware("http")
async def enforce_security(request: Request, call_next):
    # CORS preflight carries no headers to authenticate; let the CORS
    # middleware answer it.
    if request.method == "OPTIONS":
        return await call_next(request)

    allowed, status_code, detail = security_gate.inspect(
        path=request.url.path,
        api_key=request.headers.get(API_KEY_HEADER),
        client_host=request.client.host if request.client else None,
    )
    if not allowed:
        return JSONResponse(
            status_code=status_code,
            content={"detail": detail.get("detail", "request rejected")},
            headers=detail.get("headers", {}),
        )
    return await call_next(request)

inputs_dir = project_root / "inputs"
inputs_dir.mkdir(parents=True, exist_ok=True)

outputs_dir = project_root / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

queue_backend = os.getenv("QUEUE_BACKEND", "in_memory").lower()
job_queue = (
    CeleryJobQueue() if queue_backend == "celery" else InMemoryJobQueue(runner=run_render)
)
faces = avatar_store.AvatarStore()
# Avatar generation runs in this process on one worker thread (generation_jobs).
generation_jobs = GenerationJobs()

# One auditor for the process: SQUIM weights load once, on first audit.
quality_auditor = SpeechQualityAuditor()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class VoiceSampleInfo(BaseModel):
    filename: str
    path: str
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    size_bytes: int
    ready_for_cloning: bool
    duration_label: str


class VoiceSamplesResponse(BaseModel):
    samples: List[VoiceSampleInfo]
    supported_formats: List[str]
    inputs_dir: str


def _render_engines() -> dict:
    """Which render engines can run right now (by checkpoint presence)."""
    from wav2lip_engine import shared_wav2lip_engine

    return {
        render_engine.ENGINE_BLENDSHAPE: True,
        render_engine.ENGINE_WAV2LIP: shared_wav2lip_engine().available,
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "queueBackend": os.getenv("QUEUE_BACKEND", "in_memory"),
        "security": security_gate.describe(),
        "capabilities": {
            # Keys the router can select. Routable is not the same as usable:
            # modelWeights below says which ones have weights on disk.
            # Higgs and Dia are not listed: they cannot load on this stack
            # (D-32), so advertising them would promise what 503s.
            "models": ["kokoro", "xtts-v2", "openvoice-v2", "bark", "mms-tts"],
            "cloneEngines": list(CLONE_ENGINES),
            "languages": language_registry.supported_count(),
            "emotions": [preset["name"] for preset in preset_catalogue()],
            "visemes": PhonemeToVisemeMapper.get_supported_visemes(),
        },
        "modelWeights": audit_summary(),
        "visionWeights": vision_audit_summary(),
        "render": {
            "defaultEngine": render_engine.default_engine(),
            "engines": _render_engines(),
        },
    }


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def resolve_audio_path(raw: str) -> Path:
    """
    Resolve a caller-supplied audio path to a file inside the project.

    Absolute paths are honoured; relative ones are tried against the project
    root, ``outputs/`` and ``inputs/``. Anything that resolves outside those
    three roots is rejected, so a path like ``../../etc/passwd`` cannot be used
    to read arbitrary files through the audit endpoints.
    """
    candidate = Path(raw)
    roots = (project_root, outputs_dir, inputs_dir)
    if not candidate.is_absolute():
        for root in roots:
            option = (root / raw)
            if option.exists():
                candidate = option
                break
        else:
            candidate = project_root / raw

    resolved = candidate.resolve()
    if not any(
        resolved == root.resolve() or root.resolve() in resolved.parents
        for root in roots
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="audio path must be inside the project inputs/ or outputs/ folder",
        )
    if not resolved.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"audio file not found: {raw}",
        )
    return resolved


# ---------------------------------------------------------------------------
# Voice samples — used by UI to list available clone reference files
# ---------------------------------------------------------------------------
@app.get("/api/v1/audio/samples", response_model=VoiceSamplesResponse)
def list_samples() -> VoiceSamplesResponse:
    """
    Return all audio files found in the inputs/ directory.
    Files are listed with format/duration metadata so the frontend
    can display them and let the user pick one for voice cloning.
    """
    samples = list_voice_samples(inputs_dir)
    return VoiceSamplesResponse(
        samples=[
            VoiceSampleInfo(
                filename=s.filename,
                path=s.path,
                duration_seconds=round(s.duration_seconds, 2),
                sample_rate=s.sample_rate,
                channels=s.channels,
                format=s.format,
                size_bytes=s.size_bytes,
                ready_for_cloning=s.ready_for_cloning,
                duration_label=s.duration_label,
            )
            for s in samples
        ],
        supported_formats=sorted(SUPPORTED_EXTENSIONS),
        inputs_dir=str(inputs_dir),
    )



# ---------------------------------------------------------------------------
# Avatar faces -- the consent-enforcing store, and face analysis
# ---------------------------------------------------------------------------
def _quality_payload(report) -> dict:
    return report.to_dict()


async def _read_upload(file: UploadFile) -> bytes:
    """Read an upload, refusing one over the limit without buffering all of it."""
    limit = avatar_store.MAX_UPLOAD_BYTES
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"image is larger than the {limit // (1024 * 1024)} MB limit",
        )
    return data


@app.get("/api/v1/avatar/faces", response_model=AvatarFacesResponse)
def list_avatar_faces() -> AvatarFacesResponse:
    """Every image in ``inputs/faces`` and whether its provenance permits use."""
    return AvatarFacesResponse(
        avatars=[AvatarFaceEntry.model_validate(r.to_dict()) for r in faces.list()],
        consentBases=sorted(provenance.FACE_CONSENT_BASES),
        renderEngines=_render_engines(),
    )


@app.post(
    "/api/v1/avatar/faces",
    response_model=AvatarRegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_avatar_face(
    file: UploadFile = File(...),
    avatar_id: str = Form(..., alias="avatarId"),
    subject: str = Form(""),
    consent_basis: str = Form("", alias="consentBasis"),
    licence: str = Form(""),
    notes: str = Form(""),
) -> AvatarRegisterResponse:
    """
    Register a photo of a real person as an avatar.

    Uploads are always recorded as human: a caller cannot label a photo
    "synthetic" to skip the consent basis. Synthetic faces come from
    ``scripts/make_avatar.py --synthetic``, which records how they were made.
    """
    data = await _read_upload(file)
    try:
        record, report = faces.register(
            data,
            avatar_id=avatar_id,
            source=provenance.HUMAN,
            subject=subject,
            licence=licence,
            consent_basis=consent_basis,
            notes=notes,
        )
    except avatar_store.AvatarRejected as err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(err), "quality": _quality_payload(err.report)},
        ) from err
    except avatar_store.AvatarError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    except FaceEngineUnavailable as err:  # the landmarker bundle is missing
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(err)
        ) from err
    return AvatarRegisterResponse(
        avatar=AvatarFaceEntry.model_validate(record.to_dict()),
        quality=_quality_payload(report) if report is not None else None,
    )


def _diffusion_weights() -> dict:
    """Whether the Stable Diffusion weights are on disk, as ``{present, detail}``."""
    status_ = next(s for s in audit_vision_weights() if s.key == "avatar-diffusion")
    return {"present": status_.present, "detail": status_.detail}


@app.get("/api/v1/avatar/generate/options")
def avatar_generate_options() -> dict:
    """The attribute choices ``POST /avatar/generate`` accepts, and whether it can run."""
    weights = _diffusion_weights()
    return {
        "age": list(avatar_generator.AGES),
        "presentation": list(avatar_generator.PRESENTATIONS),
        "hair": list(avatar_generator.HAIR),
        "available": bool(weights.get("present")),
        "detail": weights.get("detail", ""),
    }


@app.post(
    "/api/v1/avatar/generate",
    response_model=AvatarGenerateResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_202_ACCEPTED,
)
def generate_avatar_face(request: AvatarGenerateRequest) -> AvatarGenerateResponse:
    """
    Queue a synthetic face. It is registered as ``synthetic`` (it depicts
    nobody) with its prompt and seed recorded, then usable like any avatar.

    Refused up front when the diffusion weights are missing (503, with the
    fetch command) or the id is taken (409), so a doomed job is never queued.
    """
    weights = _diffusion_weights()
    if not weights.get("present"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=weights.get("detail", "avatar-diffusion weights missing"))
    if not request.overwrite:
        try:
            faces.get(request.avatar_id)
        except avatar_store.AvatarNotFound:
            pass
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"avatar {request.avatar_id!r} already exists; choose another id or set overwrite",
            )

    def work() -> dict:
        from face_engine import FACE_ENGINE_LOCK, shared_face_engine

        def check(image):
            with FACE_ENGINE_LOCK:
                return shared_face_engine().check_quality(image)

        generated = avatar_generator.generate_registered_avatar(
            faces,
            check,
            request.avatar_id,
            prompt=avatar_generator.build_prompt(request.age, request.presentation, request.hair, request.glasses),
            seed=request.seed,
            attempts=request.attempts,
            steps=request.steps,
            overwrite=request.overwrite,
        )
        record = faces.get(request.avatar_id)
        return {
            "avatarId": request.avatar_id,
            "imageUrl": record.to_dict()["imageUrl"],
            "seed": generated.seed,
            "rejectedSeeds": {str(k): v for k, v in generated.rejected_seeds.items()},
            "provenance": record.provenance,
        }

    return AvatarGenerateResponse(taskId=generation_jobs.submit(work), status="QUEUED")


@app.get(
    "/api/v1/avatar/generate/{task_id}",
    response_model=AvatarGenerateResponse,
    response_model_exclude_none=True,
)
def get_avatar_generation(task_id: str) -> AvatarGenerateResponse:
    job = generation_jobs.get(task_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown generation task {task_id!r}")
    return AvatarGenerateResponse.model_validate(job)


def _avatar_or_http(avatar_id: str, usable: bool = True):
    try:
        return faces.require_usable(avatar_id) if usable else faces.get(avatar_id)
    except avatar_store.AvatarError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    except avatar_store.AvatarNotFound as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err)) from err
    except avatar_store.AvatarConsentError as err:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(err)) from err


@app.get("/api/v1/avatar/faces/{avatar_id}/image")
def get_avatar_image(avatar_id: str) -> FileResponse:
    """The avatar image. Refused for an image whose provenance forbids use."""
    record = _avatar_or_http(avatar_id)
    return FileResponse(record.path, media_type="image/png")


@app.delete("/api/v1/avatar/faces/{avatar_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_avatar_face(avatar_id: str) -> None:
    _avatar_or_http(avatar_id, usable=False)
    faces.delete(avatar_id)


@app.post("/api/v1/avatar/face/analyze", response_model=FaceAnalysisResponse)
async def analyze_face(
    file: Optional[UploadFile] = File(None),
    avatar_id: Optional[str] = Form(None, alias="avatarId"),
    include_landmarks: bool = Form(False, alias="includeLandmarks"),
) -> FaceAnalysisResponse:
    """
    Landmarks, head pose, blendshapes and the quality verdict for one photo.

    Send either an image ``file`` (analysed and discarded, never stored) or
    the ``avatarId`` of a registered face.
    """
    from face_engine import FACE_ENGINE_LOCK, shared_face_engine

    if (file is None) == (avatar_id is None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="send exactly one of: an image 'file', or an 'avatarId'",
        )
    try:
        if file is not None:
            image, notices = avatar_store.decode_image(await _read_upload(file))
        else:
            # The exactly-one-of check above guarantees this; the assert states
            # it for the type checker, which cannot follow an XOR.
            assert avatar_id is not None
            image, notices = avatar_store.decode_image(_avatar_or_http(avatar_id).path)
        with FACE_ENGINE_LOCK:
            report = shared_face_engine().check_quality(image)
    except avatar_store.AvatarError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    except FaceEngineUnavailable as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(err)
        ) from err
    return FaceAnalysisResponse(
        quality=_quality_payload(report),
        analysis=(
            report.analysis.to_dict(include_landmarks=include_landmarks)
            if report.analysis is not None
            else None
        ),
        embeddedNotices=notices,
    )


# ---------------------------------------------------------------------------
# Render jobs
# ---------------------------------------------------------------------------
def _render_response(job_id: str, queued_job) -> RenderJobResponse:
    result = queued_job.result
    started = queued_job.status.value != "QUEUED"
    return RenderJobResponse(
        jobId=job_id,
        status=queued_job.status,
        engine=(result or {}).get("engine") or queued_job.engine,
        progress=round(queued_job.progress, 3) if started else None,
        videoUrl=(result or {}).get("videoUrl"),
        error=queued_job.error,
        result=result,
    )


@app.post(
    "/api/v1/avatar/render-job",
    response_model=RenderJobResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_render_job(job: AvatarRenderJob, engine: Optional[str] = None) -> RenderJobResponse:
    """
    Queue a render. ``?engine=blendshape|wav2lip`` picks the lip-sync engine.

    A job that could never render -- unknown avatar, an image without
    consent, audio this server cannot read, an engine with no weights -- is
    rejected here with the reason, not accepted and failed later.
    """
    if getattr(job_queue, "executes", False):
        try:
            engine = render_engine.validate_engine(engine)
            render_engine.preflight(job, engine, faces)
        except avatar_store.AvatarError as err:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
        except avatar_store.AvatarNotFound as err:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err)) from err
        except avatar_store.AvatarConsentError as err:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(err)) from err
        except render_engine.RenderError as err:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    try:
        queued_job = job_queue.enqueue(job, engine)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001 - broker or status store unreachable
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"render queue unavailable: {error}. Is Redis running (./start-docker.sh)?",
        ) from error
    return _render_response(queued_job.job.job_id, queued_job)


@app.get(
    # ``:path`` because jobId is an unconstrained contract string and may
    # contain a slash; without it such a job could be queued but never polled.
    "/api/v1/avatar/render-job/{job_id:path}",
    response_model=RenderJobResponse,
    response_model_exclude_none=True,
)
def get_render_job(job_id: str) -> RenderJobResponse:
    queued_job = job_queue.get(job_id)
    if queued_job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="render job not found")
    return _render_response(job_id, queued_job)


@app.post(
    "/api/v1/avatar/render-job/{job_id:path}/lipsync-score",
    response_model=LipSyncScoreResponse,
)
def score_render_job(job_id: str) -> LipSyncScoreResponse:
    """SyncNet LSE-C / LSE-D for a finished render. The score names its method."""
    import lipsync_metric

    queued_job = job_queue.get(job_id)
    if queued_job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="render job not found")
    if queued_job.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"render job is {queued_job.status.value}; there is no video to score yet",
        )
    try:
        score = lipsync_metric.score_video(queued_job.result["outputPath"])
    except lipsync_metric.SyncNetUnavailable as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(err)
        ) from err
    except (lipsync_metric.LipSyncMetricError, FileNotFoundError) as err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(err)
        ) from err
    return LipSyncScoreResponse(jobId=job_id, score=score.to_dict())


@app.post(
    "/api/v1/audio/align",
    response_model=AlignmentResponse,
    status_code=status.HTTP_200_OK,
)
def align_audio(request: AlignmentRequest) -> AlignmentResponse:
    """Standalone forced alignment extracting millisecond phoneme/viseme timestamps."""
    try:
        audio_path = resolve_audio_path(request.audio_path)
        aligner = ForcedAligner()
        timestamps = aligner.align(
            audio_path_or_tensor=str(audio_path),
            transcript=request.transcript,
            sample_rate=request.sample_rate,
            language=request.language,
        )
        duration_s = timestamps[-1].end_ms / 1000.0 if timestamps else 0.0
        return AlignmentResponse(
            phonemeTimestamps=timestamps,
            durationSeconds=duration_s,
            phonemeCount=len(timestamps),
        )
    except HTTPException:
        raise
    except FileNotFoundError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Alignment failed: {str(err)}",
        ) from err


def _job_response(task_id: str, task_status: str, result: dict) -> SynthesisJobResponse:
    """Project a completed VoiceEngineRouter result onto the API response."""
    return SynthesisJobResponse(
        taskId=task_id,
        status=task_status,
        modelUsed=result.get("model"),
        outputPath=result.get("output_path"),
        durationSeconds=result.get("duration_seconds"),
        phonemeTimestamps=result.get("phoneme_timestamps"),
        alignmentMethod=result.get("alignment_method"),
        emotion=result.get("emotion"),
        qualityReport=result.get("quality_report"),
        language=result.get("language"),
        latencyMs=result.get("latency_ms"),
    )


@app.post(
    "/api/v1/audio/synthesize",
    response_model=SynthesisJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_synthesis_job(request: AudioSynthesisRequest) -> SynthesisJobResponse:
    # Route before queueing, so an engine with no weights is a 503 that names
    # the fetch command now, not a queued job that fails (or downloads
    # gigabytes) later, and an unsupported language is a 400 the caller can fix.
    router = get_router()
    try:
        model_key = router.select_model(
            mode=request.mode.value,
            language=request.language,
            quality=request.quality,
            style=request.style,
            text=request.text,
            clone_engine=request.clone_engine,
        )
        router.preflight(model_key, request.speaker_wav, request.language)
    except VoiceConsentRequired as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(error)) from error
    except ModelWeightsMissing as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    try:
        task = synthesize_audio.delay(request.model_dump(by_alias=True, mode="json"))
        task_status = getattr(task, "status", "QUEUED")
        if task_status == "PENDING":
            task_status = "QUEUED"
        # For eager (in-memory) mode, task result is available immediately
        result = task.result if hasattr(task, "result") else None
        if task_status == "SUCCESS" and isinstance(result, dict):
            return _job_response(task.id, task_status, result)
        return SynthesisJobResponse(taskId=task.id, status=task_status)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Synthesis service unavailable: {str(error)}",
        ) from error


@app.get("/api/v1/audio/synthesize/{task_id}", response_model=SynthesisJobResponse)
def get_synthesis_job(task_id: str) -> SynthesisJobResponse:
    try:
        task = celery.AsyncResult(task_id)
        state_mapping = {
            "PENDING": "QUEUED",
            "STARTED": "PROCESSING",
            "SUCCESS": "SUCCESS",
            "FAILURE": "FAILED",
            "RETRY": "RETRYING",
            "REVOKED": "CANCELLED",
        }
        task_status = state_mapping.get(task.state, task.state)
        if task.state == "SUCCESS" and isinstance(task.result, dict):
            return _job_response(task_id, task_status, task.result)
        return SynthesisJobResponse(taskId=task_id, status=task_status)
    except Exception as error:  # noqa: BLE001 - a broken result backend must not 500 the poll
        # Reported as UNKNOWN so the client keeps a stable shape, but logged:
        # a status store that cannot be read is exactly the failure that would
        # otherwise leave a job looking stuck with no trace of why.
        logger.warning("Could not read status of synthesis task %s: %s", task_id, error)
        return SynthesisJobResponse(taskId=task_id, status="UNKNOWN")

# ---------------------------------------------------------------------------
# Phase 3 — multilingual catalogue, emotion presets, quality auditing
# ---------------------------------------------------------------------------
@app.get("/api/v1/audio/languages", response_model=LanguagesResponse)
def list_languages(q: str = "", limit: int = 100) -> LanguagesResponse:
    """
    Search the MMS-TTS language catalogue.

    ``q`` matches ISO-639-3 codes and language names; ``limit=0`` returns the
    whole catalogue.
    """
    limit = max(0, min(limit, language_registry.supported_count()))
    matches = language_registry.search(q, limit=limit)
    return LanguagesResponse(
        total=language_registry.supported_count(),
        returned=len(matches),
        query=q,
        source=language_registry.catalogue_source(),
        languages=[LanguageEntry.model_validate(info.to_dict()) for info in matches],
    )


@app.get("/api/v1/audio/languages/{code}", response_model=LanguageEntry)
def describe_language(code: str) -> LanguageEntry:
    """Resolve one code and report which backends can speak it."""
    return LanguageEntry.model_validate(language_registry.resolve(code).to_dict())


@app.get("/api/v1/audio/emotions", response_model=EmotionPresetsResponse)
def list_emotions() -> EmotionPresetsResponse:
    """The emotion prosody presets and the prosody each one applies."""
    return EmotionPresetsResponse(
        presets=[EmotionPresetEntry.model_validate(p) for p in preset_catalogue()],
    )


@app.post(
    "/api/v1/audio/quality-audit",
    response_model=QualityAuditResponse,
    status_code=status.HTTP_200_OK,
)
def audit_quality(request: QualityAuditRequest) -> QualityAuditResponse:
    """Predict MOS, PESQ, STOI and SI-SDR for a generated clip (SQUIM)."""
    audio_path = resolve_audio_path(request.audio_path)
    reference = (
        resolve_audio_path(request.reference_path) if request.reference_path else None
    )
    try:
        report = quality_auditor.audit(audio_path, reference_path=reference)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality audit failed: {err}",
        ) from err
    return QualityAuditResponse(audioPath=str(audio_path), report=report.to_dict())


@app.post(
    "/api/v1/audio/voice-similarity",
    response_model=VoiceSimilarityResponse,
    status_code=status.HTTP_200_OK,
)
def voice_similarity(request: VoiceSimilarityRequest) -> VoiceSimilarityResponse:
    """
    Speaker-similarity score between a cloning reference and its output.

    The response names the scoring method: only an ``ecapa-tdnn`` result counts
    as evidence for the assignment's >85% cloning-similarity threshold.
    """
    reference = resolve_audio_path(request.reference_path)
    generated = resolve_audio_path(request.generated_path)
    try:
        report = quality_auditor.speaker_similarity(reference, generated)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity scoring failed: {err}",
        ) from err
    return VoiceSimilarityResponse(report=report.to_dict())
