import asyncio
import logging
import os
import threading
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
    LiveControlMessage,
    LiveSayMessage,
    LiveStartMessage,
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
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, TypeAdapter, ValidationError

import audit_log
import avatar_generator
import live_engine
import manifest
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
from job_store import JobStore
from model_registry import (
    audit_summary,
    audit_vision_weights,
    log_vision_audit,
    log_weight_audit,
    vision_audit_summary,
)
from quality_auditor import SpeechQualityAuditor
from request_context import (
    REQUEST_ID_HEADER,
    bind_request_id,
    configure_log_output,
    current_request_id,
    install_logging,
    new_request_id,
    request_id_middleware,
    unbind_request_id,
)
from security import API_KEY_HEADER, SecurityGate
from voice_engine import CLONE_ENGINES, ModelWeightsMissing, VoiceConsentRequired

install_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Audit model weights before serving a single request.

    Printed as well as logged: the router's fallbacks are silent by design, so
    a missing checkpoint has to be impossible to miss in the server output.
    """
    configure_log_output()
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
    # Without this a browser page cannot read the id off the response.
    expose_headers=[REQUEST_ID_HEADER],
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

# Registered last, so it is the outermost layer: even a 401 or 429 from the
# security gate carries a request id.
app.middleware("http")(request_id_middleware)

inputs_dir = project_root / "inputs"
inputs_dir.mkdir(parents=True, exist_ok=True)

outputs_dir = project_root / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

queue_backend = os.getenv("QUEUE_BACKEND", "in_memory").lower()
job_queue = (
    CeleryJobQueue()
    if queue_backend == "celery"
    # Celery keeps its state in Redis; the in-process queue keeps it in SQLite
    # (outputs/jobs.sqlite, or JOBS_DB) so a restart does not forget jobs.
    else InMemoryJobQueue(runner=run_render, store=JobStore())
)
faces = avatar_store.AvatarStore()
# Avatar generation runs in this process on one worker thread (generation_jobs).
generation_jobs = GenerationJobs(JobStore())

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

    Relative paths are looked up under the project root, ``outputs/`` and
    ``inputs/`` (so ``outputs/speech.wav`` and ``speech.wav`` both work), but
    whatever they resolve to must lie inside ``inputs/`` or ``outputs/``. The
    project root itself is NOT an allowed location: it holds ``.env``, the
    source and the docs, and naming one of those would at best turn the audit
    endpoints into a file-existence probe. ``../../etc/passwd`` is rejected the
    same way.
    """
    candidate = Path(raw)
    lookup_roots = (project_root, outputs_dir, inputs_dir)
    roots = (outputs_dir, inputs_dir)

    def allowed(path: Path) -> bool:
        try:
            return any(root.resolve() in path.resolve().parents for root in roots)
        except (ValueError, OSError):  # e.g. an embedded NUL byte
            return False

    if not candidate.is_absolute():
        # Only a lookup that lands inside inputs/ or outputs/ counts. A name
        # that exists elsewhere in the project (".env") is skipped, so it gets
        # the same "not found" as a name that exists nowhere.
        for root in lookup_roots:
            option = root / raw
            if allowed(option) and option.exists():
                candidate = option
                break
        else:
            candidate = outputs_dir / raw

    resolved = candidate.resolve()
    if not any(root.resolve() in resolved.parents for root in roots):
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


def resolve_voice_reference(raw: str) -> Path:
    """
    A cloning reference named by the caller: a file inside ``inputs/``, or a 400.

    "Outside inputs/" and "does not exist" get the *same* answer. Before this,
    ``speakerWav=/etc/passwd`` answered 403 (it exists, it has no provenance
    record) while a missing path answered 400, which let any caller test which
    files exist anywhere on the host.
    """
    refusal = HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="speakerWav must be a recording inside the inputs/ folder. Pick one from GET /api/v1/audio/samples.",
    )
    candidate = Path(raw)
    try:
        resolved = (candidate if candidate.is_absolute() else inputs_dir / raw).resolve()
    except (ValueError, OSError) as err:  # e.g. an embedded NUL byte
        raise refusal from err
    if inputs_dir.resolve() not in resolved.parents or not resolved.is_file():
        raise refusal
    return resolved


def _scrub(text: str) -> str:
    """Take server file-system locations out of an error message before a client sees it."""
    import tempfile

    for location, label in ((str(project_root), "<project>"), (str(Path.home()), "~"), (tempfile.gettempdir(), "<tmp>")):
        text = text.replace(location, label)
    return text


def _audio_failure(what: str, err: Exception) -> HTTPException:
    """
    The response for an audio operation that raised.

    A file that is not audio is the caller's problem (422); anything else is
    ours (500), logged in full under the request id. Neither echoes a server path.
    """
    import soundfile as sf

    if isinstance(err, sf.LibsndfileError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_scrub(f"{what}: the file is not audio this server can read ({err})"),
        )
    logger.exception("%s failed", what)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=_scrub(f"{what} failed: {err}"),
    )


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
        # Quality-gating a photo runs landmark detection: off the event loop.
        record, report = await run_in_threadpool(
            faces.register,
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
    audit_log.shared_audit().record(
        "face_registered", subject=avatar_id, basis=consent_basis or None, source=provenance.HUMAN,
        image_sha256=manifest.sha256_file(record.path),
    )
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
        audit_log.shared_audit().record(
            "face_generated", subject=request.avatar_id, basis="synthetic", seed=generated.seed,
            generator=generated.model, image_sha256=manifest.sha256_file(record.path),
        )
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


@app.get("/api/v1/audit")
def get_audit_trail(event: Optional[str] = None, subject: Optional[str] = None, since: Optional[str] = None, limit: int = 100) -> dict:
    """
    The consent audit trail, newest first: every use of a voice or a face, and every refusal, with the
    basis it happened under. Filter by ``event``, ``subject`` (an avatar id or a recording's hash) and
    ``since`` (ISO time). Entries hold ids and hashes, never names.
    """
    if event and event not in audit_log.EVENTS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"unknown event {event!r}; one of {list(audit_log.EVENTS)}")
    trail = audit_log.shared_audit()
    return {"entries": trail.query(event=event, subject=subject, since=since, limit=limit), "head": trail.head(), "events": list(audit_log.EVENTS)}


@app.get("/api/v1/audit/verify")
def verify_audit_trail() -> dict:
    """Recompute the hash chain: is the trail intact? Reports the first entry where it breaks."""
    return audit_log.shared_audit().verify_chain()


MAX_VERIFY_BYTES = 200 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024


async def _spool_upload(upload: UploadFile, limit: int, suffix: str = "") -> Path:
    """Stream an upload to a temp file outside ``outputs/`` (which is served publicly), refusing one over ``limit``."""
    import tempfile

    handle = tempfile.NamedTemporaryFile(prefix="verify-", suffix=suffix, delete=False)
    total = 0
    try:
        while chunk := await upload.read(1 << 20):
            total += len(chunk)
            if total > limit:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"the file is larger than the {limit // (1024 * 1024)} MB limit",
                )
            handle.write(chunk)
    except BaseException:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)
        raise
    handle.close()
    return Path(handle.name)


@app.post("/api/v1/provenance/verify")
async def verify_provenance(
    file: Optional[UploadFile] = File(None),
    manifest_file: Optional[UploadFile] = File(None, alias="manifest"),
    path: Optional[str] = Form(None),
) -> dict:
    """
    Is this audio or video file one of ours?

    Send the file (``file``) or name one under ``outputs/`` or ``inputs/`` (``path``), and optionally its
    manifest (``manifest``; ``<video>.manifest.json`` beside a named file is used automatically). The
    answer reports each kind of evidence separately (audio watermark, video watermark, signed manifest,
    audit record) and a verdict with its meaning. A file with no marks gets ``no_evidence``, which
    explicitly does not mean the content is real.
    """
    import json

    import authenticity

    if (file is None) == (path is None):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="send exactly one of: a 'file' upload, or a 'path' under outputs/")
    document = None
    if manifest_file is not None:
        raw = await manifest_file.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="the manifest is larger than 1 MB")
        try:
            document = json.loads(raw)
        except ValueError as err:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="the manifest is not valid JSON") from err
    spooled: Optional[Path] = None
    try:
        if file is not None:
            spooled = await _spool_upload(file, MAX_VERIFY_BYTES, suffix=Path(file.filename or "").suffix[:12])
            target = spooled
        else:
            target = resolve_audio_path(path or "")
        try:
            shown = Path((file.filename or "") if file is not None else target.name).name[:120] or None
            return await run_in_threadpool(authenticity.verify_file, target, document, shown)
        except ValueError as err:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=_scrub(str(err))) from err
    finally:
        if spooled is not None:
            spooled.unlink(missing_ok=True)


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
    # Only the upload read is awaited here. Decoding and landmark detection are
    # CPU-bound and take seconds on a cold model; run directly in this ``async
    # def`` they froze the whole event loop, so every other request (even a
    # trivial language lookup) waited behind them. They run on a worker thread.
    data = await _read_upload(file) if file is not None else None

    def analyse():
        if data is not None:
            image, notices = avatar_store.decode_image(data)
        else:
            # The exactly-one-of check above guarantees this; the assert states
            # it for the type checker, which cannot follow an XOR.
            assert avatar_id is not None
            image, notices = avatar_store.decode_image(_avatar_or_http(avatar_id).path)
        with FACE_ENGINE_LOCK:
            return shared_face_engine().check_quality(image), notices

    try:
        report, notices = await run_in_threadpool(analyse)
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
        raise _audio_failure("Alignment", err) from err


# ---------------------------------------------------------------------------
# Live interactive avatar: WS /api/v1/live (R-18, R-19)
# ---------------------------------------------------------------------------
# Each session holds a face animator and keeps a sentence pipeline busy, so the
# number running at once is capped; a client over the cap is told to retry.
LIVE_MAX_SESSIONS = int(os.getenv("LIVE_MAX_SESSIONS", "2"))
LIVE_START_TIMEOUT_S = 15.0
LIVE_IDLE_TIMEOUT_S = 600.0
_live_active = 0
_live_lock = threading.Lock()
_live_message = TypeAdapter(LiveSayMessage | LiveControlMessage)


async def _live_fail(ws: WebSocket, code: str, detail: str, close: int = 1008) -> None:
    """Tell the client what went wrong (so it can fix it), then close."""
    try:
        await ws.send_json({"type": "error", "code": code, "detail": _scrub(detail)})
        await ws.close(code=close)
    except (RuntimeError, WebSocketDisconnect):
        pass  # the client already left


@app.websocket("/api/v1/live")
async def live_avatar(ws: WebSocket) -> None:
    """
    A live avatar session. The client sends one ``start`` message, then any number
    of ``say`` (speak this text), ``interrupt`` (stop speaking, keep the session)
    and finally ``stop``. The server answers ``ready``, then per sentence an
    ``audio`` message followed by its PCM16 chunk and the avatar's JPEG frames as
    binary messages (framing in ``live_engine``), a ``chunk`` message with that
    sentence's timings, and ``done`` when the text is finished. Protocol errors are
    ``error`` messages; the ones that cannot be recovered also close the socket.

    HTTP middleware does not run for WebSockets, so the key check, the consent
    checks, the request id and the session cap all happen here.
    """
    global _live_active
    await ws.accept()
    token = bind_request_id(new_request_id(ws.headers.get(REQUEST_ID_HEADER)))
    reserved = False
    session = None
    receiver = None
    try:
        try:
            raw = await asyncio.wait_for(ws.receive_json(), LIVE_START_TIMEOUT_S)
            start = LiveStartMessage.model_validate(raw)
        except asyncio.TimeoutError:
            return await _live_fail(ws, "start_timeout", f"send a start message within {LIVE_START_TIMEOUT_S:.0f} s", 1008)
        except (ValidationError, ValueError, TypeError) as err:
            return await _live_fail(ws, "bad_start", f"the first message must be a valid start message: {err}", 1003)
        except WebSocketDisconnect:
            return

        allowed, _status_code, detail = security_gate.inspect(
            path="/api/v1/live",
            api_key=start.api_key or ws.headers.get(API_KEY_HEADER),
            client_host=ws.client.host if ws.client else None,
        )
        if not allowed:
            return await _live_fail(ws, "unauthorised", str(detail.get("detail", "request rejected")))

        with _live_lock:
            if _live_active >= LIVE_MAX_SESSIONS:
                full = True
            else:
                _live_active += 1
                reserved, full = True, False
        if full:
            return await _live_fail(
                ws, "busy", f"{LIVE_MAX_SESSIONS} live sessions are already running; try again shortly", 1013
            )

        # The same checks the REST route makes before queueing: weights, consent, language.
        try:
            speaker = None
            if start.mode == "clone":
                if not start.speaker_wav:
                    raise ValueError("clone mode needs speakerWav: a recording inside inputs/ (GET /api/v1/audio/samples)")
                speaker = str(resolve_voice_reference(start.speaker_wav))
            router = get_router()
            model_key = router.select_model(
                mode=start.mode, language=start.language, quality="balanced", style=None,
                text="", clone_engine=start.clone_engine,
            )
            router.preflight(model_key, speaker, start.language)
            session = live_engine.LiveSession(
                router, faces, start.avatar_id, language=start.language, mode=start.mode,
                emotion=start.emotion, emotion_intensity=start.emotion_intensity, fps=start.fps,
                max_side=start.max_side, speaker_wav=speaker, clone_engine=start.clone_engine,
            )
            info = await run_in_threadpool(session.open)
        except HTTPException as err:
            return await _live_fail(ws, "bad_request", str(err.detail))
        except VoiceConsentRequired as err:
            return await _live_fail(ws, "consent", str(err))
        except avatar_store.AvatarConsentError as err:
            return await _live_fail(ws, "consent", str(err))
        except avatar_store.AvatarNotFound as err:
            return await _live_fail(ws, "avatar_not_found", str(err))
        except ModelWeightsMissing as err:
            return await _live_fail(ws, "model_unavailable", str(err), 1011)
        except (ValueError, avatar_store.AvatarError, FaceEngineUnavailable, live_engine.LiveError) as err:
            return await _live_fail(ws, "bad_request", str(err))

        await ws.send_json({"type": "ready", "sessionId": session.session_id, "avatarId": start.avatar_id,
                            "mode": start.mode, "model": model_key, "sampleRate": 24000,
                            "requestId": current_request_id(), **info})
        logger.info("Live session %s opened: avatar=%s mode=%s %dx%d@%d", session.session_id,
                    start.avatar_id, start.mode, info["width"], info["height"], info["fps"])

        # A receiver task keeps reading while speech is being sent, so an
        # `interrupt` is seen between frames instead of after the whole text.
        inbox: asyncio.Queue = asyncio.Queue()

        async def receive() -> None:
            while True:
                try:
                    inbox.put_nowait(await ws.receive_json())
                except (ValueError, TypeError):  # not JSON: tell the client, keep listening
                    inbox.put_nowait({"type": "_malformed"})
                except (WebSocketDisconnect, RuntimeError):  # the client left
                    inbox.put_nowait({"type": "_closed"})
                    return

        receiver = asyncio.create_task(receive())
        queued: list = []

        async def next_message() -> dict:
            if queued:
                return queued.pop(0)
            return await asyncio.wait_for(inbox.get(), LIVE_IDLE_TIMEOUT_S)

        while True:
            try:
                raw = await next_message()
            except asyncio.TimeoutError:
                return await _live_fail(ws, "idle", "no message for too long; the session was closed", 1000)
            kind = raw.get("type") if isinstance(raw, dict) else None
            if kind in ("_closed", "stop"):
                break
            if kind == "_malformed":
                await ws.send_json({"type": "error", "code": "malformed", "detail": "messages must be JSON objects"})
                continue
            try:
                message = _live_message.validate_python(raw)
            except ValidationError as err:
                await ws.send_json({"type": "error", "code": "bad_message", "detail": _scrub(str(err))[:300]})
                continue
            if not isinstance(message, LiveSayMessage):  # an interrupt with nothing to interrupt
                await ws.send_json({"type": "interrupted"})
                continue

            generator = live_engine.stream_text(session, message.text)
            stopped = interrupted = False
            try:
                async for event in generator:
                    # Look for control messages without losing queued `say`s.
                    while not inbox.empty():
                        waiting = inbox.get_nowait()
                        waiting_kind = waiting.get("type") if isinstance(waiting, dict) else None
                        if waiting_kind in ("_closed", "stop"):
                            stopped = True
                        elif waiting_kind == "interrupt":
                            interrupted = True
                        else:
                            queued.append(waiting)
                    if stopped or interrupted:
                        break
                    if event.kind == "frame":
                        await ws.send_bytes(event.payload)
                    elif event.kind == "audio":
                        await ws.send_json({"type": "audio", **event.meta})
                        await ws.send_bytes(event.payload)
                    else:
                        await ws.send_json({"type": event.kind, **event.meta})
                        if event.kind == "chunk":
                            logger.info("Live chunk %s: %s", event.meta["chunk"], event.meta)
            except live_engine.LiveError as err:
                await ws.send_json({"type": "error", "code": "speech_failed", "detail": str(err)})
            except WebSocketDisconnect:
                stopped = True
            except Exception as err:  # noqa: BLE001 - one bad sentence must not kill the session silently
                logger.exception("Live speech failed")
                await ws.send_json({"type": "error", "code": "speech_failed", "detail": _scrub(f"{type(err).__name__}: {err}")})
            finally:
                await generator.aclose()
            if stopped:
                break
            if interrupted:
                await ws.send_json({"type": "interrupted"})
    except WebSocketDisconnect:
        pass
    finally:
        if receiver is not None:
            receiver.cancel()
        if session is not None:
            session.cleanup()
            logger.info("Live session %s closed", session.session_id)
        if reserved:
            with _live_lock:
                _live_active -= 1
        unbind_request_id(token)
        try:
            await ws.close()
        except (RuntimeError, WebSocketDisconnect):
            pass


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
        watermark=result.get("watermark"),
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
    if request.speaker_wav:
        # Resolved (and confined to inputs/) before anything else looks at it.
        request = request.model_copy(update={"speaker_wav": str(resolve_voice_reference(request.speaker_wav))})
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
        raise _audio_failure("Quality audit", err) from err
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
        raise _audio_failure("Similarity scoring", err) from err
    return VoiceSimilarityResponse(report=report.to_dict())
