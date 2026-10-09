"""
The HTTP and WebSocket API: the one process the browser studio talks to.

Where it sits. Every request from the studio lands here first. This module
checks the request (key, rate limit, consent, file location), then hands the
real work to the module that owns it: speech to ``voice_engine`` through the
Celery task in ``celery_app``, renders to ``job_queue`` and ``render_engine``,
faces to ``avatar_store``, live sessions to ``live_engine``. It holds almost no
logic of its own; it translates between HTTP and those modules, and turns
their exceptions into status codes a client can act on.

Sections, in file order: start-up audit and middleware, health, path
resolution, voice samples, avatar faces and generation, the consent audit
trail, provenance checks, render jobs and batches, voice-to-avatar, protected
voices, parameters and metrics, alignment, the live WebSocket, speech
synthesis, the language / emotion catalogues and quality scoring, and last the
built studio served as static files.

Concepts used throughout, explained once here:

**FastAPI routes.** ``@app.get("/path")`` (or ``post``, ``delete``,
``websocket``) registers the function under it as the handler for that method
and path. FastAPI reads the function's parameters to decide where each value
comes from: a name in ``{braces}`` in the path is a path parameter, a Pydantic
model is the JSON body, ``File(...)`` / ``Form(...)`` are multipart form
fields, and any other simple type is a query-string parameter.

**Response models and status codes.** ``response_model=X`` makes FastAPI
validate the return value against ``X`` and use it for the ``/docs`` schema.
``status_code=`` sets the code a successful call returns: 201 Created for a
new resource, 202 Accepted for work queued to finish later, 204 No Content
for a delete. ``response_model_exclude_none=True`` drops ``None`` fields so
the client does not see keys that do not apply yet.

**HTTPException.** Raising ``HTTPException(status_code, detail)`` anywhere in
a handler ends the request with that code and ``{"detail": ...}`` as the
body. The codes used here: 400 the request is wrong, 403 consent is missing,
404 the thing does not exist, 409 it conflicts with what exists, 413 too
large, 422 well-formed but unusable, 503 a model or service is not available.

**def versus async def.** FastAPI runs a plain ``def`` handler on a worker
thread, so it may block. An ``async def`` handler runs on the event loop, the
single thread that serves every connection in turn; it must not block, or
every other request waits. Blocking work inside an ``async def`` is moved off
the loop with ``run_in_threadpool`` (see ``register_avatar_face``).
"""

import asyncio
import json
import logging
import os
import uuid
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# backend/app.py -> parents[1] is the repository root, which holds .env,
# inputs/ and outputs/.
project_root = Path(__file__).resolve().parents[1]
dotenv_path = project_root / ".env"
# Loaded before the project imports below on purpose: some of those modules
# read settings when they are imported (celery_app reads QUEUE_BACKEND to pick
# eager or broker mode), so the values must already be in os.environ.
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    # No .env at the root: python-dotenv searches upward from the working
    # directory instead.
    load_dotenv()

# Every request and response shape is a Pydantic model in contracts.py; this
# module only imports them, so the wire format is defined in one place.
from celery_app import celery, get_router, run_render, synthesize_audio
from contracts import (
    AudioSynthesisRequest,
    AvatarFaceEntry,
    AvatarFacesResponse,
    AvatarGenerateRequest,
    AvatarGenerateResponse,
    AvatarRegisterResponse,
    LiveAudioStartMessage,
    LiveControlMessage,
    LiveSayMessage,
    LiveStartMessage,
    FaceAnalysisResponse,
    LipSyncScoreResponse,
    AvatarRenderJob,
    AvatarStylizeRequest,
    RenderBatchRequest,
    RenderQuality,
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
import voice_engine as voice_engine_module
import metrics
import parameters
import protected_voices
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

# Installs the log-record factory that stamps every log line with the current
# request id (request_context explains how). Done before the first logger use.
install_logging()
logger = logging.getLogger(__name__)


# A *lifespan* is FastAPI's start-up / shut-down hook. The code before
# ``yield`` runs once when the server starts, before the first request; code
# after it would run at shut-down. @asynccontextmanager turns the generator
# into the context-manager object FastAPI expects.
@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Audit model weights before serving a single request.

    Printed as well as logged: the router's fallbacks are silent by design, so
    a missing checkpoint has to be impossible to miss in the server output.

    Two audits are printed: the speech models (``log_weight_audit``) and the
    vision weights (``log_vision_audit``). Nothing is loaded here; the audits
    only check which files are on disk, so start-up stays fast.
    """
    # Log output is attached here, at start-up, not on import: a test that only
    # imports the app must stay quiet.
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
    # The server serves requests while suspended here; nothing to release at shut-down.
    yield


# The application object. Every route, middleware and mount below is attached
# to it, and uvicorn serves it as ``app:app`` (module ``app``, attribute ``app``).
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
# *Middleware* is code that wraps every HTTP request: it sees the request
# before any route and the response after it. CORS (cross-origin resource
# sharing) is the browser rule that a page from one origin (scheme, host and
# port) may only read responses from another origin if that server says so in
# Access-Control-* headers. The studio dev server (port 5173) and the API
# (port 8000) are different origins, so this middleware adds those headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or [],
    # The regex is only used when no explicit list is configured: then any
    # http://localhost or http://127.0.0.1 port is accepted, nothing else.
    allow_origin_regex=None if _cors_origins else r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    # Lets the browser send cookies or auth headers on cross-origin calls.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Without this a browser page cannot read the id off the response.
    expose_headers=[REQUEST_ID_HEADER],
)

# API-key auth + per-identity rate limiting. Both are configured from .env and
# auth is off by default so local development needs no key.
security_gate = SecurityGate()


# ``@app.middleware("http")`` registers a function middleware. It receives the
# request and ``call_next``; awaiting ``call_next(request)`` runs everything
# inside it (inner middleware, then the route) and returns the response.
# Returning a response *without* calling it stops the request here.
@app.middleware("http")
async def enforce_security(request: Request, call_next):
    """
    Refuse a request that fails the API-key check or the rate limit.

    ``SecurityGate.inspect`` decides and returns ``(allowed, status, detail)``
    rather than raising, so this function builds the 401 or 429 response
    itself. Public paths (``/health``, ``/docs``, ``/outputs/...``) are let
    through by the gate. WebSockets never pass through here; the live route
    calls the same gate itself.
    """
    # CORS preflight carries no headers to authenticate; let the CORS
    # middleware answer it.
    if request.method == "OPTIONS":
        return await call_next(request)

    # request.client is None for some test clients and unix sockets, hence the guard.
    allowed, status_code, detail = security_gate.inspect(
        path=request.url.path,
        api_key=request.headers.get(API_KEY_HEADER),
        client_host=request.client.host if request.client else None,
    )
    if not allowed:
        # Built by hand rather than raised: an HTTPException raised in a
        # middleware is not turned into a response by FastAPI's handlers.
        # The headers carry WWW-Authenticate (401) or Retry-After (429).
        return JSONResponse(
            status_code=status_code,
            content={"detail": detail.get("detail", "request rejected")},
            headers=detail.get("headers", {}),
        )
    return await call_next(request)

# Registered last, so it is the outermost layer: even a 401 or 429 from the
# security gate carries a request id.
# (Same decorator as above, called as a plain function because the middleware
# is defined in request_context.) Each add wraps the previous ones, so the
# order outside-in is: request id, security gate, CORS, then the route.
app.middleware("http")(request_id_middleware)

# inputs/ holds what the owner supplies (voice samples, face photos with their
# provenance sidecars); outputs/ holds what the server makes. Both are created
# on start so a fresh checkout runs without manual setup.
inputs_dir = project_root / "inputs"
inputs_dir.mkdir(parents=True, exist_ok=True)

outputs_dir = project_root / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
# ``mount`` attaches a whole sub-application at a path prefix. StaticFiles
# serves files from a folder, so a finished video at outputs/x.mp4 is fetched
# as GET /outputs/x.mp4. The security gate lists /outputs/ as public, which is
# why uploads under verification are spooled elsewhere (see _spool_upload).
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

# Which render queue to use: "in_memory" (default, one process, no Redis) or
# "celery" (jobs go to Redis and a separate worker).
queue_backend = os.getenv("QUEUE_BACKEND", "in_memory").lower()
job_queue = (
    CeleryJobQueue()
    if queue_backend == "celery"
    # Celery keeps its state in Redis; the in-process queue keeps it in SQLite
    # (outputs/jobs.sqlite, or JOBS_DB) so a restart does not forget jobs.
    else InMemoryJobQueue(runner=run_render, store=JobStore())
)
# The consent-enforcing face store over inputs/faces: every route that touches
# an avatar image goes through it, so a face without provenance is refused in
# one place.
faces = avatar_store.AvatarStore()
# Avatar generation runs in this process on one worker thread (generation_jobs).
generation_jobs = GenerationJobs(JobStore())

# One auditor for the process: SQUIM weights load once, on first audit.
quality_auditor = SpeechQualityAuditor()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class VoiceSampleInfo(BaseModel):
    """
    One recording in ``inputs/`` as the studio's voice picker shows it.

    Defined here rather than in ``contracts.py`` because only this listing
    route uses it; it never crosses the audio-to-vision boundary.
    ``ready_for_cloning`` is the ``audio_utils`` verdict on whether the clip is
    usable as a cloning reference.
    """

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
    """The body of ``GET /api/v1/audio/samples``: the recordings and the formats accepted."""

    samples: List[VoiceSampleInfo]
    supported_formats: List[str]
    inputs_dir: str


def _render_engines() -> dict:
    """Which render engines can run right now (by checkpoint presence).

    Returns ``{engine name: bool}``. The blendshape engine needs no extra
    weights, so it is always True; Wav2Lip is True only when its checkpoint is
    on disk. Reported by ``/health`` and the face listing so the studio can
    grey out an engine that would be refused.
    """
    # A function-level import: the module is loaded on the first call, not
    # when app.py is imported. Several routes below use the same pattern for
    # modules only they need.
    from wav2lip_engine import shared_wav2lip_engine

    return {
        render_engine.ENGINE_BLENDSHAPE: True,
        render_engine.ENGINE_WAV2LIP: shared_wav2lip_engine().available,
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
# The first route. ``@app.get("/health")`` makes ``health`` the handler for
# GET /health; a returned dict is sent as JSON with status 200.
@app.get("/health")
def health() -> dict:
    """
    What this server can do right now, for the studio, Docker and the doctor script.

    Always 200 with ``"status": "ok"`` while the process answers; the detail
    is in the other keys: the queue backend, the security settings, which
    models can be routed to, which weights are on disk, and which render
    engines can run. Golden rule 1 (no silent fallbacks) is met here: a
    missing model shows up in ``modelWeights`` instead of being hidden.
    The path is public, so it needs no API key.
    """
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
        # Which checkpoints are on disk, speech and vision, with a fetch hint for each missing one.
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

    Returns the resolved absolute path. Raises ``HTTPException`` 400 for a
    path outside both folders and 404 for a file that does not exist. Used
    by the alignment, quality-audit, similarity and provenance routes.
    """
    candidate = Path(raw)
    # Where a relative name is looked for, in order...
    lookup_roots = (project_root, outputs_dir, inputs_dir)
    # ...and where the result is allowed to end up.
    roots = (outputs_dir, inputs_dir)

    def allowed(path: Path) -> bool:
        """True when ``path``, with ``..`` and symlinks resolved, lies under inputs/ or outputs/."""
        # resolve() makes the path absolute and follows ``..`` and symlinks, so
        # "outputs/../.env" is seen for what it is. Checking ``parents`` (every
        # ancestor folder) is the containment test.
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
        # for ... else: the else branch runs only when the loop ended without
        # ``break``, i.e. no root had the file. outputs/ is then assumed, so the
        # 404 below names a sensible location.
        else:
            candidate = outputs_dir / raw

    # The final check applies to absolute paths too, which skipped the lookup.
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

    Returns the resolved path of an existing file. Consent itself (the
    provenance sidecar) is checked later by the voice router's preflight;
    this function only confines the location.
    """
    # Built once and raised from every refusal branch, so all of them are
    # byte-for-byte the same answer.
    refusal = HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="speakerWav must be a recording inside the inputs/ folder. Pick one from GET /api/v1/audio/samples.",
    )
    candidate = Path(raw)
    try:
        # A relative name is taken as relative to inputs/, never the working directory.
        resolved = (candidate if candidate.is_absolute() else inputs_dir / raw).resolve()
    except (ValueError, OSError) as err:  # e.g. an embedded NUL byte
        # ``raise ... from err`` chains the original error into the traceback
        # for the logs, while the client still sees only the refusal.
        raise refusal from err
    if inputs_dir.resolve() not in resolved.parents or not resolved.is_file():
        raise refusal
    return resolved


def _scrub(text: str) -> str:
    """Take server file-system locations out of an error message before a client sees it.

    Exceptions from libraries often quote full paths, which would tell a
    caller the server's user name and folder layout. Each known prefix is
    swapped for a short label (``<project>``, ``~``, ``<tmp>``). Only the copy
    sent to the client is scrubbed; callers that log, log the original.
    """
    import tempfile

    # The project root is replaced first: it usually sits inside the home
    # folder, and replacing home first would leave it as "~/..." instead.
    for location, label in ((str(project_root), "<project>"), (str(Path.home()), "~"), (tempfile.gettempdir(), "<tmp>")):
        text = text.replace(location, label)
    return text


def _audio_failure(what: str, err: Exception) -> HTTPException:
    """
    The response for an audio operation that raised.

    A file that is not audio is the caller's problem (422); anything else is
    ours (500), logged in full under the request id. Neither echoes a server path.

    It *returns* the exception rather than raising it, so the caller writes
    ``raise _audio_failure(...) from err`` and the traceback keeps the cause.
    """
    import soundfile as sf

    # libsndfile is the C library soundfile wraps; it raises this when the
    # bytes are not a format it can decode.
    if isinstance(err, sf.LibsndfileError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_scrub(f"{what}: the file is not audio this server can read ({err})"),
        )
    # logger.exception logs at ERROR level and appends the current traceback.
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

    ``response_model=VoiceSamplesResponse`` in the decorator makes FastAPI
    check the returned object and publish its schema in ``/docs``. Each file
    is probed by ``audio_utils.probe_audio``, which reads the header where
    soundfile can; a file that cannot be read is skipped, not fatal.
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
    """A face-quality report as the JSON dict the API returns (one place to change the shape)."""
    return report.to_dict()


async def _read_upload(file: UploadFile) -> bytes:
    """Read an upload, refusing one over the limit without buffering all of it.

    ``UploadFile`` is FastAPI's handle on one file in a multipart form. Its
    ``read`` is a coroutine (it may wait on the network), so it is awaited.
    Returns the bytes; raises 413 (Request Entity Too Large) past
    ``avatar_store.MAX_UPLOAD_BYTES`` (15 MB).
    """
    limit = avatar_store.MAX_UPLOAD_BYTES
    # Ask for one byte more than allowed: getting it back proves the file is
    # too large without reading the rest of it into memory.
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"image is larger than the {limit // (1024 * 1024)} MB limit",
        )
    return data


@app.get("/api/v1/avatar/faces", response_model=AvatarFacesResponse)
def list_avatar_faces() -> AvatarFacesResponse:
    """Every image in ``inputs/faces`` and whether its provenance permits use.

    Faces without consent are listed too (marked unusable), so the owner can
    see why one is refused. ``consentBases`` lists the bases a registration
    may claim; ``renderEngines`` says which engines can run.
    """
    # model_validate builds the Pydantic model from a plain dict, checking it.
    return AvatarFacesResponse(
        avatars=[AvatarFaceEntry.model_validate(r.to_dict()) for r in faces.list()],
        consentBases=sorted(provenance.FACE_CONSENT_BASES),
        renderEngines=_render_engines(),
    )


# 201 Created: a new resource (the avatar) exists after this call.
@app.post(
    "/api/v1/avatar/faces",
    response_model=AvatarRegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_avatar_face(
    # A *multipart* request carries files and form fields in one body, which
    # is how a browser form uploads a photo. ``File(...)`` and ``Form(...)``
    # tell FastAPI to read these parameters from that body; ``...`` means
    # required, and ``alias`` is the field name on the wire (camelCase).
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

    Returns the stored entry and its quality report. Raises 422 when the photo
    fails the quality gate (the report says why), 400 for a bad or taken
    id, a missing consent basis or a missing subject, 413 for an oversized file, and 503 when the face
    landmarker bundle is not on disk.
    """
    data = await _read_upload(file)
    try:
        # Quality-gating a photo runs landmark detection: off the event loop.
        # run_in_threadpool(fn, *args, **kwargs) runs fn on a worker thread and
        # lets this coroutine await the result, so the loop keeps serving other
        # requests in the meantime. This route is ``async def`` because the
        # upload read above is a coroutine; the slow part is pushed out here.
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
    # AvatarRejected is caught before AvatarError: the more specific
    # exception goes first. ``detail`` may be a dict as well as a string; here
    # it carries the quality report so the studio can show what failed.
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
    # The consent audit trail (audit_log): one append-only, hash-chained row
    # per use or refusal of a face or voice. It stores ids, the consent basis
    # and a SHA-256 of the image, never a name, so a reviewer can later prove
    # which file was registered on which basis.
    audit_log.shared_audit().record(
        "face_registered", subject=avatar_id, basis=consent_basis or None, source=provenance.HUMAN,
        image_sha256=manifest.sha256_file(record.path),
    )
    return AvatarRegisterResponse(
        avatar=AvatarFaceEntry.model_validate(record.to_dict()),
        quality=_quality_payload(report) if report is not None else None,
    )


def _diffusion_weights() -> dict:
    """Whether the Stable Diffusion weights are on disk, as ``{present, detail}``.

    Shared by the generate and stylize routes (both use the same diffusion
    model), so each can refuse with a 503 before queueing doomed work.
    """
    # ``status_`` with a trailing underscore so the name does not shadow the
    # imported ``status`` module. next() returns the first match from the
    # generator; the audit always contains this key.
    status_ = next(s for s in audit_vision_weights() if s.key == "avatar-diffusion")
    return {"present": status_.present, "detail": status_.detail}


@app.get("/api/v1/avatar/generate/options")
def avatar_generate_options() -> dict:
    """The attribute choices ``POST /avatar/generate`` accepts, and whether it can run.

    The studio builds its generate form from this, so the choices live in one
    place (``avatar_generator``). ``available`` is False with a ``detail``
    naming the fetch step when the weights are missing.
    """
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

    202 Accepted: the job is queued, not done. The response carries a
    ``taskId`` to poll with ``GET /api/v1/avatar/generate/{task_id}``.
    Diffusion takes seconds on a GPU and minutes on a CPU, far longer than an
    HTTP request should wait.
    """
    weights = _diffusion_weights()
    if not weights.get("present"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=weights.get("detail", "avatar-diffusion weights missing"))
    if not request.overwrite:
        # try / except / else: the else branch runs only when no exception
        # was raised, i.e. the id was found, which is the conflict.
        try:
            faces.get(request.avatar_id)
        except avatar_store.AvatarNotFound:
            pass
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"avatar {request.avatar_id!r} already exists; choose another id or set overwrite",
            )

    # A *closure*: ``work`` is defined inside the route, so it can use
    # ``request`` later, after this function has returned. generation_jobs
    # runs it on its own worker thread.
    def work() -> dict:
        """Generate, quality-check and register the face; return the job's result dict."""
        from face_engine import FACE_ENGINE_LOCK, shared_face_engine

        def check(image):
            """Quality-gate one candidate image with the shared face engine."""
            # The MediaPipe landmarker is one shared native object that is not
            # documented as thread-safe; the lock keeps every other thread
            # (request handlers, the render worker) off it while this runs.
            with FACE_ENGINE_LOCK:
                return shared_face_engine().check_quality(image)

        # Tries seeds seed, seed+1, ... up to ``attempts`` of them; a candidate
        # that fails the quality gate is recorded in rejected_seeds with its
        # reason and the next seed is tried.
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
        # Recorded with basis "synthetic": the face depicts nobody, and the
        # seed and generator name are kept so it can be reproduced.
        audit_log.shared_audit().record(
            "face_generated", subject=request.avatar_id, basis="synthetic", seed=generated.seed,
            generator=generated.model, image_sha256=manifest.sha256_file(record.path),
        )
        return {
            "avatarId": request.avatar_id,
            "imageUrl": record.to_dict()["imageUrl"],
            "seed": generated.seed,
            # JSON object keys must be strings, so the integer seeds are converted.
            "rejectedSeeds": {str(k): v for k, v in generated.rejected_seeds.items()},
            "provenance": record.provenance,
        }

    # submit() returns at once with an id; the work runs on generation_jobs'
    # single worker thread (one heavy model at a time, golden rule 5).
    return AvatarGenerateResponse(taskId=generation_jobs.submit(work), status="QUEUED")


@app.get("/api/v1/avatar/styles")
def avatar_styles() -> dict:
    """The styles ``POST /avatar/stylize`` accepts, with the strength each uses, and whether it can run.

    ``strength`` is the img2img denoising strength: how far the result may
    move away from the source photo (0 keeps it, 1 ignores it).
    """
    import style_transfer

    weights = _diffusion_weights()
    # A dict comprehension: one entry per style, built in a single expression.
    return {"styles": {name: {"prompt": s.prompt, "strength": s.strength} for name, s in style_transfer.STYLES.items()},
            "available": bool(weights.get("present")), "detail": weights.get("detail", "")}


@app.post(
    "/api/v1/avatar/stylize",
    response_model=AvatarGenerateResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_202_ACCEPTED,
)
def stylize_avatar(request: AvatarStylizeRequest) -> AvatarGenerateResponse:
    """
    Queue a restyled copy of an avatar (Stable Diffusion img2img). The copy inherits the source's
    provenance and records what it came from; the result reports SFace identity similarity to the
    source. Refused up front: missing weights (503), unknown source (404), a source without consent
    (403), a taken id (409).

    Every check runs before anything is queued. Poll the returned ``taskId`` like a generation task.
    """
    import style_transfer

    weights = _diffusion_weights()
    if not weights.get("present"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=weights.get("detail", "avatar-diffusion weights missing"))
    # The source must be usable (registered with consent): a restyled copy of
    # a face is still that person's face.
    try:
        faces.require_usable(request.avatar_id)
    except avatar_store.AvatarNotFound as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err)) from err
    except avatar_store.AvatarConsentError as err:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(err)) from err
    try:
        faces.get(request.new_avatar_id)
    except avatar_store.AvatarNotFound:
        pass
    else:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"avatar {request.new_avatar_id!r} already exists; choose another id")

    def work() -> dict:
        """Restyle, quality-check and register the copy; return the job's result dict."""
        from face_engine import FACE_ENGINE_LOCK, shared_face_engine

        def check(image):
            """Quality-gate the restyled image with the shared face engine (same lock as above)."""
            with FACE_ENGINE_LOCK:
                return shared_face_engine().check_quality(image)

        result = style_transfer.stylize_registered_avatar(
            faces, check, request.avatar_id, request.style, request.new_avatar_id, seed=request.seed, steps=request.steps)
        record = faces.get(request.new_avatar_id)
        # The copy inherits the source's basis, and derived_from links the two
        # rows in the trail.
        audit_log.shared_audit().record(
            "face_generated", subject=request.new_avatar_id, basis=str(record.provenance.get("consentBasis") or record.provenance.get("source") or ""),
            derived_from=request.avatar_id, style=request.style, seed=request.seed, image_sha256=manifest.sha256_file(record.path),
        )
        return {**result, "imageUrl": record.to_dict()["imageUrl"], "provenance": record.provenance}

    return AvatarGenerateResponse(taskId=generation_jobs.submit(work), status="QUEUED")


@app.get(
    "/api/v1/avatar/generate/{task_id}",
    response_model=AvatarGenerateResponse,
    response_model_exclude_none=True,
)
def get_avatar_generation(task_id: str) -> AvatarGenerateResponse:
    """
    Poll a generate or stylize task: QUEUED, PROCESSING, then COMPLETED with
    the result or FAILED with the error.

    ``{task_id}`` in the route path becomes the ``task_id`` argument. 404 for
    an id this server never issued or has since dropped (the tracker keeps
    the most recent tasks only).
    """
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

    The filters are query-string parameters (``?event=voice_use&limit=20``):
    FastAPI treats any simple-typed argument that is not in the path as one.
    ``head`` is the newest entry's hash, which can be stored elsewhere to
    detect a rewritten chain later. 400 for an unknown event name.
    """
    # An unknown event name would silently match nothing; refusing it tells
    # the caller about the typo.
    if event and event not in audit_log.EVENTS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"unknown event {event!r}; one of {list(audit_log.EVENTS)}")
    trail = audit_log.shared_audit()
    return {"entries": trail.query(event=event, subject=subject, since=since, limit=limit), "head": trail.head(), "events": list(audit_log.EVENTS)}


@app.get("/api/v1/audit/verify")
def verify_audit_trail() -> dict:
    """Recompute the hash chain: is the trail intact? Reports the first entry where it breaks.

    Each row's hash covers its content plus the previous row's hash, so an
    edited, deleted or reordered row breaks every hash after it.
    """
    return audit_log.shared_audit().verify_chain()


# Upload limits for provenance checks: a video file may be large (200 MB);
# a manifest is a small JSON document (1 MB).
MAX_VERIFY_BYTES = 200 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024


async def _spool_upload(upload: UploadFile, limit: int, suffix: str = "") -> Path:
    """Stream an upload to a temp file outside ``outputs/`` (which is served publicly), refusing one over ``limit``.

    *Spooling* means copying the upload to disk piece by piece instead of
    holding it all in memory, which matters for a 200 MB video. Its readers
    (``authenticity.verify_file``, ``voice_to_avatar.prepare``,
    ``protected_voices`` registration) take a file path, which a temp file
    provides.

    Returns the temp file's path. The caller owns it and must delete it
    (each caller does so in a ``finally``). Raises 413 past ``limit``; on any
    error the partial file is removed before the exception propagates.
    """
    import tempfile

    # delete=False: the file must outlive this handle, because it is closed
    # here and reopened by path later. ``suffix`` lets the verify route keep
    # the uploaded file's extension on the temp name.
    handle = tempfile.NamedTemporaryFile(prefix="verify-", suffix=suffix, delete=False)
    total = 0
    try:
        # ``:=`` (the walrus operator) assigns and tests in one step: read
        # 1 MiB (1 << 20 bytes) at a time until read() returns b"", the end.
        while chunk := await upload.read(1 << 20):
            total += len(chunk)
            if total > limit:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"the file is larger than the {limit // (1024 * 1024)} MB limit",
                )
            handle.write(chunk)
    # BaseException, not Exception, so a cancelled request (CancelledError
    # is a BaseException) also cleans up its partial file before re-raising.
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

    Errors: 400 when both or neither of ``file`` / ``path`` is sent or the
    manifest is not JSON; 413 for an oversized file or manifest; 422 when the
    file cannot be read as audio or video; plus ``resolve_audio_path``'s 400
    and 404 for a named path.
    """
    import json

    import authenticity

    # Exactly-one-of: ``(a is None) == (b is None)`` is True when both are
    # missing or both are present, the two cases to refuse.
    if (file is None) == (path is None):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="send exactly one of: a 'file' upload, or a 'path' under outputs/")
    document = None
    if manifest_file is not None:
        # The same one-byte-over trick as _read_upload: detect "too big"
        # without reading all of it.
        raw = await manifest_file.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="the manifest is larger than 1 MB")
        try:
            document = json.loads(raw)
        except ValueError as err:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="the manifest is not valid JSON") from err
    # Set only when a temp file was made, so ``finally`` deletes uploads and
    # never a file the caller named under outputs/.
    spooled: Optional[Path] = None
    try:
        if file is not None:
            # The client's file name is untrusted: only its extension is kept,
            # capped at 12 characters.
            spooled = await _spool_upload(file, MAX_VERIFY_BYTES, suffix=Path(file.filename or "").suffix[:12])
            target = spooled
        else:
            target = resolve_audio_path(path or "")
        try:
            # The name echoed back in the report: the base name only (Path.name
            # strips any folders), capped at 120 characters.
            shown = Path((file.filename or "") if file is not None else target.name).name[:120] or None
            # Reading the media and looking for watermarks is blocking work, so
            # it runs on a worker thread, as in register_avatar_face.
            return await run_in_threadpool(authenticity.verify_file, target, document, shown)
        except ValueError as err:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=_scrub(str(err))) from err
    finally:
        if spooled is not None:
            spooled.unlink(missing_ok=True)


def _avatar_or_http(avatar_id: str, usable: bool = True):
    """
    Look an avatar up and translate the store's exceptions into HTTP codes.

    With ``usable`` (the default) the face must also have consent. Returns
    the record. Raises 400 for a malformed id, 404 for an unknown one and 403
    for a face whose provenance forbids use.
    """
    # Each store exception maps to one status code; keeping the mapping here
    # means every route answers the same way for the same problem.
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
    """The avatar image. Refused for an image whose provenance forbids use.

    ``FileResponse`` streams a file from disk as the response body, with the
    given content type. ``image/png`` because ``avatar_store.register``
    stores every image as PNG.
    """
    record = _avatar_or_http(avatar_id)
    return FileResponse(record.path, media_type="image/png")


# 204 No Content: success with an empty body, the usual answer to a DELETE.
@app.delete("/api/v1/avatar/faces/{avatar_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_avatar_face(avatar_id: str) -> None:
    """
    Remove an avatar image and its provenance sidecar.

    ``usable=False``: the lookup skips the consent check, so a face without
    consent can still be deleted. 404 for an unknown id, 400 for a malformed one.
    """
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

    ``include_landmarks`` adds the full landmark list (hundreds of points),
    off by default to keep the response small. Errors: 400 for both or
    neither input or an image that cannot be decoded, the ``_avatar_or_http``
    codes for an id, 503 when the landmarker bundle is missing.
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
        """Decode the image and run the quality check; runs on a worker thread."""
        if data is not None:
            image, notices = avatar_store.decode_image(data)
        else:
            # The exactly-one-of check above guarantees this; the assert states
            # it for the type checker, which cannot follow an XOR.
            assert avatar_id is not None
            image, notices = avatar_store.decode_image(_avatar_or_http(avatar_id).path)
        with FACE_ENGINE_LOCK:
            return shared_face_engine().check_quality(image), notices

    # Exceptions raised on the worker thread are re-raised here, at the
    # await, so the usual try/except works across the thread hop.
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
        # Rights notices found inside the file (a copyright line in PNG text,
        # a JPEG comment or XMP), returned with the analysis.
        embeddedNotices=notices,
    )


# ---------------------------------------------------------------------------
# Render jobs
# ---------------------------------------------------------------------------
def _render_response(job_id: str, queued_job) -> RenderJobResponse:
    """
    Project a queue entry onto the API's render-job shape.

    Shared by submit, poll and batch routes so a job looks the same
    everywhere. ``progress`` is left out (None, then dropped by
    ``response_model_exclude_none``) until the job has started, so a queued
    job does not claim 0 % done.
    """
    result = queued_job.result
    started = queued_job.status.value != "QUEUED"
    return RenderJobResponse(
        jobId=job_id,
        status=queued_job.status,
        # ``(result or {})`` lets .get() work before there is a result. The
        # engine that actually ran wins over the one requested.
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

    The body is an ``AvatarRenderJob``, the frozen audio-to-vision contract;
    FastAPI validates it and answers 422 on its own if it is malformed.
    ``engine`` is a query parameter. Returns 202 with the job id to poll.
    """
    queued_job = _submit_render(job, engine)
    return _render_response(queued_job.job.job_id, queued_job)


def _submit_render(job: AvatarRenderJob, engine: Optional[str]):
    """Validate one job and put it on the queue; raises ``HTTPException`` with the reason if it cannot run.

    Shared by the single-job and the batch endpoint so both apply exactly the same rules.

    Returns the ``QueuedJob``. Status codes: 400 bad id, engine or audio;
    404 unknown avatar; 403 no consent; 409 a job id already queued; 503 the
    queue's broker or store cannot be reached.
    """
    # The preflight only runs when this queue will really render the job
    # (``executes``): a queue with no runner stores jobs without rendering,
    # and getattr's default covers a queue object without the attribute.
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
        return job_queue.enqueue(job, engine)
    # enqueue raises ValueError for a job id that is already queued.
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001 - broker or status store unreachable
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"render queue unavailable: {error}. Is Redis running (./start-docker.sh)?",
        ) from error


# A batch is a list of ordinary render jobs submitted in one request (cap: contracts.MAX_BATCH_JOBS).
batch_store = JobStore()  # kind "batch": batchId -> what happened to each item, so a batch survives a restart


@app.post("/api/v1/avatar/render-batch", status_code=status.HTTP_202_ACCEPTED)
def create_render_batch(request: RenderBatchRequest, engine: Optional[str] = None) -> dict:
    """
    Queue up to ``MAX_BATCH_JOBS`` (50) renders at once. Each item is checked on its own, so one bad item
    (unknown avatar, no consent, malformed, duplicate id) is reported against its index and does not
    stop the others. The response lists every item; poll ``GET .../render-batch/{batchId}``.

    That is why ``RenderBatchRequest.jobs`` holds raw dicts, not
    ``AvatarRenderJob`` objects: if FastAPI validated the whole list, one
    malformed item would 422 the entire request.
    """
    items: List[Dict[str, Any]] = []
    for index, raw in enumerate(request.jobs):
        # The id is read defensively, before validation, so even a rejected
        # item can be reported with the jobId the caller sent.
        job_id = raw.get("jobId") if isinstance(raw, dict) and isinstance(raw.get("jobId"), str) else None
        try:
            # Validated here, item by item, so a failure is caught per item.
            job = AvatarRenderJob.model_validate(raw)
            _submit_render(job, engine)
            items.append({"index": index, "jobId": job.job_id, "accepted": True})
        except ValidationError as err:
            # Only the first problem is reported, as "field.path: message";
            # ``loc`` is the path to the bad field as a tuple.
            first = err.errors()[0]
            where = ".".join(str(part) for part in first["loc"])
            items.append({"index": index, "jobId": job_id, "accepted": False, "httpStatus": 422,
                          "detail": f"{where}: {first['msg']}"})
        # _submit_render's refusals arrive as HTTPException; their code and
        # reason are recorded against this item instead of ending the request.
        except HTTPException as err:
            items.append({"index": index, "jobId": job_id, "accepted": False,
                          "httpStatus": err.status_code, "detail": str(err.detail)})
    batch_id = uuid.uuid4().hex[:12]
    # Only the item list is stored; each accepted job's live state is read
    # from the render queue when the batch is polled.
    batch_store.put("batch", batch_id, {"items": items})
    accepted = sum(1 for item in items if item["accepted"])
    return {"batchId": batch_id, "total": len(items), "accepted": accepted,
            "rejected": len(items) - accepted, "jobs": items}


@app.get("/api/v1/avatar/render-batch/{batch_id}")
def get_render_batch(batch_id: str) -> dict:
    """Live state of every job in a batch, with counts by state and ``done`` once none is still running."""
    record = batch_store.get("batch", batch_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="render batch not found")
    jobs, counts = [], {}
    for item in record["items"]:
        # Copy only the keys this item has: accepted items have no httpStatus.
        state = {k: item[k] for k in ("index", "jobId", "httpStatus", "detail") if k in item}
        if item["accepted"]:
            queued = job_queue.get(item["jobId"])
            # None means the queue lost the job; say so rather than hide it.
            state["status"] = queued.status.value if queued else "LOST"
            if queued is not None:
                # model_dump(by_alias=True) gives the camelCase wire names;
                # only the three fields useful in a batch summary are kept.
                state.update({k: v for k, v in _render_response(item["jobId"], queued).model_dump(
                    by_alias=True, exclude_none=True).items() if k in ("videoUrl", "error", "progress")})
        else:
            state["status"] = "REJECTED"
        counts[state["status"]] = counts.get(state["status"], 0) + 1
        jobs.append(state)
    # Done once nothing is waiting or running; COMPLETED, FAILED, REJECTED
    # and LOST are all final.
    done = not any(counts.get(name) for name in ("QUEUED", "PROCESSING"))
    return {"batchId": batch_id, "total": len(jobs), "counts": counts, "done": done, "jobs": jobs}


@app.get(
    # ``:path`` because jobId is an unconstrained contract string and may
    # contain a slash; without it such a job could be queued but never polled.
    "/api/v1/avatar/render-job/{job_id:path}",
    response_model=RenderJobResponse,
    response_model_exclude_none=True,
)
def get_render_job(job_id: str) -> RenderJobResponse:
    """
    Poll one render: status, progress (0..1 once started), and the video URL
    when it has COMPLETED. 404 for a job id the queue does not know.
    """
    queued_job = job_queue.get(job_id)
    if queued_job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="render job not found")
    return _render_response(job_id, queued_job)


@app.post(
    "/api/v1/avatar/render-job/{job_id:path}/lipsync-score",
    response_model=LipSyncScoreResponse,
)
def score_render_job(job_id: str) -> LipSyncScoreResponse:
    """SyncNet LSE-C / LSE-D for a finished render. The score names its method.

    LSE-C is SyncNet's confidence that lips and audio are in sync (higher is
    better); LSE-D is the distance between their embeddings (lower is
    better). A POST, not a GET, because it computes something and stores the
    result on the job. 404 unknown job, 409 not finished yet, 503 SyncNet
    weights missing, 422 a video SyncNet cannot score.
    """
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
    # Keep the score with the job so /api/v1/metrics can report it.
    job_queue.update(job_id, result={**queued_job.result, "lipsync": score.to_dict()})
    return LipSyncScoreResponse(jobId=job_id, score=score.to_dict())


# ---------------------------------------------------------------------------
# Voice-to-avatar (T8.2): the caller's own recording drives the face
# ---------------------------------------------------------------------------
# Largest recording accepted for voice-to-avatar.
MAX_SUPPLIED_AUDIO_BYTES = 50 * 1024 * 1024


@app.post("/api/v1/avatar/voice-to-avatar", status_code=status.HTTP_202_ACCEPTED)
async def voice_to_avatar_route(
    file: UploadFile = File(...),
    avatar_id: str = Form(..., alias="avatarId"),
    # Required: a recording of a real voice is used only with a stated basis,
    # which goes into the speech record and the video's manifest.
    consent_basis: str = Form(..., alias="consentBasis"),
    transcript: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    engine: Optional[str] = Form(None),
    render_quality: RenderQuality = Form(RenderQuality.PREVIEW, alias="renderQuality"),
    # ge / le are validation bounds (>= 0, <= 2); FastAPI answers 422 outside them.
    motion_intensity: Optional[float] = Form(None, alias="motionIntensity", ge=0.0, le=2.0),
) -> dict:
    """
    Upload a recording (with the basis the voice is used under) and an avatar id; get a render job.

    Without a ``transcript`` the words come from speech recognition (Whisper base). The audio is
    aligned like synthesised speech, and the video's manifest says it was supplied, not generated.

    Returns the render job (as ``GET .../render-job/{id}`` would) plus a
    ``speech`` block describing the prepared audio. Errors: 404 / 403 / 400
    for the avatar or engine, 413 for an oversized file, 422 when the audio
    cannot be decoded, transcribed or aligned, and the render preflight's codes.
    """
    import voice_to_avatar

    # Cheap refusals first: an unusable face or engine is known before any decoding or ASR.
    try:
        faces.require_usable(avatar_id)
        engine = render_engine.validate_engine(engine)
    except avatar_store.AvatarNotFound as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err)) from err
    except avatar_store.AvatarConsentError as err:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(err)) from err
    except (avatar_store.AvatarError, render_engine.RenderError) as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err

    upload = await _spool_upload(file, MAX_SUPPLIED_AUDIO_BYTES)
    try:
        # Decoding, speech recognition and alignment are slow model work: off the loop.
        speech = await run_in_threadpool(voice_to_avatar.prepare, upload, consent_basis, transcript, language)
    except voice_to_avatar.VoiceToAvatarError as err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(err)) from err
    finally:
        # The temp upload goes whatever happened; prepare() has written the
        # audio it needs to speech.audio_path.
        upload.unlink(missing_ok=True)

    # The same contract object a synthesised voice produces, so the renderer
    # cannot tell supplied audio from generated audio (golden rule 4). The
    # audio is passed as a file:// URI; the emotion vector is fixed at neutral.
    job = AvatarRenderJob.model_validate({
        "jobId": f"v2a-{uuid.uuid4().hex[:12]}", "avatarId": avatar_id, "audioUrl": speech.audio_path.resolve().as_uri(),
        "sampleRate": voice_to_avatar.OUTPUT_RATE, "durationSeconds": speech.duration_seconds,
        "phonemeTimestamps": speech.phoneme_timestamps, "emotionVector": {"happy": 0.0, "neutral": 1.0, "eyeblinkRate": 1.0},
        "renderQuality": render_quality, "targetFps": 25,
        # ``**{...}`` merges the key in only when it was sent, so the
        # contract's own default applies otherwise.
        **({"motionIntensity": motion_intensity} if motion_intensity is not None else {}),
    })
    queued = _submit_render(job, engine)
    # No response_model on this route, so the job is dumped to a dict by hand
    # (camelCase, no None fields) and the speech block is added beside it.
    return {**_render_response(job.job_id, queued).model_dump(by_alias=True, exclude_none=True), "speech": speech.to_dict()}


# ---------------------------------------------------------------------------
# Protected voices (T8.9): an opt-out list of voices that must not be cloned
# ---------------------------------------------------------------------------
# Largest recording accepted when adding a protected voice.
MAX_VOICE_UPLOAD_BYTES = 25 * 1024 * 1024


@app.get("/api/v1/abuse/protected-voices")
def list_protected_voices() -> dict:
    """Opaque ids of the protected voices. No audio and no names are kept; matching uses a speaker embedding.

    A *speaker embedding* is a fixed-length vector that summarises how a
    voice sounds; two recordings of one person give nearby vectors.
    ``threshold`` is the cosine similarity at which a clone request or its
    output counts as a match (``$PROTECTED_VOICE_THRESHOLD`` overrides it).
    """
    return {"voices": protected_voices.shared().list(), "threshold": protected_voices.threshold()}


@app.post("/api/v1/abuse/protected-voices", status_code=status.HTTP_201_CREATED)
async def add_protected_voice(file: UploadFile = File(...)) -> dict:
    """Protect a voice: upload a recording of it; only its speaker embedding is kept and the audio is discarded.

    Returns ``{"id": ...}`` with 201. 413 for an oversized file, 422 when no
    embedding can be computed from the recording.
    """
    path = await _spool_upload(file, MAX_VOICE_UPLOAD_BYTES)
    try:
        # Computing the embedding runs the speaker encoder: off the loop.
        voice_id = await run_in_threadpool(protected_voices.shared().register, path)
    except (RuntimeError, ValueError, OSError) as err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"could not use that recording: {err}") from err
    finally:
        path.unlink(missing_ok=True)  # the audio is not kept, only the embedding
    return {"id": voice_id}


@app.delete("/api/v1/abuse/protected-voices/{voice_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_protected_voice(voice_id: str) -> None:
    """Take a voice off the protected list. 404 for an unknown or already removed id.

    The embedding is blanked; a tombstone row keeps the id so it is never reused.
    """
    if not protected_voices.shared().remove(voice_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="protected voice not found")


@app.get("/api/v1/parameters")
def get_parameters() -> dict:
    """Every customisation parameter with its range and what measuring it found, and the count against the 50+ target (N-15).

    Read-only: ``parameters.catalogue`` builds it from the contract schemas
    and the measured notes kept beside them.
    """
    return parameters.catalogue()


@app.get("/api/v1/metrics")
def get_metrics() -> dict:
    """Queue depth, render timings and lip-sync scores so far, and the size of the audit trail (R-51).

    Only the in-memory queue can list its jobs. With Celery the queue,
    render and quality sections say ``available: False`` and why, rather
    than reporting zeros that look like real numbers (golden rule 8).
    """
    # The bound method ``job_queue.jobs`` if the backend has one, else None.
    list_jobs = getattr(job_queue, "jobs", None)
    body: Dict[str, Any] = {"queueBackend": queue_backend}
    if list_jobs is None:
        # A chained assignment: the three keys share one explanatory dict.
        body["queue"] = body["renders"] = body["quality"] = {
            "available": False, "reason": f"the {queue_backend} queue backend cannot list its jobs; use the in_memory backend or inspect Redis"}
    else:
        body.update(metrics.summarise(list_jobs()))
    body["audit"] = audit_log.shared_audit().head()
    return body


@app.post(
    "/api/v1/audio/align",
    response_model=AlignmentResponse,
    status_code=status.HTTP_200_OK,
)
def align_audio(request: AlignmentRequest) -> AlignmentResponse:
    """Standalone forced alignment extracting millisecond phoneme/viseme timestamps.

    *Forced alignment* takes audio and its transcript and finds when each
    sound was spoken. A *viseme* is the mouth shape for a group of phonemes
    that look alike (p, b and m all close the lips). The duration reported
    is the end of the last phoneme, not the length of the file.

    A plain ``def`` route, so FastAPI already runs it on a worker thread and
    the slow alignment does not block the event loop. Errors: 400 / 404 from
    the path checks, 422 for a file that is not audio, 500 otherwise.
    """
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
    # Re-raised untouched: the path checks above already chose the right
    # code, and the broad ``except Exception`` below would turn it into a 500.
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
# A client has this long after connecting to send its start message, and a
# session with no message for LIVE_IDLE_TIMEOUT_S (10 minutes) is closed, so
# a forgotten tab cannot hold one of the few session slots forever.
LIVE_START_TIMEOUT_S = 15.0
LIVE_IDLE_TIMEOUT_S = 600.0
# Sessions currently holding a slot. A plain counter guarded by a lock:
# ``+=`` is a read then a write, and two connections arriving together could
# otherwise both read the same value and both get in.
_live_active = 0
_live_lock = threading.Lock()
# A TypeAdapter validates against a type that is not itself a BaseModel,
# here the union "say or control message". Pydantic picks whichever member
# the data matches.
_live_message = TypeAdapter(LiveSayMessage | LiveControlMessage)


async def _live_fail(ws: WebSocket, code: str, detail: str, close: int = 1008) -> None:
    """Tell the client what went wrong (so it can fix it), then close.

    ``code`` is this API's short machine-readable reason ("busy",
    "consent"...); ``close`` is the WebSocket close code from RFC 6455:
    1000 normal, 1003 data the server cannot accept, 1008 policy violation
    (the default), 1011 server error, 1013 try again later.
    """
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

    A *WebSocket* is one long-lived, two-way connection: after an HTTP
    upgrade handshake either side can send messages at any time, as text or
    as bytes, until one side closes it. That is what lets the server push
    audio and frames as they are made instead of answering one request.

    The order of checks, cheapest first: the start message, the API key and
    rate limit, a free session slot, then the voice and face checks, and
    only then the slow ``session.open``. Whatever happens, ``finally``
    releases the slot, stops the receiver task and deletes the session's
    temp files.
    """
    # ``global`` because this function assigns to the module-level counter;
    # without it ``_live_active += 1`` would create a local variable.
    global _live_active
    # Completes the handshake. Until accept() nothing can be sent, and the key
    # check below needs the start message, so the socket is accepted first.
    await ws.accept()
    # Bound by hand (the request-id middleware is HTTP only) so every log line
    # of this session carries one id; the token restores the old value in finally.
    token = bind_request_id(new_request_id(ws.headers.get(REQUEST_ID_HEADER)))
    # What ``finally`` has to undo. Each is set only once its step succeeds,
    # so cleanup never releases a slot it did not take.
    reserved = False
    session = None
    receiver = None
    try:
        try:
            # asyncio.wait_for raises TimeoutError if the awaited call takes
            # longer than the limit. receive_json reads one text message and
            # parses it as JSON.
            raw = await asyncio.wait_for(ws.receive_json(), LIVE_START_TIMEOUT_S)
            start = LiveStartMessage.model_validate(raw)
        except asyncio.TimeoutError:
            return await _live_fail(ws, "start_timeout", f"send a start message within {LIVE_START_TIMEOUT_S:.0f} s", 1008)
        except (ValidationError, ValueError, TypeError) as err:
            return await _live_fail(ws, "bad_start", f"the first message must be a valid start message: {err}", 1003)
        except WebSocketDisconnect:
            return

        # The same gate as the HTTP middleware. The key may come in the start
        # message because a browser's WebSocket API cannot set custom headers;
        # a header still works for clients that can send one.
        allowed, _status_code, detail = security_gate.inspect(
            path="/api/v1/live",
            api_key=start.api_key or ws.headers.get(API_KEY_HEADER),
            client_host=ws.client.host if ws.client else None,
        )
        if not allowed:
            return await _live_fail(ws, "unauthorised", str(detail.get("detail", "request rejected")))

        # Test and take a slot in one locked step. The reply is sent after the
        # lock is released: never await while holding a threading lock.
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
        # Each failure maps to its own error code so the client can tell a
        # consent problem from a missing model; then the socket closes.
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
            # preflight checks consent for a clone reference, then that the
            # weights are on disk, before any model loads; check_kokoro_voice
            # rejects an unknown or missing speaker voice with ValueError.
            router.preflight(model_key, speaker, start.language)
            voice_engine_module.check_kokoro_voice(start.voice)
            # Building the session is cheap; open() below does the slow part.
            session = live_engine.LiveSession(
                router, faces, start.avatar_id, language=start.language, mode=start.mode,
                emotion=start.emotion, emotion_intensity=start.emotion_intensity, fps=start.fps,
                max_side=start.max_side, speaker_wav=speaker, clone_engine=start.clone_engine, voice=start.voice,
                motion_intensity=start.motion_intensity,
            )
            # open() checks face consent, decodes the photo and runs landmark
            # detection: blocking work, so it goes to a worker thread.
            info = await run_in_threadpool(session.open)
        # resolve_voice_reference raises HTTPException even here; only its
        # detail is reused, as an error message on the socket.
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

        # ``ready`` tells the client the frame size and fps it will receive
        # (``**info`` spreads open()'s dict into this one) and the request id
        # to quote in a bug report. Each audio message also carries its own
        # sampleRate, taken from the synthesised file.
        await ws.send_json({"type": "ready", "sessionId": session.session_id, "avatarId": start.avatar_id,
                            "mode": start.mode, "model": model_key, "sampleRate": 24000,
                            "requestId": current_request_id(), **info})
        logger.info("Live session %s opened: avatar=%s mode=%s %dx%d@%d", session.session_id,
                    start.avatar_id, start.mode, info["width"], info["height"], info["fps"])

        # A receiver task keeps reading while speech is being sent, so an
        # `interrupt` is seen between frames instead of after the whole text.
        # The *inbox* is an asyncio.Queue: a first-in, first-out buffer that
        # one coroutine puts into and another takes from, without threads or
        # locks (both run on the event loop). Messages starting with "_" are
        # the receiver's own markers, never sent by a client: _closed,
        # _malformed and _pcm (binary audio).
        inbox: asyncio.Queue = asyncio.Queue()

        async def receive() -> None:
            """Read every incoming message and put it in ``inbox``; return when the client leaves."""
            while True:
                try:
                    # receive() (not receive_json) returns the raw ASGI
                    # message dict, which may hold "text" or "bytes".
                    message = await ws.receive()
                except (WebSocketDisconnect, RuntimeError):  # the client left
                    inbox.put_nowait({"type": "_closed"})
                    return
                if message.get("type") == "websocket.disconnect":
                    inbox.put_nowait({"type": "_closed"})
                    return
                if message.get("bytes") is not None:  # a binary message is a chunk of the client's own audio
                    inbox.put_nowait({"type": "_pcm", "data": message["bytes"]})
                    continue
                try:
                    inbox.put_nowait(json.loads(message.get("text") or ""))
                except ValueError:  # not JSON: tell the client, keep listening
                    inbox.put_nowait({"type": "_malformed"})

        # create_task schedules receive() to run concurrently on the event
        # loop and returns at once; the main loop below carries on alongside it.
        receiver = asyncio.create_task(receive())
        # `say` messages that arrived while speaking, held to be handled in
        # order once the current text ends.
        queued: list = []
        audio_rate: Optional[int] = None  # set by audio_start; None means no audio stream is open
        audio_chunks = 0

        async def next_message() -> dict:
            """The next message to handle: held-back ones first, else wait on the inbox (with the idle timeout)."""
            if queued:
                return queued.pop(0)
            return await asyncio.wait_for(inbox.get(), LIVE_IDLE_TIMEOUT_S)

        # The session's main loop: one message per pass, dispatched on its type.
        while True:
            try:
                raw = await next_message()
            except asyncio.TimeoutError:
                return await _live_fail(ws, "idle", "no message for too long; the session was closed", 1000)
            # Valid JSON need not be an object ("3" or "[]" parse too), so the
            # type is read only from a dict.
            kind = raw.get("type") if isinstance(raw, dict) else None
            if kind in ("_closed", "stop"):
                break
            if kind == "_malformed":
                await ws.send_json({"type": "error", "code": "malformed", "detail": "messages must be JSON objects"})
                continue
            # -- streaming audio input (R-41): the client's own speech drives the mouth ----------
            if kind == "audio_start":
                try:
                    audio_start = LiveAudioStartMessage.model_validate(raw)
                except ValidationError as err:
                    await ws.send_json({"type": "error", "code": "bad_message", "detail": _scrub(str(err))[:300]})
                    continue
                audio_rate, audio_chunks = audio_start.sample_rate, 0
                # The client's own voice drives the face, so its consent basis
                # is recorded in the audit trail before any audio is used.
                audit_log.shared_audit().record("audio_supplied", subject=session.session_id, basis=audio_start.consent_basis,
                                                live=True, avatar=start.avatar_id, sampleRate=audio_rate)
                await ws.send_json({"type": "audio_ready", "drive": live_engine.AUDIO_DRIVE, "note": live_engine.AUDIO_DRIVE_NOTE,
                                    "maxChunkSeconds": live_engine.MAX_PCM_SECONDS})
                continue
            if kind == "_pcm":
                if audio_rate is None:
                    await ws.send_json({"type": "error", "code": "audio_not_started",
                                        "detail": "send an audio_start message (sampleRate, consentBasis) before audio chunks"})
                    continue
                try:
                    # ``async for`` pulls events from the async generator one at
                    # a time; each frame is sent as soon as it is rendered.
                    # Frames go out as binary messages (13-byte media header +
                    # JPEG, see live_engine); everything else as JSON text.
                    async for event in live_engine.stream_audio_chunk(session, audio_chunks, raw["data"], audio_rate):
                        if event.kind == "frame":
                            await ws.send_bytes(event.payload)
                        else:
                            await ws.send_json({"type": event.kind, **event.meta})
                    audio_chunks += 1
                # A bad chunk is reported and skipped; the session stays open.
                except live_engine.LiveError as err:
                    await ws.send_json({"type": "error", "code": "bad_audio", "detail": str(err)})
                continue
            # The end of the client's audio stream: report how many chunks
            # were animated and go back to accepting text.
            if kind == "audio_end":
                await ws.send_json({"type": "done", "drive": live_engine.AUDIO_DRIVE, "chunks": audio_chunks})
                audio_rate, audio_chunks = None, 0
                continue
            # Anything else must be a `say` or an `interrupt`. A bad message is
            # answered with an error (capped at 300 characters) and the session
            # carries on; only the start message is fatal when invalid.
            try:
                message = _live_message.validate_python(raw)
            except ValidationError as err:
                await ws.send_json({"type": "error", "code": "bad_message", "detail": _scrub(str(err))[:300]})
                continue
            if not isinstance(message, LiveSayMessage):  # an interrupt with nothing to interrupt
                await ws.send_json({"type": "interrupted"})
                continue

            # Kept in a variable (not written inline in ``async for``) so the
            # ``finally`` below can close it explicitly.
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
                    # Audio goes as a JSON description first, then the binary
                    # PCM16 chunk it describes.
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
                # aclose() runs the generator's own ``finally`` now, which
                # cancels the next sentence's synthesis if it has not started.
                await generator.aclose()
            if stopped:
                break
            if interrupted:
                await ws.send_json({"type": "interrupted"})
    except WebSocketDisconnect:
        pass
    # Cleanup in reverse order of set-up, whichever way the session ended.
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
        # Closing an already-closed socket raises; that case is expected here.
        try:
            await ws.close()
        except (RuntimeError, WebSocketDisconnect):
            pass


def _job_response(task_id: str, task_status: str, result: dict) -> SynthesisJobResponse:
    """Project a completed VoiceEngineRouter result onto the API response.

    The router's result uses snake_case keys; the response model's fields are
    camelCase on the wire. ``.get`` leaves a key the engine did not report as
    None instead of raising.
    """
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
    """
    Turn text into speech: pick the model, check it can run, then queue it.

    With the in-memory backend Celery runs eagerly (``task_always_eager``):
    ``.delay()`` executes the task inline and the response already holds the
    finished result (SUCCESS) or the error (FAILED). With the Celery backend the task
    goes to Redis and the response is QUEUED; poll
    ``GET /api/v1/audio/synthesize/{task_id}``.

    Errors: 400 bad speaker path, language or voice; 403 a clone reference
    without consent; 503 missing weights or an unreachable queue.
    """
    # Route before queueing, so an engine with no weights is a 503 that names
    # the fetch command now, not a queued job that fails (or downloads
    # gigabytes) later, and an unsupported language is a 400 the caller can fix.
    router = get_router()
    if request.speaker_wav:
        # Resolved (and confined to inputs/) before anything else looks at it.
        # Pydantic models are treated as read-only here; model_copy(update=...)
        # returns a new request with the one field replaced.
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
        voice_engine_module.check_kokoro_voice(request.voice)
    except VoiceConsentRequired as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(error)) from error
    except ModelWeightsMissing as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    try:
        # A Celery message must be plain JSON, so the request is dumped to a
        # dict (mode="json" turns enums and the like into plain values).
        task = synthesize_audio.delay(request.model_dump(by_alias=True, mode="json"))
        task_status = getattr(task, "status", "QUEUED")
        # Celery says PENDING for "not started (or unknown)"; the API's word is QUEUED.
        if task_status == "PENDING":
            task_status = "QUEUED"
        # For eager (in-memory) mode, task result is available immediately
        result = task.result if hasattr(task, "result") else None
        if task_status == "SUCCESS" and isinstance(result, dict):
            return _job_response(task.id, task_status, result)
        return SynthesisJobResponse(taskId=task.id, status=_SYNTH_STATES.get(task_status, task_status), error=_task_error(task))
    # Anything raised while queueing (Redis down, a broken result store) is
    # reported as 503 Service Unavailable with the cause.
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Synthesis service unavailable: {str(error)}",
        ) from error


# Celery state names that the submit route renames; any other state passes
# through as Celery reports it. Defined after the route that uses it, which
# works because the name is only looked up when the route runs.
_SYNTH_STATES = {"FAILURE": "FAILED"}


def _task_error(task: Any) -> Optional[str]:
    """The exception a failed task raised, as ``Type: message`` with server paths removed; None otherwise.

    For a failed Celery task, ``task.result`` holds the exception object
    rather than a return value, hence the isinstance check.
    """
    if getattr(task, "state", None) != "FAILURE" or not isinstance(getattr(task, "result", None), BaseException):
        return None
    return _scrub(f"{type(task.result).__name__}: {task.result}")


@app.get("/api/v1/audio/synthesize/{task_id}", response_model=SynthesisJobResponse)
def get_synthesis_job(task_id: str) -> SynthesisJobResponse:
    """
    Poll a synthesis task by id; the full result once it has succeeded.

    Never 404s: Celery reports an id it has never seen as PENDING, so an
    unknown id reads as QUEUED. A result store that cannot be read gives
    status UNKNOWN (logged), not a 500.
    """
    try:
        # AsyncResult is a handle that looks the task up in the result
        # backend; it does not wait for the task.
        task = celery.AsyncResult(task_id)
        # Celery's state names mapped to the API's vocabulary.
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
        return SynthesisJobResponse(taskId=task_id, status=task_status, error=_task_error(task))
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

    Both are query-string parameters with defaults, so ``GET
    /api/v1/audio/languages`` alone returns the first 100.
    """
    # Clamp the limit to 0..catalogue size, so a negative or huge value from
    # the caller cannot produce a surprising slice.
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
    """Resolve one code and report which backends can speak it.

    The code is normalised to ISO-639-3 first, so a two-letter code works.
    An unknown code is not a 404: it comes back marked as unsupported.
    """
    return LanguageEntry.model_validate(language_registry.resolve(code).to_dict())


@app.get("/api/v1/audio/voices")
def list_voices() -> dict:
    """The Kokoro speaker voices this server offers, with gender (so the studio can match the face) and whether each is on disk.

    ``present`` is checked per voice, because each Kokoro voice is its own
    file (``voices/<id>.pt``) in the local Hugging Face cache; the check makes
    no network call.
    """
    return {"default": voice_engine_module.DEFAULT_KOKORO_VOICE, "voices": [
        {"id": key, **info, "present": voice_engine_module.kokoro_voice_present(key)} for key, info in voice_engine_module.KOKORO_VOICES.items()]}


@app.get("/api/v1/audio/emotions", response_model=EmotionPresetsResponse)
def list_emotions() -> EmotionPresetsResponse:
    """The emotion prosody presets and the prosody each one applies.

    *Prosody* is the melody of speech: pitch, speaking rate and loudness. A
    preset is a named set of those adjustments (see ``emotion_engine``).
    """
    return EmotionPresetsResponse(
        presets=[EmotionPresetEntry.model_validate(p) for p in preset_catalogue()],
    )


@app.post(
    "/api/v1/audio/quality-audit",
    response_model=QualityAuditResponse,
    status_code=status.HTTP_200_OK,
)
def audit_quality(request: QualityAuditRequest) -> QualityAuditResponse:
    """Predict MOS, PESQ, STOI and SI-SDR for a generated clip (SQUIM).

    MOS is a 1-5 naturalness score, PESQ perceived quality, STOI
    intelligibility, SI-SDR distortion. SQUIM (torchaudio) *predicts* them
    without a matching recording of the same words. Its MOS model still needs
    some clean speech to compare against: ``reference_path`` names one, and
    without it the clip is compared with itself, which biases MOS upward and
    is flagged in the report's warnings. The report names the scorer that ran.
    """
    audio_path = resolve_audio_path(request.audio_path)
    reference = (
        resolve_audio_path(request.reference_path) if request.reference_path else None
    )
    try:
        # The first audit loads the SQUIM weights; later ones reuse them.
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

    ECAPA-TDNN is a speaker-verification network: it turns each clip into a
    speaker embedding, and the score is the cosine similarity of the two.
    When it cannot load, an MFCC comparison is used and labelled as such.
    Both paths go through ``resolve_audio_path``; a failure becomes 422 or
    500 through ``_audio_failure``.
    """
    reference = resolve_audio_path(request.reference_path)
    generated = resolve_audio_path(request.generated_path)
    try:
        report = quality_auditor.speaker_similarity(reference, generated)
    except Exception as err:  # noqa: BLE001
        raise _audio_failure("Similarity scoring", err) from err
    return VoiceSimilarityResponse(report=report.to_dict())


# The built studio (Docker image, T7.2): served from the same origin as the API when FRONTEND_DIST
# points at a `npm run build` output. Mounted last, because a mount at "/" would otherwise answer
# for every API path registered after it.
_frontend_dist = os.getenv("FRONTEND_DIST")
# Mounted only when the build is really there, so a development server
# without one simply has no studio at "/" (Vite serves it instead).
if _frontend_dist and Path(_frontend_dist, "index.html").is_file():
    # html=True serves index.html for "/" and for a folder path.
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="studio")
