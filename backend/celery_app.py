import logging
import os
import threading
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[1]
dotenv_path = project_root / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    load_dotenv()

from celery import Celery

from contracts import AudioSynthesisRequest, AvatarRenderJob
from voice_engine import VoiceEngineRouter

logger = logging.getLogger(__name__)

# One router per worker process. Building a new one per task threw away every
# lazily-loaded model and paid the cold-start cost on every request.
_router: Optional[VoiceEngineRouter] = None
_router_lock = threading.Lock()


def get_router() -> VoiceEngineRouter:
    global _router
    if _router is None:
        with _router_lock:
            if _router is None:
                _router = VoiceEngineRouter()
    return _router



is_in_memory = os.getenv("QUEUE_BACKEND", "in_memory").lower() != "celery"

if is_in_memory:
    celery = Celery(
        "avatar_platform",
        broker="memory://",
        backend="cache+memory://",
    )
else:
    celery = Celery(
        "avatar_platform",
        broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
        backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
    )

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_always_eager=is_in_memory,
    task_store_eager_result=is_in_memory,
)


def run_render(job: AvatarRenderJob, engine: Optional[str] = None, progress=None) -> dict:
    """Render a job and return the result payload (shared by both queues)."""
    import render_engine

    return render_engine.render_job(job, engine=engine, progress=progress).to_dict()


def _render_status_store():
    from job_queue import CeleryJobQueue

    return CeleryJobQueue()


def _record(job_id: str, **changes) -> None:
    """
    Write a render job's state where the API reads it.

    A broken status store must not turn a finished render into a crash, but
    it must be loud: the API would otherwise report QUEUED forever.
    """
    try:
        _render_status_store().update(job_id, **changes)
    except Exception as err:  # noqa: BLE001
        logger.error("Could not record status of render job %s (%s): %s", job_id, changes.get("status"), err)


@celery.task(name="avatar.process_render_job")
def process_render_job(payload: dict, engine: Optional[str] = None) -> dict:
    """Render a queued ``AvatarRenderJob`` to an MP4 and record the outcome."""
    job = AvatarRenderJob.model_validate(payload)
    _record(job.job_id, status="PROCESSING")
    try:
        result = run_render(job, engine)
    except Exception as err:  # noqa: BLE001 - the failure is the job's result
        logger.exception("Render job %s failed", job.job_id)
        error = f"{type(err).__name__}: {err}"
        _record(job.job_id, status="FAILED", error=error)
        return {"jobId": job.job_id, "status": "FAILED", "error": error}
    _record(job.job_id, status="COMPLETED", progress=1.0, result=result)
    return {"jobId": job.job_id, "status": "COMPLETED", "result": result}


@celery.task(name="avatar.synthesize_audio")
def synthesize_audio(payload: dict) -> dict:
    """Run the Developer 1 voice engine from a Celery task."""
    request = AudioSynthesisRequest.model_validate(payload)
    result = get_router().synthesize(
        text=request.text,
        mode=request.mode.value,
        language=request.language,
        quality=request.quality,
        style=request.style,
        speed=request.speed,
        pitch=request.pitch,
        return_alignment=request.return_alignment,
        emotion=request.emotion,
        emotion_intensity=request.emotion_intensity,
        emotion_vector=request.emotion_vector,
        audit_quality=request.audit_quality,
        speaker_wav=request.speaker_wav,
        output_filename=request.output_filename,
    )
    return result.__dict__