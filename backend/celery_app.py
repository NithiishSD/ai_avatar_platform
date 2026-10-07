"""
The task queue: where every piece of heavy work actually runs.

Synthesis and rendering both load models and take seconds, so they must not
happen inside an HTTP request - a 6 GB card cannot hold one model per web
worker. The API validates, enqueues, and returns a task id; this module does
the work.

**Celery** is a distributed task queue. A *task* is a normal function wrapped
in ``@celery.task``; calling ``task.delay(args)`` serialises the arguments,
puts a message on a *broker*, and returns immediately. A *worker* process
consumes the message and runs the function, writing the return value to a
*result backend* that the API can poll.

In development there is no worker and no Redis: ``task_always_eager`` makes
``.delay()`` run the function inline, so the whole pipeline works with a single
process. That is what ``QUEUE_BACKEND=in_memory`` means.

**How to say this in an interview:** "Long-running model inference is pushed to
a Celery task so requests stay fast and bounded; in development the same code
path runs eagerly in-process, so there is no separate code path to test."
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# .env has to be loaded BEFORE `from celery import Celery` below, because
# Celery reads its configuration from the environment at import time. This is
# why the imports in this file are deliberately not all at the top, and why
# a linter complaining about import order here would be wrong.
#
# __file__ is this file's path; .resolve() makes it absolute and follows
# symlinks; parents[1] is the repo root (parents[0] would be backend/).
project_root = Path(__file__).resolve().parents[1]
dotenv_path = project_root / ".env"
if dotenv_path.exists():
    # Explicit path, so the file is found no matter which directory the
    # process was started from - a worker started in backend/ and an API
    # started at the root must read the same config.
    load_dotenv(dotenv_path=dotenv_path)
else:
    # Fall back to load_dotenv's own search (cwd upwards). Keeps a fresh
    # checkout with no .env working off the defaults rather than crashing.
    load_dotenv()

from celery import Celery

from contracts import AudioSynthesisRequest, AvatarRenderJob
from voice_engine import VoiceEngineRouter

logger = logging.getLogger(__name__)

# One router per worker process. Building a new one per task threw away every
# lazily-loaded model and paid the cold-start cost on every request.
#
# Module-level singleton: module bodies execute once per process, so these two
# names are created once and shared by every task in that process.
_router: Optional[VoiceEngineRouter] = None
_router_lock = threading.Lock()


def get_router() -> VoiceEngineRouter:
    """
    Build the router once, safely, however many threads ask at once.

    This is the *double-checked locking* pattern. The first ``if`` is the fast
    path: once the router exists, no thread takes the lock at all. The second
    ``if``, inside the lock, is the correct one: two threads can both pass the
    first check before either takes the lock, and without the re-check the
    second would build a duplicate router and load every model twice.

    **How to say this in an interview:** "Double-checked locking - cheap read
    on the common path, lock only on first construction, re-check inside the
    lock because the first check is not atomic with the assignment."
    """
    # `global` is required to *rebind* a module-level name from inside a
    # function. Without it, `_router = ...` would create a local variable and
    # the singleton would never be populated.
    global _router
    if _router is None:
        with _router_lock:
            if _router is None:
                _router = VoiceEngineRouter()
    return _router



# Default to in_memory, so a fresh checkout with no .env runs with no broker.
# `!= "celery"` rather than `== "in_memory"` means any unrecognised value fails
# safe to the mode that needs no external service.
is_in_memory = os.getenv("QUEUE_BACKEND", "in_memory").lower() != "celery"

if is_in_memory:
    # Celery reads CELERY_BROKER_URL / CELERY_RESULT_BACKEND straight from the
    # environment, and they take precedence over both the constructor
    # arguments below and any later conf assignment. Since .env sets both to
    # Redis for the celery backend, in_memory mode was silently storing its
    # eager results in a Redis that need not be running - the one thing
    # "QUEUE_BACKEND=in_memory" promises it does not do. They belong to the
    # celery backend only, so clear them before the app reads them.
    #
    # The lesson generalises: a library that reads the environment itself can
    # outrank the arguments you passed it, so "I set it in code" is not proof.
    # See docs/LEARNING_NOTES.md.
    for _celery_env in ("CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"):
        # dict.pop(key, None) removes and returns the value, or None if absent -
        # so this both clears the variable and tells us whether it was set,
        # without a separate `in` check.
        if os.environ.pop(_celery_env, None):
            logger.debug(
                "QUEUE_BACKEND=in_memory: ignoring %s (in-process broker only)",
                _celery_env,
            )
    celery = Celery(
        "avatar_platform",
        # memory:// is an in-process broker and cache+memory:// an in-process
        # result store. Neither touches the network, so no Redis is required.
        broker="memory://",
        backend="cache+memory://",
    )
else:
    celery = Celery(
        "avatar_platform",
        broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
        # Note the different database numbers (/0 and /1): the broker's queues
        # and the result store share one Redis server but not one keyspace.
        backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
    )

celery.conf.update(
    # JSON, not Celery's old default pickle. pickle can execute arbitrary code
    # on deserialisation, so a compromised broker would become remote code
    # execution. accept_content locks the worker to JSON on the way in too.
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Store and compare times in UTC. Local timestamps in a queue are how you
    # get an hour of duplicate or skipped work twice a year.
    timezone="UTC",
    enable_utc=True,
    # The switch that makes in_memory mode work: .delay() runs the task inline
    # and synchronously instead of publishing it.
    task_always_eager=is_in_memory,
    # Eager tasks normally skip the result backend entirely. Storing them keeps
    # AsyncResult(task_id) working, so the API's polling endpoint behaves the
    # same in both modes - and it is why the cleared env vars above mattered.
    task_store_eager_result=is_in_memory,
)


def run_render(job: AvatarRenderJob, engine: Optional[str] = None, progress=None) -> dict:
    """Render a job and return the result payload (shared by both queues)."""
    # Imported lazily: render_engine pulls in torch, OpenCV and MediaPipe, and
    # this module is imported by the API process too. A top-level import would
    # cost seconds of startup and VRAM in a process that may never render.
    import render_engine

    return render_engine.render_job(job, engine=engine, progress=progress).to_dict()


def _render_status_store():
    """
    The status store, built per call rather than held open.

    Deliberately not a module singleton: a Celery worker may be forked, and a
    Redis connection inherited across a fork is not safe to share. Building one
    per call is cheap compared with a render.
    """
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
        # logger.error, not logger.exception: the traceback of a Redis timeout
        # adds noise, and the message already names the job and the state that
        # was lost. Swallowing it entirely would violate golden rule 1.
        logger.error("Could not record status of render job %s (%s): %s", job_id, changes.get("status"), err)


# The decorator registers this function as a Celery task. The explicit `name`
# matters: without it Celery derives one from the module path, so moving the
# function would silently orphan messages already sitting on the broker that
# still refer to the old name.
@celery.task(name="avatar.process_render_job")
def process_render_job(payload: dict, engine: Optional[str] = None) -> dict:
    """Render a queued ``AvatarRenderJob`` to an MP4 and record the outcome."""
    # The task receives a plain dict - only JSON crossed the broker - so the
    # first thing it does is re-validate it into the frozen contract. The
    # message may have been queued by an older version of the API.
    job = AvatarRenderJob.model_validate(payload)
    # Status as a string, not JobStatus.PROCESSING: this value is going into
    # JSON in Redis, and job_queue.update unwraps enums for exactly this.
    _record(job.job_id, status="PROCESSING")
    try:
        result = run_render(job, engine)
    except Exception as err:  # noqa: BLE001 - the failure is the job's result
        logger.exception("Render job %s failed", job.job_id)
        # type(err).__name__ keeps the exception class in the message, so the
        # client sees "InsufficientVRAM: ..." rather than an opaque string.
        error = f"{type(err).__name__}: {err}"
        _record(job.job_id, status="FAILED", error=error)
        # Returns a FAILED payload instead of re-raising. Re-raising would mark
        # the Celery task as failed, which is true but useless - the client
        # polls our status record, and a traceback is not a useful API reply.
        return {"jobId": job.job_id, "status": "FAILED", "error": error}
    _record(job.job_id, status="COMPLETED", progress=1.0, result=result)
    return {"jobId": job.job_id, "status": "COMPLETED", "result": result}


@celery.task(name="avatar.synthesize_audio")
def synthesize_audio(payload: dict) -> dict:
    """Run the Developer 1 voice engine from a Celery task."""
    request = AudioSynthesisRequest.model_validate(payload)
    # Every field is passed explicitly rather than **request.model_dump():
    # the router's signature is the real contract here, so a field added to the
    # request model fails visibly as a missing argument instead of being
    # silently swallowed by a **kwargs the router ignores.
    result = get_router().synthesize(
        text=request.text,
        # .value because the router takes a plain string, not the enum.
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
        clone_engine=request.clone_engine,
        output_filename=request.output_filename,
    )
    # __dict__ turns the SynthesisResult dataclass into a plain dict so Celery
    # can JSON-serialise it. Works because every field is already a primitive;
    # a nested object would need dataclasses.asdict() or a .to_dict().
    return result.__dict__
