"""
Two interchangeable render queues behind one shape.

``InMemoryJobQueue`` keeps jobs in this process and is the development default;
``CeleryJobQueue`` publishes them to a broker for a separate worker. ``app.py``
picks one from ``QUEUE_BACKEND`` and never cares which it got, because both
expose the same four methods: ``enqueue``, ``get``, ``update``, ``executes``.

That is *duck typing* - Python's "if it has the methods, it fits" - rather than
a shared abstract base class. There is no ABC here on purpose: two
implementations with no shared behaviour would inherit nothing from a base
class, so it would only add a file to read.

**How to say this in an interview:** "The queue is swappable behind a common
method set, so development runs with no broker at all and production runs
Celery, with no branching in the calling code."
"""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional

from contracts import AvatarRenderJob, JobStatus
from job_store import JobStore
from request_context import run_in_context
from redis import Redis

# __name__ gives this logger the module's dotted path ("job_queue"), so log
# output names its source and the app can tune verbosity per module. Never
# use logging.info() directly - that writes to the root logger and loses that.
logger = logging.getLogger(__name__)

# Renders a job and returns the result payload. ``report(done, total)`` is
# called as frames are written.
#
# A type *alias* for a callable signature: (job, engine, report) -> dict.
# Naming it means the three places that pass a renderer around all agree on
# the shape, and the type checker catches a mismatched argument order.
RenderRunner = Callable[[AvatarRenderJob, Optional[str], Callable[[int, int], None]], dict]


@dataclass(frozen=True)
class QueuedJob:
    """
    One job plus everything known about its progress.

    ``frozen=True`` makes instances immutable: assigning to a field raises.
    That is deliberate for shared state - a status update cannot be applied
    half-way by one thread while another reads it. Updates go through
    ``dataclasses.replace()``, which builds a new instance, so a reader always
    sees a complete consistent snapshot rather than a partly-mutated object.
    """

    job: AvatarRenderJob
    status: JobStatus = JobStatus.QUEUED
    engine: Optional[str] = None
    progress: float = 0.0
    result: Optional[dict] = None
    error: Optional[str] = None


class InMemoryJobQueue:
    """
    Development queue: jobs live in this process.

    With a ``runner`` the queue also executes them, on one worker thread --
    one, because the GPU holds one heavy model at a time and two renders
    would only fight over it. Without a runner it just records the job, which
    is what the contract tests need.
    """

    def __init__(self, runner: Optional[RenderRunner] = None, store: Optional[JobStore] = None):
        # job_id -> QueuedJob. With a ``store`` every change is also written to
        # SQLite and read back below, so a restart does not forget jobs (R-21).
        self._jobs: Dict[str, QueuedJob] = {}
        self._store = store
        # A mutex. FastAPI serves requests on multiple threads, so two
        # requests can touch _jobs at once; without this, a dict write could
        # interleave with a read. Python's GIL does not save you here: it
        # protects single bytecodes, not a check-then-act sequence like the
        # "already exists" test in enqueue.
        self._lock = threading.Lock()
        self._runner = runner
        # max_workers=1 is the VRAM constraint expressed as code: a second
        # concurrent render would try to load a second model onto a 6 GB card.
        # thread_name_prefix only affects log readability.
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")
            if runner is not None
            else None
        )
        if store is not None:
            self._recover()

    KIND = "render"

    def _save(self, job_id: str) -> None:
        queued = self._jobs.get(job_id)
        if self._store is None or queued is None:
            return
        self._store.put(self.KIND, job_id, {
            "job": queued.job.model_dump(by_alias=True, mode="json"),
            "status": queued.status.value,
            "engine": queued.engine,
            "progress": queued.progress,
            "result": queued.result,
            "error": queued.error,
        })

    def _recover(self) -> None:
        """
        Load jobs left by an earlier process.

        Finished ones come back as they were. A job that was QUEUED never
        started, so it is run now. A job that was PROCESSING died with the old
        process; it becomes FAILED with a reason the client can act on, rather
        than staying "PROCESSING" forever.
        """
        assert self._store is not None
        rerun = []
        for job_id, record in self._store.all(self.KIND):
            status = JobStatus(record["status"])
            error = record.get("error")
            if status == JobStatus.PROCESSING:
                status, error = JobStatus.FAILED, "interrupted by a server restart; submit the job again"
            elif status == JobStatus.QUEUED and self._executor is None:
                pass  # nothing here can run it; leave it queued for a worker
            elif status == JobStatus.QUEUED:
                rerun.append(job_id)
            self._jobs[job_id] = QueuedJob(
                job=AvatarRenderJob.model_validate(record["job"]),
                status=status,
                engine=record.get("engine"),
                progress=float(record.get("progress") or 0.0),
                result=record.get("result"),
                error=error,
            )
            if error != record.get("error"):
                self._save(job_id)
        for job_id in rerun:
            logger.info("Re-running render job %s that was queued before the restart", job_id)
            assert self._executor is not None
            self._executor.submit(self._run, job_id)

    # @property exposes a computed value as an attribute (`queue.executes`,
    # no parentheses). CeleryJobQueue sets `executes = True` as a plain class
    # attribute, and callers cannot tell the difference - which is the point.
    @property
    def executes(self) -> bool:
        """Whether an enqueued job will actually be rendered."""
        return self._runner is not None

    def enqueue(self, job: AvatarRenderJob, engine: Optional[str] = None) -> QueuedJob:
        # `with self._lock:` acquires and - crucially - releases even if the
        # body raises. A manual acquire()/release() pair would deadlock the
        # whole queue the first time the ValueError below fired.
        with self._lock:
            if job.job_id in self._jobs:
                raise ValueError(f"jobId already exists: {job.job_id}")
            queued_job = QueuedJob(job=job, engine=engine)
            self._jobs[job.job_id] = queued_job
            self._save(job.job_id)
        # Submitted *outside* the lock: the render takes seconds, and holding
        # the mutex across it would block every status poll in the meantime.
        if self._executor is not None:
            # submit() returns immediately; the thread pool runs _run later.
            # We deliberately drop the Future - progress is tracked in _jobs,
            # and nothing here awaits the result.
            # run_in_context: the worker thread logs under the id of the request that queued the job.
            self._executor.submit(run_in_context(self._run), job.job_id)
        return queued_job

    def get(self, job_id: str) -> Optional[QueuedJob]:
        with self._lock:
            # .get() rather than [job_id]: an unknown id is a 404, not a crash.
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes) -> None:
        # **changes collects arbitrary keyword arguments into a dict, so one
        # method serves status, progress, result and error updates.
        with self._lock:
            current = self._jobs.get(job_id)
            if current is not None:
                # replace() copies the frozen dataclass with these fields
                # overridden. Since QueuedJob is immutable, this is the only
                # way to "change" it - and it is atomic from a reader's view.
                self._jobs[job_id] = replace(current, **changes)
                self._save(job_id)

    def _run(self, job_id: str) -> None:
        """
        The worker thread body. Leading underscore = internal by convention.

        Nothing calls this directly; the executor does. Note it never raises:
        an exception escaping a thread-pool task would vanish into the Future
        and the job would sit at PROCESSING forever.
        """
        queued = self.get(job_id)
        if queued is None or self._runner is None:
            return
        self.update(job_id, status=JobStatus.PROCESSING)
        try:
            result = self._runner(
                queued.job,
                queued.engine,
                # The progress callback, closing over job_id. max(1, total)
                # guards against a divide-by-zero when a clip has no frames.
                lambda done, total: self.update(job_id, progress=done / max(1, total)),
            )
        except Exception as err:  # noqa: BLE001 - the failure is the job's result
            # A bare `except Exception` is usually a smell, but here the job's
            # failure *is* the product: it has to become a FAILED status the
            # client can poll, not a traceback nobody sees. logger.exception
            # records the full traceback; the client gets the short form.
            logger.exception("Render job %s failed", job_id)
            self.update(job_id, status=JobStatus.FAILED, error=f"{type(err).__name__}: {err}")
            return
        self.update(job_id, status=JobStatus.COMPLETED, progress=1.0, result=result)


class CeleryJobQueue:
    """
    Publishes validated jobs to the configured Celery broker.

    State lives in Redis instead of a dict, so a separate worker process (or
    several) can pick jobs up and the API can restart without losing them.
    """

    # Plain class attribute, matching InMemoryJobQueue's @property. A Celery
    # queue always has a worker by definition, so there is nothing to compute.
    executes = True

    def __init__(self, redis_client: Optional[Redis] = None):
        # Accepting an optional client is dependency injection: the tests pass
        # a FakeRedis, so they exercise this logic with no server running
        # (golden rule 6). Defaulting to a real client keeps production simple.
        self._redis = redis_client or Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            # Redis speaks bytes. decode_responses=True hands back str, so
            # json.loads works without a .decode() at every call site.
            decode_responses=True,
        )

    @staticmethod
    def _key(job_id: str) -> str:
        """
        Namespace every key, because Redis is one flat keyspace.

        @staticmethod = no self needed; it is grouped here only because it
        belongs to this class's storage scheme.
        """
        return f"avatar:render-job:{job_id}"

    def enqueue(self, job: AvatarRenderJob, engine: Optional[str] = None) -> QueuedJob:
        # Imported inside the function, not at module top: celery_app imports
        # this module, so a top-level import would be circular. Python caches
        # modules in sys.modules, so this costs nothing after the first call.
        from celery_app import process_render_job

        queued_job = QueuedJob(job=job, engine=engine)
        record = json.dumps(
            {
                # by_alias=True writes the camelCase names the contract
                # declares; mode="json" converts enums and other non-JSON
                # types into primitives so json.dumps cannot fail on them.
                "job": job.model_dump(by_alias=True, mode="json"),
                # .value unwraps the Enum into a plain string for storage.
                "status": queued_job.status.value,
                "engine": engine,
            }
        )
        # SET NX claims the id atomically: two requests cannot both pass a
        # separate exists-then-set check.
        if not self._redis.set(self._key(job.job_id), record, nx=True):
            raise ValueError(f"jobId already exists: {job.job_id}")
        try:
            # .delay() is Celery's "send this task to the broker" shorthand.
            # The payload is re-dumped rather than reusing `record` because the
            # task wants the job alone, not the status wrapper.
            process_render_job.delay(job.model_dump(by_alias=True, mode="json"), engine)
        except Exception:
            # Never published: do not leave a record that says QUEUED forever.
            # Compensating for a failed second step after a successful first
            # one - the two-step write is not a transaction, so the rollback
            # has to be explicit. `raise` re-raises the original error with its
            # traceback intact.
            self._redis.delete(self._key(job.job_id))
            raise
        return queued_job

    def update(self, job_id: str, **changes) -> None:
        """Merge ``status`` / ``progress`` / ``result`` / ``error`` into the stored record."""
        stored_job = self._redis.get(self._key(job_id))
        if stored_job is None:
            # Silently ignore an update for a job that no longer exists: it
            # expired or was deleted, and a late progress report is not an error.
            return
        # Same narrowing as get(): redis-py types a sync client's reply as
        # possibly awaitable because one class serves sync and async use.
        assert isinstance(stored_job, (str, bytes))
        payload = json.loads(stored_job)
        for name, value in changes.items():
            # JobStatus is not JSON-serialisable, so unwrap it; everything else
            # passes through untouched.
            payload[name] = value.value if isinstance(value, JobStatus) else value
        # Read-modify-write without a lock: two workers updating the same job
        # could lose one update. Acceptable because exactly one worker owns a
        # job at a time; it would need a WATCH/MULTI transaction otherwise.
        self._redis.set(self._key(job_id), json.dumps(payload))

    def get(self, job_id: str) -> Optional[QueuedJob]:
        stored_job = self._redis.get(self._key(job_id))
        if stored_job is None:
            return None
        # Narrows the type for the checker (redis-py's stub returns Any).
        assert isinstance(stored_job, (str, bytes))
        payload = json.loads(stored_job)
        return QueuedJob(
            # Re-validated on the way out, not trusted. The stored JSON came
            # from this process, but it round-tripped through an external
            # store, so it is untrusted input again - and a schema change would
            # otherwise surface as a confusing AttributeError much later.
            job=AvatarRenderJob.model_validate(payload["job"]),
            status=JobStatus(payload["status"]),
            # .get() for fields absent on an older record; [] for the two that
            # are always written. The distinction is forward compatibility.
            engine=payload.get("engine"),
            # `or 0.0` also catches a stored null, which float(None) would not.
            progress=float(payload.get("progress") or 0.0),
            result=payload.get("result"),
            error=payload.get("error"),
        )
