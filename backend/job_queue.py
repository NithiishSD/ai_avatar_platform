import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional

from contracts import AvatarRenderJob, JobStatus
from redis import Redis

logger = logging.getLogger(__name__)

# Renders a job and returns the result payload. ``report(done, total)`` is
# called as frames are written.
RenderRunner = Callable[[AvatarRenderJob, Optional[str], Callable[[int, int], None]], dict]


@dataclass(frozen=True)
class QueuedJob:
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

    def __init__(self, runner: Optional[RenderRunner] = None):
        self._jobs: Dict[str, QueuedJob] = {}
        self._lock = threading.Lock()
        self._runner = runner
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")
            if runner is not None
            else None
        )

    @property
    def executes(self) -> bool:
        """Whether an enqueued job will actually be rendered."""
        return self._runner is not None

    def enqueue(self, job: AvatarRenderJob, engine: Optional[str] = None) -> QueuedJob:
        with self._lock:
            if job.job_id in self._jobs:
                raise ValueError(f"jobId already exists: {job.job_id}")
            queued_job = QueuedJob(job=job, engine=engine)
            self._jobs[job.job_id] = queued_job
        if self._executor is not None:
            self._executor.submit(self._run, job.job_id)
        return queued_job

    def get(self, job_id: str) -> Optional[QueuedJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes) -> None:
        with self._lock:
            current = self._jobs.get(job_id)
            if current is not None:
                self._jobs[job_id] = replace(current, **changes)

    def _run(self, job_id: str) -> None:
        queued = self.get(job_id)
        if queued is None or self._runner is None:
            return
        self.update(job_id, status=JobStatus.PROCESSING)
        try:
            result = self._runner(
                queued.job,
                queued.engine,
                lambda done, total: self.update(job_id, progress=done / max(1, total)),
            )
        except Exception as err:  # noqa: BLE001 - the failure is the job's result
            logger.exception("Render job %s failed", job_id)
            self.update(job_id, status=JobStatus.FAILED, error=f"{type(err).__name__}: {err}")
            return
        self.update(job_id, status=JobStatus.COMPLETED, progress=1.0, result=result)


class CeleryJobQueue:
    """Publishes validated jobs to the configured Celery broker."""

    executes = True

    def __init__(self, redis_client: Optional[Redis] = None):
        self._redis = redis_client or Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )

    @staticmethod
    def _key(job_id: str) -> str:
        return f"avatar:render-job:{job_id}"

    def enqueue(self, job: AvatarRenderJob, engine: Optional[str] = None) -> QueuedJob:
        from celery_app import process_render_job

        queued_job = QueuedJob(job=job, engine=engine)
        record = json.dumps(
            {
                "job": job.model_dump(by_alias=True, mode="json"),
                "status": queued_job.status.value,
                "engine": engine,
            }
        )
        # SET NX claims the id atomically: two requests cannot both pass a
        # separate exists-then-set check.
        if not self._redis.set(self._key(job.job_id), record, nx=True):
            raise ValueError(f"jobId already exists: {job.job_id}")
        try:
            process_render_job.delay(job.model_dump(by_alias=True, mode="json"), engine)
        except Exception:
            # Never published: do not leave a record that says QUEUED forever.
            self._redis.delete(self._key(job.job_id))
            raise
        return queued_job

    def update(self, job_id: str, **changes) -> None:
        """Merge ``status`` / ``progress`` / ``result`` / ``error`` into the stored record."""
        stored_job = self._redis.get(self._key(job_id))
        if stored_job is None:
            return
        payload = json.loads(stored_job)
        for name, value in changes.items():
            payload[name] = value.value if isinstance(value, JobStatus) else value
        self._redis.set(self._key(job_id), json.dumps(payload))

    def get(self, job_id: str) -> Optional[QueuedJob]:
        stored_job = self._redis.get(self._key(job_id))
        if stored_job is None:
            return None
        assert isinstance(stored_job, (str, bytes))
        payload = json.loads(stored_job)
        return QueuedJob(
            job=AvatarRenderJob.model_validate(payload["job"]),
            status=JobStatus(payload["status"]),
            engine=payload.get("engine"),
            progress=float(payload.get("progress") or 0.0),
            result=payload.get("result"),
            error=payload.get("error"),
        )
