"""
In-process tracker for slow one-off jobs that are not renders (avatar generation).

Stable Diffusion takes seconds on a GPU and minutes on a CPU, so the HTTP
request cannot wait for it: it returns a ``taskId`` at once and the client
polls. One worker thread runs the jobs, because the diffusion model is a
heavy model and the GPU holds one at a time (golden rule 5).

State lives in this process, so a restart forgets pending generations. The
faces it already registered are on disk and unaffected. Persistence for every
job type is T6.1.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from request_context import run_in_context

logger = logging.getLogger(__name__)

# Finished jobs kept for polling; the oldest are dropped past this many.
MAX_KEPT = 200


class GenerationJobs:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="generate")

    def submit(self, work: Callable[[], Dict[str, Any]]) -> str:
        """Run ``work`` in the background; return the id to poll."""
        task_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[task_id] = {"taskId": task_id, "status": "QUEUED"}
            while len(self._jobs) > MAX_KEPT:  # dicts keep insertion order
                self._jobs.pop(next(iter(self._jobs)))
        # The worker thread logs under the id of the request that queued the job.
        self._executor.submit(run_in_context(self._run), task_id, work)
        return task_id

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(task_id)
            return dict(job) if job is not None else None

    def _set(self, task_id: str, **fields: Any) -> None:
        with self._lock:
            if task_id in self._jobs:
                self._jobs[task_id].update(fields)

    def _run(self, task_id: str, work: Callable[[], Dict[str, Any]]) -> None:
        self._set(task_id, status="PROCESSING")
        try:
            result = work()
        except Exception as err:  # noqa: BLE001 - the failure is the job's result
            logger.exception("Generation job %s failed", task_id)
            self._set(task_id, status="FAILED", error=f"{type(err).__name__}: {err}")
            return
        self._set(task_id, status="COMPLETED", result=result)
