"""
In-process tracker for slow one-off jobs that are not renders (avatar generation).

Stable Diffusion takes seconds on a GPU and minutes on a CPU, so the HTTP
request cannot wait for it: it returns a ``taskId`` at once and the client
polls. One worker thread runs the jobs, because the diffusion model is a
heavy model and the GPU holds one at a time (golden rule 5).

With a ``JobStore`` every state change is also written to SQLite, so a client
polling after an API restart still gets an answer. A job that had not finished
cannot be resumed (its work is a closure that died with the process), so it is
reported FAILED with that reason; the faces already registered are on disk.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from job_store import JobStore
from request_context import run_in_context

logger = logging.getLogger(__name__)

# Finished jobs kept for polling; the oldest are dropped past this many.
MAX_KEPT = 200


class GenerationJobs:
    KIND = "generate"

    def __init__(self, store: Optional[JobStore] = None) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._store = store
        if store is not None:
            for task_id, record in store.all(self.KIND):
                if record["status"] in ("QUEUED", "PROCESSING"):
                    record = {**record, "status": "FAILED", "error": "interrupted by a server restart; submit it again"}
                    store.put(self.KIND, task_id, record)
                self._jobs[task_id] = record
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="generate")

    def submit(self, work: Callable[[], Dict[str, Any]]) -> str:
        """Run ``work`` in the background; return the id to poll."""
        task_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[task_id] = {"taskId": task_id, "status": "QUEUED"}
            self._persist(task_id)
            while len(self._jobs) > MAX_KEPT:  # dicts keep insertion order
                self._jobs.pop(next(iter(self._jobs)))
        # The worker thread logs under the id of the request that queued the job.
        self._executor.submit(run_in_context(self._run), task_id, work)
        return task_id

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(task_id)
            return dict(job) if job is not None else None

    def _persist(self, task_id: str) -> None:
        if self._store is not None:
            self._store.put(self.KIND, task_id, self._jobs[task_id])

    def _set(self, task_id: str, **fields: Any) -> None:
        with self._lock:
            if task_id in self._jobs:
                self._jobs[task_id].update(fields)
                self._persist(task_id)

    def _run(self, task_id: str, work: Callable[[], Dict[str, Any]]) -> None:
        self._set(task_id, status="PROCESSING")
        try:
            result = work()
        except Exception as err:  # noqa: BLE001 - the failure is the job's result
            logger.exception("Generation job %s failed", task_id)
            self._set(task_id, status="FAILED", error=f"{type(err).__name__}: {err}")
            return
        self._set(task_id, status="COMPLETED", result=result)
