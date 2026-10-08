"""
Concurrent submissions to the queue (N-08): no job lost, none run twice, ids stay unique.

The race being guarded is check-then-act: ``enqueue`` asks "is this id taken?"
and then records it. Without the lock two threads can both see "free". The
duplicate-id test hammers exactly that gap with a tiny thread-switch interval,
and was shown to fail when the lock is removed (see docs/12-PROGRESS.md).
"""

import sys
import threading
import unittest
from collections import Counter

from contracts import AvatarRenderJob, JobStatus
from job_queue import InMemoryJobQueue
from test_contracts import VALID_JOB


def job(job_id: str) -> AvatarRenderJob:
    return AvatarRenderJob.model_validate({**VALID_JOB, "jobId": job_id})


class QueueConcurrencyTests(unittest.TestCase):
    def setUp(self):
        # Force the interpreter to switch threads as often as it can, so a missing
        # lock shows up as a failure instead of hiding behind lucky scheduling.
        self.old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        self.addCleanup(sys.setswitchinterval, self.old_interval)

    def hammer(self, queue, ids):
        """Enqueue ``ids`` from one thread each, released together. Returns (accepted, refused)."""
        barrier = threading.Barrier(len(ids))
        accepted, refused = [], []
        lock = threading.Lock()

        def submit(job_id):
            barrier.wait()
            try:
                queue.enqueue(job(job_id), "blendshape")
                outcome = accepted
            except ValueError:
                outcome = refused
            with lock:
                outcome.append(job_id)

        threads = [threading.Thread(target=submit, args=(i,)) for i in ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return accepted, refused

    def test_sixty_concurrent_distinct_jobs_all_run_exactly_once(self):
        runs = Counter()
        runs_lock = threading.Lock()

        def runner(j, engine, report):
            with runs_lock:
                runs[j.job_id] += 1
            return {"engine": engine}

        queue = InMemoryJobQueue(runner=runner)
        ids = [f"J{i}" for i in range(60)]
        accepted, refused = self.hammer(queue, ids)
        queue._executor.shutdown(wait=True)

        self.assertEqual((len(accepted), len(refused)), (60, 0))
        self.assertEqual(set(runs), set(ids))                     # nothing lost
        self.assertTrue(all(n == 1 for n in runs.values()), runs)  # nothing run twice
        self.assertTrue(all(queue.get(i).status == JobStatus.COMPLETED for i in ids))

    def test_the_same_id_submitted_by_many_threads_is_accepted_exactly_once(self):
        runs = []
        queue = InMemoryJobQueue(runner=lambda j, e, r: runs.append(j.job_id) or {})
        accepted, refused = self.hammer(queue, ["SAME"] * 64)
        queue._executor.shutdown(wait=True)
        self.assertEqual(len(accepted), 1, "check-then-act race: the id was accepted more than once")
        self.assertEqual(len(refused), 63)
        self.assertEqual(runs, ["SAME"])

    def test_a_mixed_burst_keeps_every_distinct_job_and_refuses_every_repeat(self):
        queue = InMemoryJobQueue(runner=lambda j, e, r: {})
        ids = [f"M{i % 50}" for i in range(80)]  # 50 distinct ids, 30 repeats
        accepted, refused = self.hammer(queue, ids)
        queue._executor.shutdown(wait=True)
        self.assertEqual(len(accepted), 50)
        self.assertEqual(len(set(accepted)), 50)
        self.assertEqual(len(refused), 30)

    def test_status_updates_from_many_threads_are_not_lost(self):
        queue = InMemoryJobQueue()
        queue.enqueue(job("U1"), None)
        threads = [threading.Thread(target=lambda n=n: queue.update("U1", progress=n / 100)) for n in range(1, 101)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertIn(round(queue.get("U1").progress, 2), {n / 100 for n in range(1, 101)})  # a whole value, never torn


if __name__ == "__main__":
    unittest.main()
