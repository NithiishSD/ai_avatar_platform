"""Job persistence (R-21): jobs survive an API restart."""

import tempfile
import threading
import unittest
from pathlib import Path

from contracts import AvatarRenderJob, JobStatus
from job_queue import InMemoryJobQueue
from job_store import JobStore
from test_contracts import VALID_JOB


def job(job_id="J1") -> AvatarRenderJob:
    return AvatarRenderJob.model_validate({**VALID_JOB, "jobId": job_id})


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "nested" / "jobs.sqlite"

    def test_round_trip_upsert_and_order(self):
        store = JobStore(self.path)
        store.put("render", "a", {"v": 1})
        store.put("render", "b", {"v": 2})
        store.put("render", "a", {"v": 3})  # an update, not a second row
        store.put("other", "a", {"v": 9})
        self.assertEqual(store.get("render", "a"), {"v": 3})
        self.assertEqual([i for i, _ in store.all("render")], ["a", "b"])  # order first submitted: FIFO for re-runs
        self.assertIsNone(store.get("render", "zzz"))

    def test_a_second_connection_after_close_sees_the_data(self):
        first = JobStore(self.path)
        first.put("render", "a", {"v": 1})
        first.close()
        self.assertEqual(JobStore(self.path).get("render", "a"), {"v": 1})


class RestartTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "jobs.sqlite"

    def restart(self, runner=None) -> InMemoryJobQueue:
        return InMemoryJobQueue(runner=runner, store=JobStore(self.path))

    def test_a_finished_job_and_its_result_survive_a_restart(self):
        queue = self.restart(runner=lambda j, e, report: {"videoUrl": "/outputs/renders/J1.mp4", "engine": e})
        queue.enqueue(job(), "blendshape")
        queue._executor.shutdown(wait=True)

        after = self.restart()  # a new process: new queue, same file
        recovered = after.get("J1")
        self.assertEqual(recovered.status, JobStatus.COMPLETED)
        self.assertEqual(recovered.result["videoUrl"], "/outputs/renders/J1.mp4")
        self.assertEqual(recovered.progress, 1.0)
        self.assertEqual(recovered.job.job_id, "J1")

    def test_a_job_that_was_mid_render_is_failed_with_the_reason_not_left_processing(self):
        started, release = threading.Event(), threading.Event()

        def runner(j, e, report):
            started.set()
            release.wait(10)
            return {}

        before = self.restart(runner=runner)
        before.enqueue(job(), None)
        self.assertTrue(started.wait(5))
        try:
            after = self.restart()  # the "restart" happens while the first process is mid-render
            recovered = after.get("J1")
            self.assertEqual(recovered.status, JobStatus.FAILED)
            self.assertIn("restart", recovered.error)
            # ...and the correction was itself saved, so a second restart agrees.
            self.assertEqual(self.restart().get("J1").status, JobStatus.FAILED)
        finally:
            release.set()
            before._executor.shutdown(wait=True)

    def test_a_job_that_never_started_is_run_after_the_restart(self):
        self.restart().enqueue(job("J2"), "blendshape")  # recorded by a queue with no runner
        ran = []
        after = self.restart(runner=lambda j, e, report: ran.append(j.job_id) or {"ok": True})
        after._executor.shutdown(wait=True)
        self.assertEqual(ran, ["J2"])
        self.assertEqual(after.get("J2").status, JobStatus.COMPLETED)

    def test_a_recorded_job_without_a_runner_stays_queued_for_a_worker(self):
        self.restart().enqueue(job("J3"), None)
        self.assertEqual(self.restart().get("J3").status, JobStatus.QUEUED)

    def test_job_ids_stay_unique_across_a_restart(self):
        self.restart().enqueue(job(), None)
        with self.assertRaises(ValueError):
            self.restart().enqueue(job(), None)

    def test_failed_jobs_keep_their_error(self):
        def boom(j, e, report):
            raise RuntimeError("no GPU")

        queue = self.restart(runner=boom)
        queue.enqueue(job(), None)
        queue._executor.shutdown(wait=True)
        recovered = self.restart().get("J1")
        self.assertEqual(recovered.status, JobStatus.FAILED)
        self.assertIn("no GPU", recovered.error)


if __name__ == "__main__":
    unittest.main()


class GenerationRestartTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "jobs.sqlite"

    def test_finished_generation_survives_and_unfinished_is_failed_with_the_reason(self):
        from generation_jobs import GenerationJobs

        jobs = GenerationJobs(JobStore(self.path))
        done = jobs.submit(lambda: {"avatarId": "face1"})
        jobs._executor.shutdown(wait=True)
        gate = threading.Event()
        jobs2 = GenerationJobs(JobStore(self.path))
        running = jobs2.submit(lambda: gate.wait(10) and {})  # still working when the "restart" happens
        try:
            after = GenerationJobs(JobStore(self.path))
            self.assertEqual(after.get(done)["status"], "COMPLETED")
            self.assertEqual(after.get(done)["result"], {"avatarId": "face1"})
            self.assertEqual(after.get(running)["status"], "FAILED")
            self.assertIn("restart", after.get(running)["error"])
        finally:
            gate.set()
            jobs2._executor.shutdown(wait=True)
