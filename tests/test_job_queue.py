"""Render job queues: state transitions, failure capture, and the Celery task."""

import json
import threading
import unittest
from unittest import mock

from pydantic import ValidationError

import celery_app
from contracts import AvatarRenderJob, JobStatus
from job_queue import CeleryJobQueue, InMemoryJobQueue
from test_contracts import VALID_JOB


def job(job_id="J1") -> AvatarRenderJob:
    return AvatarRenderJob.model_validate({**VALID_JOB, "jobId": job_id})


def drain(queue: InMemoryJobQueue) -> None:
    queue._executor.shutdown(wait=True)


class FakeRedis:
    def __init__(self):
        self.values = {}

    def exists(self, key):
        return key in self.values

    def set(self, key, value, nx=False):
        if nx and key in self.values:
            return None
        self.values[key] = value
        return True

    def delete(self, key):
        self.values.pop(key, None)

    def get(self, key):
        return self.values.get(key)


class InMemoryQueueTests(unittest.TestCase):
    def test_without_a_runner_jobs_are_only_recorded(self):
        queue = InMemoryJobQueue()
        self.assertFalse(queue.executes)
        queued = queue.enqueue(job())
        self.assertEqual(queued.status, JobStatus.QUEUED)
        self.assertEqual(queue.get("J1").status, JobStatus.QUEUED)
        self.assertIsNone(queue.get("missing"))

    def test_duplicate_ids_are_refused(self):
        queue = InMemoryJobQueue()
        queue.enqueue(job())
        with self.assertRaises(ValueError):
            queue.enqueue(job())

    def test_runner_completes_a_job_with_its_result_and_engine(self):
        seen = {}

        def runner(render_job, engine, report):
            seen.update(job_id=render_job.job_id, engine=engine)
            report(5, 10)
            return {"videoUrl": "/outputs/renders/J1.mp4", "engine": engine}

        queue = InMemoryJobQueue(runner=runner)
        self.assertTrue(queue.executes)
        queue.enqueue(job(), engine="blendshape")
        drain(queue)
        done = queue.get("J1")
        self.assertEqual(done.status, JobStatus.COMPLETED)
        self.assertEqual(done.progress, 1.0)
        self.assertEqual(done.result["videoUrl"], "/outputs/renders/J1.mp4")
        self.assertEqual(seen, {"job_id": "J1", "engine": "blendshape"})

    def test_enqueue_returns_before_the_render_finishes(self):
        gate = threading.Event()
        queue = InMemoryJobQueue(runner=lambda *_: gate.wait(5) and {})
        queued = queue.enqueue(job())
        self.assertEqual(queued.status, JobStatus.QUEUED)
        self.assertIn(queue.get("J1").status, (JobStatus.QUEUED, JobStatus.PROCESSING))
        gate.set()
        drain(queue)
        self.assertEqual(queue.get("J1").status, JobStatus.COMPLETED)

    def test_progress_is_visible_while_processing(self):
        reached, release = threading.Event(), threading.Event()

        def runner(render_job, engine, report):
            report(3, 12)
            reached.set()
            release.wait(5)
            return {}

        queue = InMemoryJobQueue(runner=runner)
        queue.enqueue(job())
        self.assertTrue(reached.wait(5))
        current = queue.get("J1")
        self.assertEqual(current.status, JobStatus.PROCESSING)
        self.assertAlmostEqual(current.progress, 0.25)
        release.set()
        drain(queue)

    def test_a_failing_render_becomes_a_failed_job_with_the_reason(self):
        def runner(*_):
            raise RuntimeError("avatar 'x' is not registered")

        queue = InMemoryJobQueue(runner=runner)
        with self.assertLogs("job_queue", level="ERROR"):
            queue.enqueue(job("bad"))
            queue.enqueue(job("after"))
            drain(queue)
        failed = queue.get("bad")
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertEqual(failed.error, "RuntimeError: avatar 'x' is not registered")
        self.assertIsNone(failed.result)
        # One failure does not stop the worker.
        self.assertEqual(queue.get("after").status, JobStatus.FAILED)


class CeleryQueueTests(unittest.TestCase):
    def test_engine_travels_with_the_job_and_updates_are_merged(self):
        queue = CeleryJobQueue(redis_client=FakeRedis())
        with mock.patch("celery_app.process_render_job.delay") as delay:
            queue.enqueue(job(), engine="wav2lip")
        self.assertEqual(delay.call_args.args[1], "wav2lip")
        self.assertEqual(queue.get("J1").engine, "wav2lip")

        queue.update("J1", status=JobStatus.PROCESSING, progress=0.5)
        self.assertEqual(queue.get("J1").status, JobStatus.PROCESSING)
        queue.update("J1", status="COMPLETED", progress=1.0, result={"videoUrl": "/outputs/renders/J1.mp4"})
        done = queue.get("J1")
        self.assertEqual(done.status, JobStatus.COMPLETED)
        self.assertEqual(done.result["videoUrl"], "/outputs/renders/J1.mp4")
        queue.update("unknown", status="FAILED")  # no record: a no-op, not a crash


    def test_duplicate_is_refused_and_a_failed_publish_leaves_no_orphan(self):
        redis = FakeRedis()
        queue = CeleryJobQueue(redis_client=redis)
        with mock.patch("celery_app.process_render_job.delay"):
            queue.enqueue(job())
            with self.assertRaises(ValueError):
                queue.enqueue(job())
        with mock.patch("celery_app.process_render_job.delay", side_effect=ConnectionError("broker down")):
            with self.assertRaises(ConnectionError):
                queue.enqueue(job("J2"))
        self.assertIsNone(queue.get("J2"))


class RenderTaskTests(unittest.TestCase):
    def setUp(self):
        self.redis = FakeRedis()
        self.queue = CeleryJobQueue(redis_client=self.redis)
        self.redis.set(
            CeleryJobQueue._key("J1"),
            json.dumps({"job": job().model_dump(by_alias=True, mode="json"), "status": "QUEUED"}),
        )
        patcher = mock.patch("celery_app._render_status_store", return_value=self.queue)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.payload = job().model_dump(by_alias=True, mode="json")

    def test_success_is_recorded_for_the_api(self):
        with mock.patch("celery_app.run_render", return_value={"videoUrl": "/outputs/renders/J1.mp4"}) as run:
            out = celery_app.process_render_job.run(self.payload, "blendshape")
        self.assertEqual(out["status"], "COMPLETED")
        self.assertEqual(run.call_args.args[1], "blendshape")
        stored = self.queue.get("J1")
        self.assertEqual(stored.status, JobStatus.COMPLETED)
        self.assertEqual(stored.result["videoUrl"], "/outputs/renders/J1.mp4")

    def test_failure_is_recorded_not_raised(self):
        with mock.patch("celery_app.run_render", side_effect=PermissionError("no consent record")):
            with self.assertLogs("celery_app", level="ERROR"):
                out = celery_app.process_render_job.run(self.payload)
        self.assertEqual(out["status"], "FAILED")
        stored = self.queue.get("J1")
        self.assertEqual(stored.status, JobStatus.FAILED)
        self.assertIn("no consent record", stored.error)

    def test_a_broken_status_store_is_loud_but_does_not_lose_the_render(self):
        with mock.patch("celery_app._render_status_store", side_effect=ConnectionError("redis down")):
            with mock.patch("celery_app.run_render", return_value={"ok": True}):
                with self.assertLogs("celery_app", level="ERROR") as logs:
                    out = celery_app.process_render_job.run(self.payload)
        self.assertEqual(out["status"], "COMPLETED")
        self.assertTrue(any("redis down" in line for line in logs.output))

    def test_invalid_payload_is_rejected(self):
        # ValidationError specifically: assertRaises(Exception) would also
        # pass on a typo that raised AttributeError, proving nothing.
        with self.assertRaises(ValidationError):
            celery_app.process_render_job.run({**self.payload, "targetFps": 0})


if __name__ == "__main__":
    unittest.main()
