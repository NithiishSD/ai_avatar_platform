import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parents[1] / "backend"
if not backend_dir.exists():
    backend_dir = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(backend_dir))

from app import app
from job_queue import CeleryJobQueue, InMemoryJobQueue
from celery_app import celery
from contracts import AudioSynthesisRequest, AvatarRenderJob
from test_contracts import VALID_JOB
from test_voice_engine import reference, weights_on_disk


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


class RenderJobApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        import app as app_module
        app_module.job_queue = InMemoryJobQueue()

    def test_create_render_job_returns_queued_status(self):
        response = self.client.post("/api/v1/avatar/render-job", json=VALID_JOB)

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json(), {"jobId": "AVT-9821-X", "status": "QUEUED"})

    def test_health_reports_service_and_queue_backend(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["queueBackend"], "in_memory")
        # Phase 3 added a capability block; the original two keys must remain.
        self.assertIn("mms-tts", payload["capabilities"]["models"])
        self.assertGreater(payload["capabilities"]["languages"], 1000)

    def test_cors_headers_allow_vite_dev_server(self):
        response = self.client.get("/health", headers={"Origin": "http://localhost:5173"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")

    def test_status_endpoint_returns_queued_job(self):
        self.client.post("/api/v1/avatar/render-job", json=VALID_JOB)

        response = self.client.get("/api/v1/avatar/render-job/AVT-9821-X")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "QUEUED")

    def test_invalid_payload_is_rejected_before_enqueue(self):
        invalid_job = {**VALID_JOB, "targetFps": 0}

        response = self.client.post("/api/v1/avatar/render-job", json=invalid_job)

        self.assertEqual(response.status_code, 422)

    def test_duplicate_job_is_conflict(self):
        self.client.post("/api/v1/avatar/render-job", json=VALID_JOB)

        response = self.client.post("/api/v1/avatar/render-job", json=VALID_JOB)

        self.assertEqual(response.status_code, 409)

    def test_unknown_job_returns_not_found(self):
        response = self.client.get("/api/v1/avatar/render-job/unknown")

        self.assertEqual(response.status_code, 404)

    def test_celery_synthesis_task_invokes_voice_engine(self):
        request = AudioSynthesisRequest(text="Hello from the worker")
        expected_result = {
            "output_path": "/tmp/speech.wav",
            "sample_rate": 24000,
            "duration_seconds": 1.2,
            "latency_ms": 42.0,
            "model": "kokoro",
            "mode": "fast",
        }

        class FakeVoiceEngine:
            def synthesize(self, **kwargs):
                self.arguments = kwargs
                return SimpleNamespace(**expected_result)

        import celery_app

        # get_router() caches one router per process. Without resetting the
        # cache for this test, the fake built here became *the* router for
        # every later test in the run.
        with patch("celery_app.VoiceEngineRouter", return_value=FakeVoiceEngine()), \
             patch.object(celery_app, "_router", None):
            result = celery_app.synthesize_audio.run(request.model_dump(mode="json", by_alias=True))

        self.assertEqual(result, expected_result)

    def test_synthesis_endpoint_queues_typed_request(self):
        fake_task = type("Task", (), {"id": "TASK-123"})()
        request = {"text": "Hello from the API", "mode": "fast", "language": "en"}

        with patch("app.synthesize_audio.delay", return_value=fake_task) as delay, \
             patch("model_registry.audit_model_weights", return_value=weights_on_disk()):
            response = self.client.post("/api/v1/audio/synthesize", json=request)

        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(body["taskId"], "TASK-123")
        self.assertEqual(body["status"], "QUEUED")
        # modelUsed is None when task is still in QUEUED state
        self.assertIsNone(body.get("modelUsed"))
        delay.assert_called_once()

    def test_engine_without_weights_is_503_with_the_fix_and_nothing_queued(self):
        request = {"text": "Hello", "mode": "high_quality", "language": "en"}
        with patch("app.synthesize_audio.delay") as delay, \
             patch("model_registry.audit_model_weights", return_value=weights_on_disk("higgs-tts-2")):
            response = self.client.post("/api/v1/audio/synthesize", json=request)
        self.assertEqual(response.status_code, 503)
        self.assertIn("scripts/fetch_models.py --only higgs-tts-2", response.json()["detail"])
        delay.assert_not_called()

    def test_cloning_an_unconsented_voice_is_403_and_nothing_queued(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            request = {"text": "Hello", "mode": "clone", "speakerWav": reference(tmp, "unknown.wav")}
            with patch("app.synthesize_audio.delay") as delay, \
                 patch("model_registry.audit_model_weights", return_value=weights_on_disk()):
                response = self.client.post("/api/v1/audio/synthesize", json=request)
        self.assertEqual(response.status_code, 403)
        self.assertIn("no provenance record", response.json()["detail"])
        delay.assert_not_called()

    def test_a_consented_voice_is_queued(self):
        import tempfile

        fake_task = type("Task", (), {"id": "TASK-CLONE"})()
        with tempfile.TemporaryDirectory() as tmp:
            wav = reference(tmp, "lj.wav", source="human", speaker="LJ",
                            licence="public domain", consent_basis="open-licence")
            request = {"text": "Hello", "mode": "clone", "speakerWav": wav}
            with patch("app.synthesize_audio.delay", return_value=fake_task) as delay, \
                 patch("model_registry.audit_model_weights", return_value=weights_on_disk()):
                response = self.client.post("/api/v1/audio/synthesize", json=request)
        self.assertEqual(response.status_code, 202)
        delay.assert_called_once()

    def test_clone_engine_is_passed_to_the_task(self):
        import tempfile

        fake_task = type("Task", (), {"id": "TASK-OV"})()
        with tempfile.TemporaryDirectory() as tmp:
            wav = reference(tmp, "lj.wav", source="human", speaker="LJ",
                            licence="public domain", consent_basis="open-licence")
            request = {"text": "Hello", "mode": "clone", "speakerWav": wav, "cloneEngine": "openvoice-v2"}
            with patch("app.synthesize_audio.delay", return_value=fake_task) as delay, \
                 patch("model_registry.audit_model_weights", return_value=weights_on_disk()):
                response = self.client.post("/api/v1/audio/synthesize", json=request)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(delay.call_args.args[0]["cloneEngine"], "openvoice-v2")

    def test_an_unknown_clone_engine_is_rejected_at_the_edge(self):
        request = {"text": "Hello", "mode": "clone", "speakerWav": "x.wav", "cloneEngine": "bark"}
        with patch("app.synthesize_audio.delay") as delay:
            response = self.client.post("/api/v1/audio/synthesize", json=request)
        self.assertEqual(response.status_code, 422)
        delay.assert_not_called()

    def test_a_missing_reference_file_is_400(self):
        request = {"text": "Hello", "mode": "clone", "speakerWav": "/nonexistent/voice.wav"}
        with patch("app.synthesize_audio.delay") as delay, \
             patch("model_registry.audit_model_weights", return_value=weights_on_disk()):
            response = self.client.post("/api/v1/audio/synthesize", json=request)
        self.assertEqual(response.status_code, 400)
        delay.assert_not_called()

    def test_unsupported_multilingual_language_is_400_before_queueing(self):
        request = {"text": "Hello", "mode": "multilingual", "language": "zzz"}
        with patch("app.synthesize_audio.delay") as delay, \
             patch("model_registry.audit_model_weights", return_value=weights_on_disk()):
            response = self.client.post("/api/v1/audio/synthesize", json=request)
        self.assertEqual(response.status_code, 400)
        self.assertIn("requires an MMS-TTS checkpoint", response.json()["detail"])
        delay.assert_not_called()

    def test_synthesis_status_reads_celery_state(self):
        pending_task = type("Task", (), {"state": "PENDING", "result": None})()

        with patch("app.celery.AsyncResult", return_value=pending_task):
            response = self.client.get("/api/v1/audio/synthesize/TASK-123")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["taskId"], "TASK-123")
        self.assertEqual(body["status"], "QUEUED")
        self.assertIsNone(body.get("modelUsed"))

    def test_celery_queue_persists_job_for_status_lookup(self):
        queue = CeleryJobQueue(redis_client=FakeRedis())
        job = __import__("contracts").AvatarRenderJob.model_validate(VALID_JOB)

        with patch("celery_app.process_render_job.delay") as delay:
            queue.enqueue(job)

        restored_job = queue.get("AVT-9821-X")
        self.assertEqual(restored_job.status.value, "QUEUED")
        self.assertEqual(restored_job.job.avatar_id, "AVATAR_FEMALE_04")
        delay.assert_called_once()

    def test_celery_queue_publishes_validated_payload(self):
        # Restore rather than hard-set False on the way out: this is the real
        # app config, and leaving eager off leaked into every later test.
        previous = celery.conf.task_always_eager
        celery.conf.task_always_eager = True
        self.addCleanup(setattr, celery.conf, "task_always_eager", previous)
        queue = CeleryJobQueue(redis_client=FakeRedis())

        queued_job = queue.enqueue(AvatarRenderJob.model_validate(VALID_JOB))

        self.assertEqual(queued_job.status.value, "QUEUED")
        self.assertEqual(queue.get("AVT-9821-X").job.job_id, "AVT-9821-X")

    def test_in_memory_mode_never_points_celery_at_redis(self):
        """
        QUEUE_BACKEND=in_memory promises no Redis is needed. Celery reads
        CELERY_BROKER_URL / CELERY_RESULT_BACKEND from the environment and they
        beat the constructor arguments, so in_memory mode used to store its
        eager results in Redis and fail when none was running.
        """
        import importlib

        env = {
            "QUEUE_BACKEND": "in_memory",
            "CELERY_BROKER_URL": "redis://localhost:6379/0",
            "CELERY_RESULT_BACKEND": "redis://localhost:6379/1",
        }
        with patch.dict(os.environ, env, clear=False):
            module = importlib.reload(importlib.import_module("celery_app"))
            try:
                self.assertNotIn("redis", module.celery.conf.broker_url)
                self.assertNotIn("redis", module.celery.conf.result_backend)
                self.assertTrue(module.celery.conf.task_always_eager)
            finally:
                # Leave the shared module as the rest of the suite expects it.
                importlib.reload(module)


if __name__ == "__main__":
    unittest.main()