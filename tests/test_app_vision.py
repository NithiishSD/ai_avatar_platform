"""API routes for avatar faces, face analysis, render jobs and the sync metric."""

import io
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app as app_module
import avatar_store
import provenance
import render_engine
from app import app
from avatar_store import AvatarStore
from job_queue import InMemoryJobQueue
from vision_fixtures import FakeFaceEngine, gradient_image, synthetic_analysis

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def png(size=128) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(gradient_image(size)).save(buffer, format="PNG")
    return buffer.getvalue()


class VisionApiCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.outputs = self.tmp / "outputs"
        self.outputs.mkdir()
        (self.tmp / "inputs").mkdir()

        self.engine = FakeFaceEngine()
        self.store = AvatarStore(root=self.tmp / "inputs" / "faces", engine=self.engine)
        patches = [
            # These tests poll job status; the rate limiter has its own tests.
            mock.patch.object(app_module.security_gate, "inspect", return_value=(True, 0, {})),
            mock.patch.object(app_module, "faces", self.store),
            # render_engine builds its own AvatarStore() from this default
            # root. Without the patch it silently read the developer's real
            # inputs/faces, so the "real render" test passed locally on a real
            # face and failed anywhere that face did not exist (CI).
            mock.patch.object(avatar_store, "FACES_DIR", self.tmp / "inputs" / "faces"),
            mock.patch.object(render_engine, "OUTPUTS_DIR", self.outputs),
            mock.patch.object(render_engine, "INPUTS_DIR", self.tmp / "inputs"),
            mock.patch.object(render_engine, "RENDERS_DIR", self.outputs / "renders"),
            mock.patch("render_engine.shared_face_engine", return_value=self.engine),
            mock.patch("face_engine.shared_face_engine", return_value=self.engine),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def add_synthetic(self, avatar_id="demo", size=512):
        self.store.register(gradient_image(size), avatar_id=avatar_id, source=provenance.SYNTHETIC)

    def upload(self, **form):
        data = {"avatarId": "alice", "subject": "Alice Example", "consentBasis": "subject-provided"}
        data.update(form)
        return self.client.post(
            "/api/v1/avatar/faces", files={"file": ("face.png", png(), "image/png")}, data=data
        )


class FaceStoreRouteTests(VisionApiCase):
    def test_list_is_empty_then_shows_registered_faces(self):
        body = self.client.get("/api/v1/avatar/faces").json()
        self.assertEqual(body["avatars"], [])
        self.assertIn("subject-provided", body["consentBases"])
        self.assertTrue(body["renderEngines"]["blendshape"])

        self.add_synthetic()
        avatars = self.client.get("/api/v1/avatar/faces").json()["avatars"]
        self.assertEqual([(a["avatarId"], a["usable"]) for a in avatars], [("demo", True)])

    def test_upload_registers_a_consented_face(self):
        response = self.upload()
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.assertTrue(body["avatar"]["usable"])
        self.assertEqual(body["avatar"]["provenance"]["source"], "human")
        self.assertEqual(body["avatar"]["provenance"]["consentBasis"], "subject-provided")
        self.assertTrue(body["quality"]["passed"])

    def test_upload_without_consent_is_refused_and_stores_nothing(self):
        response = self.upload(consentBasis="")
        self.assertEqual(response.status_code, 400)
        self.assertIn("consent basis", response.json()["detail"])
        self.assertEqual(self.store.list(), [])

    def test_an_upload_cannot_claim_to_be_synthetic(self):
        response = self.upload(source="synthetic", consentBasis="")
        self.assertEqual(response.status_code, 400)

    def test_upload_failing_the_quality_gate_returns_the_reasons(self):
        self.engine.faces = [synthetic_analysis(), synthetic_analysis()]
        response = self.upload()
        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"]
        self.assertEqual(detail["quality"]["errors"][0]["code"], "multiple_faces")
        self.assertIn("exactly one", detail["message"])

    def test_upload_of_a_non_image_and_a_bad_id(self):
        bad = self.client.post(
            "/api/v1/avatar/faces",
            files={"file": ("x.png", b"nope", "image/png")},
            data={"avatarId": "a", "subject": "A", "consentBasis": "written-consent"},
        )
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(self.upload(avatarId="../evil").status_code, 400)

    def test_oversized_upload_is_413(self):
        with mock.patch("avatar_store.MAX_UPLOAD_BYTES", 100):
            self.assertEqual(self.upload().status_code, 413)
            analyze = self.client.post(
                "/api/v1/avatar/face/analyze", files={"file": ("f.png", png(), "image/png")}
            )
            self.assertEqual(analyze.status_code, 413)

    def test_corrupt_image_is_400_not_500(self):
        response = self.client.post(
            "/api/v1/avatar/face/analyze", files={"file": ("f.png", png()[:300], "image/png")}
        )
        self.assertEqual(response.status_code, 400)

    def test_image_route_serves_only_usable_faces(self):
        self.add_synthetic()
        ok = self.client.get("/api/v1/avatar/faces/demo/image")
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.headers["content-type"], "image/png")
        self.assertEqual(self.client.get("/api/v1/avatar/faces/ghost/image").status_code, 404)

        provenance.sidecar_path(self.store.get("demo").path).unlink()
        self.assertEqual(self.client.get("/api/v1/avatar/faces/demo/image").status_code, 403)

    def test_delete(self):
        self.add_synthetic()
        self.assertEqual(self.client.delete("/api/v1/avatar/faces/demo").status_code, 204)
        self.assertEqual(self.client.delete("/api/v1/avatar/faces/demo").status_code, 404)


class FaceAnalyzeRouteTests(VisionApiCase):
    def test_uploaded_photo_is_analysed_without_being_stored(self):
        response = self.client.post("/api/v1/avatar/face/analyze", files={"file": ("f.png", png(), "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["quality"]["passed"])
        self.assertEqual(body["analysis"]["landmarkCount"], 478)
        self.assertEqual(len(body["analysis"]["blendshapes"]), 52)
        self.assertIn("yaw", body["analysis"]["headPose"])
        self.assertNotIn("landmarks", body["analysis"])  # opt-in: keeps the payload small
        self.assertEqual(self.store.list(), [])

    def test_landmarks_on_request_for_a_registered_avatar(self):
        self.add_synthetic()
        response = self.client.post(
            "/api/v1/avatar/face/analyze", data={"avatarId": "demo", "includeLandmarks": "true"}
        )
        self.assertEqual(response.status_code, 200, response.text)
        landmarks = response.json()["analysis"]["landmarks"]
        self.assertEqual(len(landmarks), 478)
        self.assertEqual(set(landmarks[0]), {"x", "y", "z"})

    def test_no_face_is_a_verdict_not_an_error(self):
        self.engine.faces = []
        body = self.client.post(
            "/api/v1/avatar/face/analyze", files={"file": ("f.png", png(), "image/png")}
        ).json()
        self.assertFalse(body["quality"]["passed"])
        self.assertEqual(body["quality"]["errors"][0]["code"], "no_face")
        self.assertIsNone(body["analysis"])

    def test_needs_exactly_one_input(self):
        self.assertEqual(self.client.post("/api/v1/avatar/face/analyze").status_code, 400)
        both = self.client.post(
            "/api/v1/avatar/face/analyze",
            files={"file": ("f.png", png(), "image/png")},
            data={"avatarId": "demo"},
        )
        self.assertEqual(both.status_code, 400)
        self.assertEqual(
            self.client.post("/api/v1/avatar/face/analyze", data={"avatarId": "ghost"}).status_code, 404
        )

    def test_missing_landmarker_is_503_with_the_fix(self):
        from face_engine import FaceEngineUnavailable

        self.engine.check_quality = mock.Mock(
            side_effect=FaceEngineUnavailable("bundle missing. Run scripts/fetch_vision_models.py")
        )
        response = self.client.post(
            "/api/v1/avatar/face/analyze", files={"file": ("f.png", png(), "image/png")}
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("fetch_vision_models.py", response.json()["detail"])


class RenderRouteTests(VisionApiCase):
    def setUp(self):
        super().setUp()
        self.ran = []

        def runner(job, engine, report):
            self.ran.append((job.job_id, engine))
            report(1, 1)
            return {"engine": engine, "videoUrl": f"/outputs/renders/{job.job_id}.mp4", "outputPath": "x.mp4"}

        self.queue = InMemoryJobQueue(runner=runner)
        patcher = mock.patch.object(app_module, "job_queue", self.queue)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.add_synthetic()
        self.wav()

    def wav(self, seconds=1.0):
        import soundfile as sf

        sf.write(self.outputs / "speech.wav", np.zeros(int(24000 * seconds), dtype=np.float32), 24000)

    def payload(self, **overrides):
        body = {
            "jobId": "R1",
            "avatarId": "demo",
            "audioUrl": "http://testserver/outputs/speech.wav",
            "sampleRate": 24000,
            "durationSeconds": 1.0,
            "phonemeTimestamps": [{"phoneme": "AA", "viseme": "viseme_aa", "startMs": 0, "endMs": 500}],
            "emotionVector": {"happy": 0.0, "neutral": 1.0, "eyeblinkRate": 1.0},
            "renderQuality": "PREVIEW",
            "targetFps": 25,
        }
        body.update(overrides)
        return body

    def wait(self, job_id):
        for _ in range(200):
            body = self.client.get(f"/api/v1/avatar/render-job/{job_id}").json()
            if body["status"] in ("COMPLETED", "FAILED"):
                return body
            time.sleep(0.02)
        self.fail("render job did not finish")

    def test_job_is_accepted_then_completes_with_a_video_url(self):
        response = self.client.post("/api/v1/avatar/render-job", json=self.payload())
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["jobId"], "R1")
        self.assertEqual(response.json()["engine"], "blendshape")
        done = self.wait("R1")
        self.assertEqual(done["status"], "COMPLETED")
        self.assertEqual(done["videoUrl"], "/outputs/renders/R1.mp4")
        self.assertEqual(done["progress"], 1.0)
        self.assertEqual(self.ran, [("R1", "blendshape")])

    def test_unrenderable_jobs_are_rejected_up_front_with_the_reason(self):
        cases = [
            (self.payload(avatarId="ghost"), {}, 404, "not registered"),
            (self.payload(audioUrl="s3://bucket/x.wav"), {}, 400, "/outputs/"),
            (self.payload(audioUrl="http://testserver/outputs/missing.wav"), {}, 400, "not found"),
            (self.payload(), {"engine": "sadtalker"}, 400, "unknown render engine"),
        ]
        for body, params, code, fragment in cases:
            response = self.client.post("/api/v1/avatar/render-job", json=body, params=params)
            self.assertEqual(response.status_code, code, response.text)
            self.assertIn(fragment, response.json()["detail"])
        self.assertEqual(self.ran, [])

    def test_job_id_with_a_slash_can_be_polled_and_nul_audio_is_400(self):
        created = self.client.post("/api/v1/avatar/render-job", json=self.payload(jobId="team/clip 1"))
        self.assertEqual(created.status_code, 202, created.text)
        self.assertEqual(self.wait("team/clip 1")["status"], "COMPLETED")
        bad = self.client.post(
            "/api/v1/avatar/render-job",
            json=self.payload(jobId="nul", audioUrl="http://testserver/outputs/a%00b.wav"),
        )
        self.assertEqual(bad.status_code, 400)

    def test_queue_outage_is_503(self):
        with mock.patch.object(self.queue, "enqueue", side_effect=ConnectionError("redis down")):
            response = self.client.post("/api/v1/avatar/render-job", json=self.payload(jobId="down"))
        self.assertEqual(response.status_code, 503)
        self.assertIn("redis down", response.json()["detail"])

    def test_face_without_consent_is_403(self):
        provenance.sidecar_path(self.store.get("demo").path).unlink()
        response = self.client.post("/api/v1/avatar/render-job", json=self.payload())
        self.assertEqual(response.status_code, 403)
        self.assertIn("no provenance record", response.json()["detail"])

    def test_wav2lip_without_weights_is_refused_not_downgraded(self):
        with mock.patch("wav2lip_engine.shared_wav2lip_engine", return_value=mock.Mock(available=False)):
            response = self.client.post(
                "/api/v1/avatar/render-job", json=self.payload(), params={"engine": "wav2lip"}
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("--accept-licence wav2lip", response.json()["detail"])
        self.assertEqual(self.ran, [])

    def test_failed_render_reports_its_error(self):
        def broken(job, engine, report):
            raise RuntimeError("ffmpeg was not found on PATH")

        queue = InMemoryJobQueue(runner=broken)
        with mock.patch.object(app_module, "job_queue", queue):
            with self.assertLogs("job_queue", level="ERROR"):
                self.client.post("/api/v1/avatar/render-job", json=self.payload(jobId="R2"))
                body = self.wait("R2")
        self.assertEqual(body["status"], "FAILED")
        self.assertIn("ffmpeg", body["error"])
        self.assertNotIn("videoUrl", body)

    def test_lipsync_score_route(self):
        self.assertEqual(self.client.post("/api/v1/avatar/render-job/none/lipsync-score").status_code, 404)

        gate_queue = InMemoryJobQueue()  # records only: the job stays QUEUED
        with mock.patch.object(app_module, "job_queue", gate_queue):
            self.client.post("/api/v1/avatar/render-job", json=self.payload(jobId="Q"))
            self.assertEqual(self.client.post("/api/v1/avatar/render-job/Q/lipsync-score").status_code, 409)

        self.client.post("/api/v1/avatar/render-job", json=self.payload())
        self.wait("R1")
        score = mock.Mock()
        score.to_dict.return_value = {"lseC": 6.1, "lseD": 7.2, "method": "syncnet-v2"}
        with mock.patch("lipsync_metric.score_video", return_value=score) as scorer:
            response = self.client.post("/api/v1/avatar/render-job/R1/lipsync-score")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["score"]["method"], "syncnet-v2")
        scorer.assert_called_once_with("x.mp4")

        import lipsync_metric

        with mock.patch(
            "lipsync_metric.score_video",
            side_effect=lipsync_metric.SyncNetUnavailable("fetch_vision_models.py --only syncnet"),
        ):
            missing = self.client.post("/api/v1/avatar/render-job/R1/lipsync-score")
        self.assertEqual(missing.status_code, 503)

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg / ffprobe not installed")
    def test_real_render_through_the_api(self):
        import celery_app

        queue = InMemoryJobQueue(runner=celery_app.run_render)
        with mock.patch.object(app_module, "job_queue", queue):
            self.client.post("/api/v1/avatar/render-job", json=self.payload(jobId="REAL"))
            body = self.wait("REAL")
        self.assertEqual(body["status"], "COMPLETED", body)
        self.assertEqual(body["result"]["frameCount"], 25)
        self.assertTrue((self.outputs / "renders" / "REAL.mp4").is_file())


class HealthTests(unittest.TestCase):
    def test_health_reports_vision_weights_and_render_engines(self):
        body = TestClient(app).get("/health").json()
        keys = {m["key"] for m in body["visionWeights"]["models"]}
        self.assertTrue({"face-landmarker", "wav2lip", "syncnet"} <= keys)
        self.assertEqual(body["render"]["defaultEngine"], "blendshape")
        self.assertTrue(body["render"]["engines"]["blendshape"])
        self.assertEqual(
            body["render"]["engines"]["wav2lip"], "wav2lip" not in body["visionWeights"]["missing"]
        )


if __name__ == "__main__":
    unittest.main()
