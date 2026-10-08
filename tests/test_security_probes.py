"""
Attack inputs against the real app (T6.5): each must get the safe answer.

These are the probes from the security review turned into tests. The ones that
found real problems: the audio endpoints accepted any file in the project
(``.env``, source, docs) and leaked absolute server paths in 500 bodies, and
``speakerWav`` told a caller which files exist anywhere on the host (403 for an
existing file, 400 for a missing one).
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import app as app_module
import avatar_store
import provenance
from job_queue import InMemoryJobQueue
from vision_fixtures import FakeFaceEngine, gradient_image


class SecurityProbeCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.inputs, self.outputs = self.root / "inputs", self.root / "outputs"
        self.inputs.mkdir()
        self.outputs.mkdir()
        (self.root / ".env").write_text("SECRET=hunter2\n")
        (self.root / "app_source.py").write_text("print('source')\n")
        for patcher in (
            mock.patch.object(app_module, "project_root", self.root),
            mock.patch.object(app_module, "inputs_dir", self.inputs),
            mock.patch.object(app_module, "outputs_dir", self.outputs),
            mock.patch.object(app_module.security_gate, "inspect", return_value=(True, 0, {})),
            # A queue that executes, so the pre-queue checks run no matter what
            # other test files left on the app (a runner-less queue skips them).
            mock.patch.object(app_module, "job_queue", InMemoryJobQueue(runner=lambda *a: {})),
            mock.patch.object(app_module, "faces", avatar_store.AvatarStore(root=self.inputs / "faces", engine=FakeFaceEngine())),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        app_module.faces.register(gradient_image(512), avatar_id="demo", source=provenance.SYNTHETIC)
        self.client = TestClient(app_module.app, raise_server_exceptions=False)

    def wav(self, folder: Path, name="clip.wav") -> Path:
        path = folder / name
        sf.write(path, np.zeros(2400, dtype=np.float32), 24000)
        return path


class AudioPathTests(SecurityProbeCase):
    def audit(self, raw):
        return self.client.post("/api/v1/audio/quality-audit", json={"audioPath": raw})

    def test_files_elsewhere_in_the_project_look_exactly_like_files_that_do_not_exist(self):
        missing = self.audit("nothing_here.wav")
        self.assertEqual(missing.status_code, 404)
        for raw in (".env", "app_source.py", str(self.root / ".env")):
            reply = self.audit(raw)
            if Path(raw).is_absolute():
                self.assertEqual(reply.status_code, 400, raw)  # an absolute path outside the folders is refused
            else:
                self.assertEqual((reply.status_code, reply.json()["detail"].replace(raw, "X")),
                                 (404, missing.json()["detail"].replace("nothing_here.wav", "X")), raw)

    def test_traversal_and_absolute_paths_outside_the_folders_are_400(self):
        for raw in ("../../../etc/passwd", "/etc/passwd", "inputs/../../etc/hosts"):
            self.assertEqual(self.audit(raw).status_code, 400, raw)
        # This one normalises to a harmless path inside outputs/, so "not found"
        # is the correct, non-leaking answer; what matters is it is not served.
        reply = self.audit("outputs/../.env")
        self.assertIn(reply.status_code, (400, 404))
        self.assertNotIn("hunter2", reply.text)

    def test_a_clip_inside_outputs_still_works_by_bare_name_and_by_prefixed_path(self):
        self.wav(self.outputs)
        with mock.patch.object(app_module.quality_auditor, "audit") as audit:
            audit.return_value.to_dict.return_value = {"mos": 4.0}
            self.assertEqual(self.audit("clip.wav").status_code, 200)
            self.assertEqual(self.audit("outputs/clip.wav").status_code, 200)

    def test_a_file_that_is_not_audio_is_422_and_never_names_a_server_path(self):
        (self.outputs / "junk.wav").write_text("this is text, not audio")
        reply = self.audit("junk.wav")
        self.assertEqual(reply.status_code, 422)
        self.assertNotIn(str(self.root), reply.text)
        self.assertNotIn(str(Path.home()), reply.text)

    def test_an_unexpected_failure_is_a_500_without_a_server_path(self):
        self.wav(self.outputs)
        with mock.patch.object(app_module.quality_auditor, "audit", side_effect=RuntimeError(f"boom at {self.root}/x.py")):
            reply = self.audit("clip.wav")
        self.assertEqual(reply.status_code, 500)
        self.assertNotIn(str(self.root), reply.text)
        self.assertIn("<project>", reply.text)


class SpeakerWavTests(SecurityProbeCase):
    def clone(self, raw):
        return self.client.post("/api/v1/audio/synthesize", json={"text": "hi", "mode": "clone", "speakerWav": raw})

    def test_outside_missing_and_traversal_all_get_the_same_answer(self):
        replies = [self.clone(raw) for raw in ("/etc/passwd", "/definitely/not/there.wav", "../../etc/hostname", str(self.root / ".env"))]
        self.assertEqual({r.status_code for r in replies}, {400})
        self.assertEqual(len({r.json()["detail"] for r in replies}), 1, "the answers differ, so the endpoint is an existence oracle")

    def test_a_recording_inside_inputs_without_consent_is_403_not_a_file_probe(self):
        wav = self.wav(self.inputs, "voice.wav")
        with mock.patch("model_registry.audit_model_weights", return_value=__import__("test_voice_engine").weights_on_disk()):
            reply = self.clone(str(wav))
        self.assertEqual(reply.status_code, 403)
        self.assertIn("no provenance record", reply.json()["detail"])


class OtherControlsTests(SecurityProbeCase):
    def test_static_files_cannot_escape_outputs(self):
        for path in ("/outputs/../.env", "/outputs/%2e%2e/.env", "/outputs/..%2f.env", "/outputs/renders/../../.env"):
            reply = self.client.get(path)
            self.assertNotEqual(reply.status_code, 200, path)
            self.assertNotIn("hunter2", reply.text)

    def test_avatar_ids_cannot_traverse(self):
        for path in ("/api/v1/avatar/faces/..%2f..%2f.env/image", "/api/v1/avatar/faces/../image"):
            self.assertEqual(self.client.get(path).status_code, 404, path)
        self.assertEqual(self.client.delete("/api/v1/avatar/faces/..%2fdemo").status_code, 404)

    def test_a_render_job_cannot_make_the_worker_fetch_or_read_foreign_urls(self):
        job = {"jobId": "sec", "avatarId": "demo", "sampleRate": 24000, "durationSeconds": 1,
               "phonemeTimestamps": [{"phoneme": "AA", "viseme": "viseme_aa", "startMs": 0, "endMs": 400}],
               "emotionVector": {"happy": 0, "neutral": 1, "eyeblinkRate": 1}, "renderQuality": "PREVIEW", "targetFps": 25}
        for url in ("http://169.254.169.254/latest/meta-data/", "file:///etc/passwd", "s3://b/a.wav", "http://x/outputs/../.env"):
            reply = self.client.post("/api/v1/avatar/render-job", json={**job, "audioUrl": url})
            self.assertEqual(reply.status_code, 400, url)

    def test_a_page_on_another_origin_gets_no_cors_permission(self):
        evil = self.client.get("/health", headers={"Origin": "https://evil.example"})
        self.assertNotIn("access-control-allow-origin", evil.headers)
        local = self.client.get("/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(local.headers.get("access-control-allow-origin"), "http://localhost:5173")

    def test_an_oversize_upload_is_413_before_it_is_decoded(self):
        big = b"\x89PNG" + b"0" * (avatar_store.MAX_UPLOAD_BYTES + 1)
        reply = self.client.post("/api/v1/avatar/faces", files={"file": ("a.png", big, "image/png")},
                                 data={"avatarId": "big", "subject": "x", "consentBasis": "subject-provided"})
        self.assertEqual(reply.status_code, 413)

    def test_health_never_contains_a_key(self):
        text = self.client.get("/health").text
        self.assertNotIn("hunter2", text)
        self.assertIn("configuredKeys", text)  # a count, not the keys


if __name__ == "__main__":
    unittest.main()
