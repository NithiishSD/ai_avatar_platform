"""
"Is this file ours?" (R-36): the verdict for each combination of evidence, the audit-record trace,
and the HTTP endpoint around it (uploads, limits, cleanup).
"""

import copy
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import app as app_module
import audit_log
import authenticity
import manifest
import watermark_engine

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def make_video(path: Path, seconds=1, tone=440):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=128x128:rate=10",
                    "-f", "lavfi", "-i", f"sine=frequency={tone}:duration={seconds}", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(path)], check=True)


MARKED = {"detected": True, "tagBitsMatching": 128, "manifestId": None}
UNMARKED = {"detected": False, "tagBitsMatching": 60, "manifestId": None}


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg / ffprobe not installed")
class AuthenticityCase(unittest.TestCase):
    def setUp(self):
        watermark_engine.reset_key_cache()
        self.addCleanup(watermark_engine.reset_key_cache)
        env = mock.patch.dict(os.environ, {"WATERMARK_KEY": "authenticity-test-key"})
        env.start()
        self.addCleanup(env.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.log = audit_log.AuditLog(":memory:")
        patcher = mock.patch("audit_log.shared_audit", return_value=self.log)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.video = self.dir / "clip.mp4"
        make_video(self.video)
        self.marks = {"video": dict(MARKED), "audio": {"detected": True, "bitsMatching": 16}}
        for name, value in (("_video_mark", lambda path: self.marks["video"]), ("_audio_mark", lambda path, info: self.marks["audio"])):
            p = mock.patch.object(authenticity, name, side_effect=value)
            p.start()
            self.addCleanup(p.stop)

    def issue(self, video=None, manifest_id=bytes(range(16))):
        video = video or self.video
        audio = self.dir / "speech.wav"
        import numpy as np
        import soundfile as sf

        sf.write(audio, np.zeros(2400, dtype=np.float32), 24000)
        with mock.patch.object(manifest, "_avatar_hash", return_value="a" * 64):
            return manifest.build_manifest(
                manifest_id=manifest_id, video_path=video, audio_path=audio,
                avatar={"avatarId": "demo", "provenance": {"source": "synthetic"}},
                render={"width": 128, "height": 128, "fps": 10, "frameCount": 10, "durationSeconds": 1.0, "engine": "blendshape",
                        "quality": "PREVIEW", "background": None, "label": True},
                video_watermark={"applied": True}, models={"speech": "kokoro"})

    def verdict(self, document=None, path=None):
        return authenticity.verify_file(path or self.video, document)["verdict"]


class VerdictTests(AuthenticityCase):
    def test_the_exact_file_with_its_manifest_is_authentic_whatever_the_marks_say(self):
        document = self.issue()
        self.assertEqual(self.verdict(document), "authentic_original")
        self.marks["video"], self.marks["audio"] = dict(UNMARKED), {"detected": False}
        self.assertEqual(self.verdict(document), "authentic_original")  # the signature + hash are the proof

    def test_a_marked_copy_without_a_matching_manifest_is_ours_but_modified(self):
        self.assertEqual(self.verdict(), "ours_modified")
        other = self.dir / "other.mp4"
        make_video(other, tone=880)
        self.assertEqual(self.verdict(self.issue(), path=other), "ours_modified")  # a manifest for different bytes

    def test_the_audio_mark_alone_is_enough_to_call_it_ours(self):
        self.marks["video"] = dict(UNMARKED)
        result = authenticity.verify_file(self.video)
        self.assertEqual(result["verdict"], "ours_modified")
        self.assertIn("audio", " ".join(result["explanation"]))

    def test_no_marks_and_no_manifest_is_no_evidence_and_says_it_proves_nothing(self):
        self.marks["video"], self.marks["audio"] = dict(UNMARKED), {"detected": False}
        result = authenticity.verify_file(self.video)
        self.assertEqual(result["verdict"], "no_evidence")
        self.assertIn("does NOT show", result["meaning"])
        self.assertIn("Nothing here detects fakes", result["limits"])

    def test_a_valid_manifest_for_another_file_with_no_marks_is_reported_as_that(self):
        self.marks["video"], self.marks["audio"] = dict(UNMARKED), {"detected": False}
        other = self.dir / "other.mp4"
        make_video(other, tone=880)
        self.assertEqual(self.verdict(self.issue(), path=other), "manifest_for_another_file")

    def test_an_edited_manifest_is_tampered_even_when_the_watermark_is_present(self):
        document = copy.deepcopy(self.issue())
        document["inputs"]["avatar"]["consentBasis"] = "written-consent"
        self.assertEqual(self.verdict(document), "tampered_manifest")  # not "ours_modified": the forgery is the finding

    def test_a_manifest_signed_by_another_key_is_foreign(self):
        with mock.patch.dict(os.environ, {"WATERMARK_KEY": "some-other-platform"}):
            watermark_engine.reset_key_cache()
            foreign = self.issue()
        watermark_engine.reset_key_cache()
        self.assertEqual(self.verdict(foreign), "foreign_manifest")


class EvidenceTests(AuthenticityCase):
    def test_a_sidecar_manifest_beside_the_file_is_used_and_a_broken_one_is_noted(self):
        Path(str(self.video) + ".manifest.json").write_text(json.dumps(self.issue()))
        result = authenticity.verify_file(self.video)
        self.assertEqual((result["verdict"], result["manifest"]["source"]), ("authentic_original", "sidecar"))
        Path(str(self.video) + ".manifest.json").write_text("{broken")
        broken = authenticity.verify_file(self.video)
        self.assertEqual(broken["verdict"], "ours_modified")
        self.assertIn("not valid JSON", " ".join(broken["explanation"]))

    def test_the_audit_record_is_found_from_a_watermark_id_read_with_bit_errors(self):
        issued_id = bytes(range(16)).hex()
        self.log.record("manifest_issued", subject=issued_id, video_sha256="f" * 64, job="JOB-42")
        corrupted = f"{int(issued_id, 16) ^ 0b10101:032x}"
        self.marks["video"] = {**MARKED, "manifestId": corrupted}
        result = authenticity.verify_file(self.video)
        self.assertEqual(result["record"]["bitErrors"], 3)
        text = " ".join(result["explanation"])
        self.assertIn("JOB-42", text)
        self.assertIn("modified copy", text)  # the issued hash is not this file's

    def test_an_id_that_matches_nothing_finds_no_record(self):
        self.log.record("manifest_issued", subject=bytes(16).hex(), video_sha256="f" * 64)
        self.marks["video"] = {**MARKED, "manifestId": "ff" * 16}
        self.assertIsNone(authenticity.verify_file(self.video)["record"])

    def test_the_report_describes_the_file_itself(self):
        facts = authenticity.verify_file(self.video)["file"]
        self.assertEqual((facts["kind"], facts["name"]), ("video", "clip.mp4"))
        self.assertEqual(facts["sha256"], manifest.sha256_file(self.video))
        self.assertAlmostEqual(facts["durationSeconds"], 1.0, delta=0.2)

    def test_something_that_is_not_media_is_refused(self):
        junk = self.dir / "junk.mp4"
        junk.write_bytes(b"this is not a video" * 50)
        with self.assertRaises(ValueError):
            authenticity.verify_file(junk)


class ModelMissingTests(AuthenticityCase):
    def test_when_the_video_model_is_missing_the_answer_says_so_instead_of_pretending(self):
        self.marks["audio"] = {"detected": False}
        with mock.patch.object(authenticity, "_video_mark", return_value={"available": False, "detected": False, "reason": "missing"}):
            result = authenticity.verify_file(self.video)
        self.assertEqual(result["verdict"], "no_evidence")
        self.assertIn("could not be checked", " ".join(result["explanation"]))


class EndpointTests(AuthenticityCase):
    def setUp(self):
        super().setUp()
        for patcher in (mock.patch.object(app_module.security_gate, "inspect", return_value=(True, 0, {})),
                        mock.patch.object(app_module, "outputs_dir", self.dir),
                        mock.patch.object(app_module, "inputs_dir", self.dir / "in"),
                        mock.patch.object(app_module, "project_root", self.dir)):
            patcher.start()
            self.addCleanup(patcher.stop)
        (self.dir / "in").mkdir()
        self.client = TestClient(app_module.app)

    def leftovers(self):
        return sorted(p.name for p in Path(tempfile.gettempdir()).glob("verify-*"))

    def test_an_uploaded_file_and_manifest_are_verified_under_the_uploads_own_name_and_the_temp_copy_is_removed(self):
        before = self.leftovers()
        reply = self.client.post("/api/v1/provenance/verify", files={
            "file": ("my holiday.mp4", self.video.read_bytes(), "video/mp4"),
            "manifest": ("m.json", json.dumps(self.issue()).encode(), "application/json")})
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(reply.json()["file"]["name"], "my holiday.mp4")      # not the verify-xxxx temp name
        self.assertEqual(self.leftovers(), before)

    def test_a_modified_upload_with_the_original_manifest_is_ours_but_modified(self):
        edited = self.dir / "edited.mp4"
        make_video(edited, tone=880)
        reply = self.client.post("/api/v1/provenance/verify", files={
            "file": ("edited.mp4", edited.read_bytes(), "video/mp4"),
            "manifest": ("m.json", json.dumps(self.issue()).encode(), "application/json")})
        self.assertEqual(reply.json()["verdict"], "ours_modified")

    def test_the_exact_bytes_upload_as_authentic(self):
        document = self.issue()
        reply = self.client.post("/api/v1/provenance/verify", files={
            "file": ("clip.mp4", self.video.read_bytes(), "video/mp4"), "manifest": ("m.json", json.dumps(document).encode(), "application/json")})
        self.assertEqual(reply.json()["verdict"], "authentic_original")

    def test_a_path_under_outputs_is_verified_with_its_sidecar(self):
        Path(str(self.video) + ".manifest.json").write_text(json.dumps(self.issue()))
        reply = self.client.post("/api/v1/provenance/verify", data={"path": "clip.mp4"})
        self.assertEqual(reply.json()["verdict"], "authentic_original")

    def test_send_exactly_one_of_file_or_path(self):
        self.assertEqual(self.client.post("/api/v1/provenance/verify").status_code, 400)
        reply = self.client.post("/api/v1/provenance/verify", data={"path": "clip.mp4"}, files={"file": ("a.mp4", b"x", "video/mp4")})
        self.assertEqual(reply.status_code, 400)

    def test_a_path_outside_outputs_or_inputs_is_refused(self):
        for raw in ("../../etc/passwd", "/etc/passwd"):
            self.assertEqual(self.client.post("/api/v1/provenance/verify", data={"path": raw}).status_code, 400, raw)

    def test_an_oversize_upload_is_413_and_leaves_no_temp_file(self):
        before = self.leftovers()
        with mock.patch.object(app_module, "MAX_VERIFY_BYTES", 1000):
            reply = self.client.post("/api/v1/provenance/verify", files={"file": ("big.mp4", b"0" * 5000, "video/mp4")})
        self.assertEqual(reply.status_code, 413)
        self.assertEqual(self.leftovers(), before)

    def test_a_bad_or_oversize_manifest_upload_is_rejected(self):
        files = {"file": ("clip.mp4", self.video.read_bytes(), "video/mp4")}
        bad = self.client.post("/api/v1/provenance/verify", files={**files, "manifest": ("m.json", b"{nope", "application/json")})
        self.assertEqual(bad.status_code, 400)
        with mock.patch.object(app_module, "MAX_MANIFEST_BYTES", 10):
            big = self.client.post("/api/v1/provenance/verify", files={**files, "manifest": ("m.json", b'{"a":"' + b"x" * 50 + b'"}', "application/json")})
        self.assertEqual(big.status_code, 413)

    def test_a_file_that_is_not_media_is_422_without_a_server_path(self):
        before = self.leftovers()
        reply = self.client.post("/api/v1/provenance/verify", files={"file": ("junk.mp4", b"not media" * 100, "video/mp4")})
        self.assertEqual(reply.status_code, 422)
        self.assertNotIn(str(self.dir), reply.text)
        self.assertNotIn("/tmp/", reply.text)
        self.assertNotIn("verify-", reply.text)  # the temporary name never reaches the caller
        self.assertIn("junk.mp4", reply.text)    # the caller's own name does
        self.assertEqual(self.leftovers(), before)


if __name__ == "__main__":
    unittest.main()
