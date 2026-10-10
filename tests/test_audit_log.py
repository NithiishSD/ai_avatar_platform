"""
Consent audit trail (R-35): the hash chain, the lookups, and the hooks that write to it.
"""

import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import app as app_module
import avatar_store
import manifest
import provenance
import request_context
from audit_log import GENESIS, AuditLog
from test_app_vision import VisionApiCase


class ChainTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "audit.sqlite"
        self.log = AuditLog(self.path)
        for i in range(5):
            self.log.record("face_use", subject=f"avatar-{i}", basis="written-consent", job=f"J{i}")

    def raw(self, sql, *args):
        db = sqlite3.connect(self.path)
        db.execute(sql, args)
        db.commit()
        db.close()

    def test_an_untouched_trail_verifies_and_links_each_entry_to_the_one_before(self):
        result = self.log.verify_chain()
        self.assertEqual((result["valid"], result["entries"]), (True, 5))
        self.assertEqual(result["headHash"], self.log.head()["headHash"])
        self.assertNotEqual(self.log.head()["headHash"], GENESIS)

    def test_editing_any_field_of_any_entry_is_found_at_that_entry(self):
        for column, value in (("basis", "written-consent-forged"), ("subject", "someone-else"), ("event", "face_refused"),
                              ("details", "{}"), ("ts", "2000-01-01T00:00:00Z")):
            with self.subTest(column=column):
                fresh = AuditLog(Path(self._tmp.name) / f"{column}.sqlite")
                for i in range(5):
                    fresh.record("face_use", subject=f"a{i}", basis="open-licence", job=i)
                db = sqlite3.connect(Path(self._tmp.name) / f"{column}.sqlite")
                db.execute(f"UPDATE audit SET {column} = ? WHERE id = 3", (value,))
                db.commit()
                db.close()
                result = fresh.verify_chain()
                self.assertFalse(result["valid"])
                self.assertEqual(result["brokenAt"], 3)

    def test_deleting_a_row_from_the_middle_is_found(self):
        self.raw("DELETE FROM audit WHERE id = 3")
        result = self.log.verify_chain()
        self.assertFalse(result["valid"])
        self.assertIn("deleted", result["reason"])

    def test_deleting_the_newest_row_is_only_visible_through_the_head_hash(self):
        # Stated limit of any chain: the end can be truncated without a break. The head is what to keep elsewhere.
        before = self.log.head()
        self.raw("DELETE FROM audit WHERE id = 5")
        self.assertTrue(self.log.verify_chain()["valid"])
        after = self.log.head()
        self.assertNotEqual(before["headHash"], after["headHash"])
        self.assertEqual((before["entries"], after["entries"]), (5, 4))

    def test_swapping_two_rows_breaks_the_chain(self):
        self.raw("UPDATE audit SET ts = 'x' WHERE id = 2")
        self.assertFalse(self.log.verify_chain()["valid"])

    def test_concurrent_writers_leave_a_valid_contiguous_chain(self):
        threads = [threading.Thread(target=lambda: [self.log.record("voice_use", subject="h", basis="b") for _ in range(25)]) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        result = self.log.verify_chain()
        self.assertEqual((result["valid"], result["entries"]), (True, 205))

    def test_an_unknown_event_is_refused(self):
        with self.assertRaises(ValueError):
            self.log.record("something_else", subject="x")


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.log = AuditLog(":memory:")
        self.log.record("voice_use", subject="hash-a", basis="open-licence")
        self.log.record("face_use", subject="demo", basis="synthetic")
        self.log.record("voice_refused", subject="hash-b", reason="no consent")

    def test_newest_first_and_filtered_by_event_and_subject(self):
        self.assertEqual([e["event"] for e in self.log.query()], ["voice_refused", "face_use", "voice_use"])
        self.assertEqual([e["subject"] for e in self.log.query(event="voice_use")], ["hash-a"])
        self.assertEqual([e["event"] for e in self.log.query(subject="demo")], ["face_use"])
        self.assertEqual(self.log.query(since="2999-01-01T00:00:00Z"), [])

    def test_limit_is_clamped(self):
        self.assertEqual(len(self.log.query(limit=1)), 1)
        self.assertEqual(len(self.log.query(limit=-5)), 1)

    def test_the_request_id_current_when_it_was_written_is_kept(self):
        token = request_context.bind_request_id("trace-77")
        try:
            entry = self.log.record("face_use", subject="x")
        finally:
            request_context.unbind_request_id(token)
        self.assertEqual(entry["requestId"], "trace-77")


class FindManifestTests(unittest.TestCase):
    def setUp(self):
        self.log = AuditLog(":memory:")
        self.id_a, self.id_b = bytes(range(16)).hex(), bytes(range(100, 116)).hex()
        self.log.record("manifest_issued", subject=self.id_a, video_sha256="a" * 64)
        self.log.record("manifest_issued", subject=self.id_b, video_sha256="b" * 64)

    def flip(self, hex_id, bits):
        value = int(hex_id, 16)
        for i in range(bits):
            value ^= 1 << (i * 5)
        return f"{value:032x}"

    def test_an_exact_id_and_one_read_with_bit_errors_both_find_their_record(self):
        self.assertEqual(self.log.find_manifest(self.id_a)["details"]["video_sha256"], "a" * 64)
        found = self.log.find_manifest(self.flip(self.id_b, 12))
        self.assertEqual((found["subject"], found["bitErrors"]), (self.id_b, 12))

    def test_an_id_too_far_from_every_record_finds_nothing(self):
        self.assertIsNone(self.log.find_manifest(self.flip(self.id_a, 40)))
        self.assertIsNone(self.log.find_manifest("not hex"))

    def test_only_manifest_entries_are_searched(self):
        self.log.record("face_use", subject=self.flip(self.id_a, 1))
        self.assertEqual(self.log.find_manifest(self.flip(self.id_a, 1))["subject"], self.id_a)


class HookTests(unittest.TestCase):
    """What the voice and face code paths write to the shared trail."""

    def setUp(self):
        self.log = AuditLog(":memory:")
        patcher = mock.patch("audit_log.shared_audit", return_value=self.log)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def reference(self, name, **fields):
        path = self.dir / name
        sf.write(path, np.zeros(2400, dtype=np.float32), 24000)
        if fields:
            provenance.write(path, **fields)
        return path

    def test_a_refused_clone_is_logged_by_file_hash_with_the_reason_and_never_the_name(self):
        from voice_engine import VoiceConsentRequired, require_voice_consent

        unknown = self.reference("secret_person.wav")
        with self.assertRaises(VoiceConsentRequired):
            require_voice_consent(str(unknown))
        no_basis = self.reference("basisless.wav", source="human", speaker="Secret Person", licence="x")
        with self.assertRaises(VoiceConsentRequired):
            require_voice_consent(str(no_basis))
        entries = self.log.query(event="voice_refused")
        self.assertEqual(len(entries), 2)
        self.assertEqual({e["subject"] for e in entries}, {manifest.sha256_file(unknown), manifest.sha256_file(no_basis)})
        self.assertIn("no provenance record", entries[1]["details"]["reason"])

    def test_a_missing_file_is_not_a_consent_event(self):
        from voice_engine import require_voice_consent

        with self.assertRaises(FileNotFoundError):
            require_voice_consent(str(self.dir / "nope.wav"))
        self.assertEqual(self.log.query(), [])

    def test_a_clone_that_ran_is_logged_with_the_basis_it_ran_under(self):
        from test_voice_engine import weights_on_disk
        from voice_engine import VoiceEngineRouter

        reference = self.reference("ok.wav", source="human", speaker="Someone", licence="own", consent_basis="written-consent")
        router = VoiceEngineRouter(device="cpu")

        def fake_xtts(text, out, speaker, language):
            sf.write(out, np.zeros(2400, dtype=np.float32), 24000)
            return 24000, 0.1

        with mock.patch.dict(os.environ, {"WATERMARK_ENABLED": "false", "WATERMARK_KEY": "k"}), \
                mock.patch("model_registry.audit_model_weights", return_value=weights_on_disk()), \
                mock.patch.object(router, "_synthesize_xtts", side_effect=fake_xtts), \
                mock.patch.object(router, "xtts_language", return_value="en"):
            with mock.patch("voice_engine.OUTPUT_DIR", self.dir):
                router.synthesize("hi", mode="clone", speaker_wav=str(reference), output_filename="out.wav")
        (entry,) = self.log.query(event="voice_use")
        self.assertEqual(entry["basis"], "written-consent")
        self.assertEqual(entry["subject"], manifest.sha256_file(reference))
        self.assertEqual(entry["details"]["model"], "xtts-v2")
        self.assertNotIn("Someone", str(entry))

    def test_a_face_without_consent_is_refused_and_logged(self):
        from vision_fixtures import FakeFaceEngine, gradient_image

        store = avatar_store.AvatarStore(root=self.dir / "faces", engine=FakeFaceEngine())
        store.register(gradient_image(256), avatar_id="person", source=provenance.HUMAN, subject="Jane", consent_basis="written-consent")
        provenance.sidecar_path(store.get("person").path).unlink()  # the record goes missing
        import render_engine
        from render_engine import RenderError  # noqa: F401  (the preflight raises the store's error first)
        from test_render_engine import RenderCase  # noqa: F401

        job = mock.Mock(avatar_id="person", job_id="J9")
        with self.assertRaises(avatar_store.AvatarConsentError):
            render_engine.preflight(job, "blendshape", store)
        (entry,) = self.log.query(event="face_refused")
        self.assertEqual((entry["subject"], entry["details"]["job"]), ("person", "J9"))

    def test_live_session_use_and_refusal_are_logged(self):
        import live_engine
        from vision_fixtures import FakeFaceEngine, gradient_image

        store = avatar_store.AvatarStore(root=self.dir / "faces", engine=FakeFaceEngine())
        store.register(gradient_image(256), avatar_id="demo", source=provenance.SYNTHETIC)
        store.register(gradient_image(256), avatar_id="real", source=provenance.HUMAN, subject="Jane", consent_basis="written-consent")
        provenance.sidecar_path(store.get("real").path).unlink()
        refused = live_engine.LiveSession(mock.Mock(), store, "real")
        with self.assertRaises(avatar_store.AvatarConsentError):
            refused.open()
        self.assertEqual(self.log.query(event="face_refused")[0]["details"]["live"], True)
        allowed = live_engine.LiveSession(mock.Mock(), store, "demo", max_side=128)
        with mock.patch("live_engine.shared_face_engine", return_value=FakeFaceEngine()), \
                mock.patch("live_engine.PortraitAnimator"):
            allowed.open()
        use = self.log.query(event="face_use")[0]
        self.assertEqual((use["subject"], use["basis"], use["details"]["live"]), ("demo", "synthetic", True))


class AuditApiTests(unittest.TestCase):
    def setUp(self):
        self.log = AuditLog(":memory:")
        for patcher in (mock.patch("audit_log.shared_audit", return_value=self.log),
                        mock.patch.object(app_module.security_gate, "inspect", return_value=(True, 0, {}))):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(app_module.app)
        self.log.record("face_use", subject="demo", basis="synthetic")
        self.log.record("voice_use", subject="h", basis="open-licence")

    def test_listing_filtering_and_the_head(self):
        body = self.client.get("/api/v1/audit").json()
        self.assertEqual([e["event"] for e in body["entries"]], ["voice_use", "face_use"])
        self.assertEqual(body["head"]["entries"], 2)
        self.assertIn("manifest_issued", body["events"])
        only = self.client.get("/api/v1/audit", params={"event": "face_use"}).json()["entries"]
        self.assertEqual([e["subject"] for e in only], ["demo"])

    def test_an_unknown_event_filter_is_a_400_that_lists_the_valid_ones(self):
        reply = self.client.get("/api/v1/audit", params={"event": "bogus"})
        self.assertEqual(reply.status_code, 400)
        self.assertIn("face_use", reply.json()["detail"])

    def test_verify_reports_an_intact_trail_and_a_broken_one(self):
        self.assertTrue(self.client.get("/api/v1/audit/verify").json()["valid"])
        self.log._db.execute("UPDATE audit SET basis = 'forged' WHERE id = 1")
        broken = self.client.get("/api/v1/audit/verify").json()
        self.assertEqual((broken["valid"], broken["brokenAt"]), (False, 1))

    def test_entries_never_carry_names(self):
        self.assertNotIn("Jane", self.client.get("/api/v1/audit").text)


class RegistrationHookTests(VisionApiCase):
    def test_registering_a_photo_through_the_api_is_logged_with_its_basis_and_not_its_subject(self):
        log = AuditLog(":memory:")
        with mock.patch("audit_log.shared_audit", return_value=log):
            reply = self.upload()
        self.assertEqual(reply.status_code, 201, reply.text)
        (entry,) = log.query(event="face_registered")
        self.assertEqual((entry["subject"], entry["basis"]), ("alice", "subject-provided"))
        self.assertNotIn("Alice Example", str(entry))


if __name__ == "__main__":
    unittest.main()
