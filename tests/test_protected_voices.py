"""The protected-voice list (T8.9): what is stored, when it matches, and that cloning is refused on a match."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import protected_voices
import provenance
from audit_log import AuditLog
from job_store import JobStore

# Fake speaker embeddings, chosen by the first sample of the file, so tests need no ECAPA weights.
VOICES = {0.1: np.array([1.0, 0.0, 0.0], np.float32), 0.2: np.array([0.0, 1.0, 0.0], np.float32),
          0.3: np.array([0.9, 0.1, 0.0], np.float32)}  # 0.3 is a close relative of 0.1


def fake_embed(path):
    data, _ = sf.read(str(path), dtype="float32")
    return VOICES[round(float(data[0]), 1)]


def wav(path, marker):
    sf.write(path, np.full(2400, marker, dtype=np.float32), 24000)
    return path


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.log = AuditLog(":memory:")
        patcher = mock.patch("audit_log.shared_audit", return_value=self.log)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.embed = mock.Mock(side_effect=fake_embed)
        self.voices = protected_voices.ProtectedVoices(JobStore(":memory:"), embed=self.embed)


class RegistryTests(Base):
    def test_only_an_embedding_and_an_opaque_id_are_kept(self):
        voice_id = self.voices.register(wav(self.dir / "alice_smith_private.wav", 0.1))
        record = self.voices._store.get(protected_voices.KIND, voice_id)
        self.assertEqual(set(record), {"embedding", "added"})
        self.assertNotIn("alice", str(record).lower())
        self.assertEqual(self.voices.list(), [{"id": voice_id, "added": record["added"]}])
        self.assertEqual(self.log.query(event="protected_voice_added")[0]["subject"], voice_id)

    def test_remove_blanks_the_embedding_and_is_logged_once(self):
        voice_id = self.voices.register(wav(self.dir / "a.wav", 0.1))
        self.assertTrue(self.voices.remove(voice_id))
        self.assertFalse(self.voices.remove(voice_id))
        self.assertFalse(self.voices.remove("nope"))
        self.assertEqual(self.voices._store.get(protected_voices.KIND, voice_id)["embedding"], [])
        self.assertEqual(self.voices.list(), [])
        self.assertEqual(len(self.log.query(event="protected_voice_removed")), 1)

    def test_an_empty_list_never_loads_the_encoder(self):
        self.assertIsNone(self.voices.check(wav(self.dir / "x.wav", 0.1), kind="reference", subject="s"))
        self.embed.assert_not_called()


class MatchTests(Base):
    def setUp(self):
        super().setUp()
        self.protected = self.voices.register(wav(self.dir / "protected.wav", 0.1))

    def test_a_close_voice_raises_an_alert_naming_the_protected_id_and_never_the_audio(self):
        alert = self.voices.check(wav(self.dir / "t.wav", 0.3), kind="output", subject="abc")
        self.assertEqual((alert["kind"], alert["protectedId"]), ("output", self.protected))
        self.assertGreater(alert["similarity"], 0.9)
        entry = self.log.query(event="abuse_alert")[0]
        self.assertEqual((entry["subject"], entry["details"]["protectedId"]), ("abc", self.protected))

    def test_a_different_voice_raises_nothing(self):
        self.assertIsNone(self.voices.check(wav(self.dir / "o.wav", 0.2), kind="reference", subject="s"))
        self.assertEqual(self.log.query(event="abuse_alert"), [])

    def test_the_threshold_is_configurable_and_decides_the_outcome(self):
        with mock.patch.dict(os.environ, {"PROTECTED_VOICE_THRESHOLD": "0.995"}):
            self.assertIsNone(self.voices.check(wav(self.dir / "t.wav", 0.3), kind="output", subject="s"))
        with mock.patch.dict(os.environ, {"PROTECTED_VOICE_THRESHOLD": "-1"}):
            self.assertIsNotNone(self.voices.check(wav(self.dir / "o.wav", 0.2), kind="output", subject="s"))


class CloneGateTests(Base):
    def reference(self, name, marker):
        path = wav(self.dir / name, marker)
        provenance.write(path, source="human", speaker="Someone", licence="CC0", consent_basis="subject-provided")
        return path

    def test_a_consented_reference_that_matches_a_protected_voice_is_refused_and_logged(self):
        from voice_engine import VoiceConsentRequired, require_voice_consent

        protected = self.voices.register(wav(self.dir / "p.wav", 0.1))
        with mock.patch("protected_voices.shared", return_value=self.voices):
            with self.assertRaisesRegex(VoiceConsentRequired, "protected list"):
                require_voice_consent(str(self.reference("close.wav", 0.3)))
        self.assertEqual(self.log.query(event="abuse_alert")[0]["details"]["protectedId"], protected)
        self.assertIn("matches protected voice", self.log.query(event="voice_refused")[0]["details"]["reason"])

    def test_a_consented_reference_that_does_not_match_is_allowed(self):
        from voice_engine import require_voice_consent

        self.voices.register(wav(self.dir / "p.wav", 0.1))
        with mock.patch("protected_voices.shared", return_value=self.voices):
            require_voice_consent(str(self.reference("far.wav", 0.2)))
        self.assertEqual(self.log.query(event="abuse_alert"), [])

    def test_with_an_unavailable_encoder_the_clone_is_refused_not_waved_through(self):
        from voice_engine import require_voice_consent

        self.voices.register(wav(self.dir / "p.wav", 0.1))
        self.embed.side_effect = RuntimeError("ECAPA-TDNN is not available")
        with mock.patch("protected_voices.shared", return_value=self.voices):
            with self.assertRaisesRegex(RuntimeError, "ECAPA"):
                require_voice_consent(str(self.reference("x.wav", 0.2)))


class ApiTests(Base):
    def setUp(self):
        super().setUp()
        import app as app_module

        for patcher in (mock.patch.object(app_module.security_gate, "inspect", return_value=(True, 0, {})),
                        mock.patch("protected_voices.shared", return_value=self.voices)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(app_module.app)

    def upload(self, marker=0.1):
        path = wav(self.dir / "up.wav", marker)
        return self.client.post("/api/v1/abuse/protected-voices", files={"file": ("my_name.wav", path.read_bytes(), "audio/wav")})

    def test_add_list_and_remove_over_http(self):
        created = self.upload()
        self.assertEqual(created.status_code, 201, created.text)
        voice_id = created.json()["id"]
        listing = self.client.get("/api/v1/abuse/protected-voices").json()
        self.assertEqual([v["id"] for v in listing["voices"]], [voice_id])
        self.assertEqual(self.client.delete(f"/api/v1/abuse/protected-voices/{voice_id}").status_code, 204)
        self.assertEqual(self.client.delete(f"/api/v1/abuse/protected-voices/{voice_id}").status_code, 404)

    def test_an_unreadable_recording_is_a_422_and_leaves_no_temp_file(self):
        self.embed.side_effect = ValueError("the recording is empty")
        before = set(Path(tempfile.gettempdir()).glob("verify-*"))
        response = self.upload()
        self.assertEqual(response.status_code, 422)
        self.assertIn("empty", response.json()["detail"])
        self.assertEqual(set(Path(tempfile.gettempdir()).glob("verify-*")), before)

    def test_an_oversized_upload_is_a_413(self):
        import app as app_module

        with mock.patch.object(app_module, "MAX_VOICE_UPLOAD_BYTES", 100):
            self.assertEqual(self.upload().status_code, 413)


if __name__ == "__main__":
    unittest.main()
