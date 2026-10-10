"""API tests for the Phase 3 endpoints and the security middleware."""

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import app as app_module
from security import API_KEY_HEADER, SecurityConfig, SecurityGate


class LanguageEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)

    def test_catalogue_reports_over_one_thousand_languages(self):
        payload = self.client.get("/api/v1/audio/languages?limit=5").json()
        self.assertGreater(payload["total"], 1000)
        self.assertEqual(payload["returned"], 5)
        self.assertTrue(payload["source"].startswith("http"))

    def test_search_narrows_the_catalogue(self):
        payload = self.client.get("/api/v1/audio/languages?q=swahili").json()
        self.assertTrue(payload["returned"] >= 1)
        self.assertTrue(
            any("swahili" in entry["name"].lower() for entry in payload["languages"])
        )

    def test_entries_carry_the_model_id_for_supported_languages(self):
        payload = self.client.get("/api/v1/audio/languages/hi").json()
        self.assertTrue(payload["mmsSupported"])
        self.assertEqual(payload["mmsModel"], "facebook/mms-tts-hin")
        self.assertEqual(payload["iso3"], "hin")

    def test_unsupported_language_is_reported_without_a_model(self):
        payload = self.client.get("/api/v1/audio/languages/ja").json()
        self.assertFalse(payload["mmsSupported"])
        self.assertIsNone(payload["mmsModel"])

    def test_unknown_code_returns_a_body_rather_than_an_error(self):
        response = self.client.get("/api/v1/audio/languages/zzz")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["mmsSupported"])


class EmotionEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)

    def test_every_roadmap_emotion_is_exposed(self):
        payload = self.client.get("/api/v1/audio/emotions").json()
        names = {preset["name"] for preset in payload["presets"]}
        self.assertTrue({"joy", "anger", "sorrow", "authority"}.issubset(names))
        self.assertEqual(payload["default"], "neutral")

    def test_presets_describe_their_prosody_and_render_hint(self):
        payload = self.client.get("/api/v1/audio/emotions").json()
        joy = next(p for p in payload["presets"] if p["name"] == "joy")
        self.assertIn("pitch_semitones", joy["prosody"])
        self.assertIn("happy", joy["renderHint"])


class QualityEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)
        self.outputs = Path(app_module.outputs_dir)
        self.clip = self.outputs / "_phase3_api_test.wav"
        tone = 0.2 * np.sin(
            2 * np.pi * 160 * np.linspace(0, 1.0, 24000, endpoint=False)
        )
        sf.write(self.clip, tone.astype(np.float32), 24000)

    def tearDown(self):
        self.clip.unlink(missing_ok=True)

    def test_audit_returns_a_report_with_its_method(self):
        fake = mock.Mock()
        fake.to_dict.return_value = {"mos": 4.2, "method": "torchaudio-squim"}
        with mock.patch.object(app_module.quality_auditor, "audit", return_value=fake):
            response = self.client.post(
                "/api/v1/audio/quality-audit",
                json={"audioPath": self.clip.name},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["report"]["method"], "torchaudio-squim")

    def test_audit_rejects_a_path_outside_the_project(self):
        response = self.client.post(
            "/api/v1/audio/quality-audit",
            json={"audioPath": "../../../etc/passwd"},
        )
        self.assertIn(response.status_code, (400, 404))

    def test_audit_reports_a_missing_file_as_404(self):
        response = self.client.post(
            "/api/v1/audio/quality-audit",
            json={"audioPath": "definitely_not_here.wav"},
        )
        self.assertEqual(response.status_code, 404)

    def test_similarity_endpoint_passes_both_paths_through(self):
        fake = mock.Mock()
        fake.to_dict.return_value = {"similarity": 0.9, "isEcapa": True}
        with mock.patch.object(
            app_module.quality_auditor, "speaker_similarity", return_value=fake
        ) as scorer:
            response = self.client.post(
                "/api/v1/audio/voice-similarity",
                json={
                    "referencePath": self.clip.name,
                    "generatedPath": self.clip.name,
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["report"]["isEcapa"])
        scorer.assert_called_once()


class HealthCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)

    def test_health_advertises_only_models_that_can_run(self):
        capabilities = self.client.get("/health").json()["capabilities"]
        self.assertEqual(
            set(capabilities["models"]),
            {"kokoro", "xtts-v2", "openvoice-v2", "bark", "mms-tts"},
        )
        self.assertEqual(capabilities["cloneEngines"], ["xtts-v2", "openvoice-v2"])

    def test_health_advertises_the_fifteen_visemes(self):
        capabilities = self.client.get("/health").json()["capabilities"]
        self.assertEqual(len(capabilities["visemes"]), 15)

    def test_health_reports_the_security_posture(self):
        security = self.client.get("/health").json()["security"]
        self.assertIn("authEnabled", security)
        self.assertEqual(security["apiKeyHeader"], API_KEY_HEADER)


class SecurityMiddlewareTests(unittest.TestCase):
    """Auth is toggled on the live app object, then restored."""

    def setUp(self):
        self.original = app_module.security_gate
        app_module.security_gate = SecurityGate(
            SecurityConfig(
                auth_enabled=True,
                api_keys=frozenset({"test-key"}),
                rate_limit_enabled=False,
                requests_per_minute=120,
                burst=10,
            )
        )
        self.client = TestClient(app_module.app)

    def tearDown(self):
        app_module.security_gate = self.original

    def test_protected_endpoint_requires_a_key(self):
        response = self.client.get("/api/v1/audio/languages")
        self.assertEqual(response.status_code, 401)
        self.assertIn("API key", response.json()["detail"])

    def test_valid_key_is_accepted(self):
        response = self.client.get(
            "/api/v1/audio/languages?limit=1", headers={API_KEY_HEADER: "test-key"}
        )
        self.assertEqual(response.status_code, 200)

    def test_health_stays_reachable_without_a_key(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_rate_limit_returns_429_with_retry_after(self):
        app_module.security_gate = SecurityGate(
            SecurityConfig(
                auth_enabled=False,
                api_keys=frozenset(),
                rate_limit_enabled=True,
                requests_per_minute=60,
                burst=1,
            )
        )
        client = TestClient(app_module.app)
        self.assertEqual(client.get("/api/v1/audio/languages?limit=1").status_code, 200)
        throttled = client.get("/api/v1/audio/languages?limit=1")
        self.assertEqual(throttled.status_code, 429)
        self.assertIn("retry-after", {k.lower() for k in throttled.headers})


class SynthesisResponseProjectionTests(unittest.TestCase):
    def test_completed_result_is_projected_onto_the_response(self):
        response = app_module._job_response(
            "task-1",
            "SUCCESS",
            {
                "model": "mms-tts",
                "output_path": "outputs/speech.wav",
                "duration_seconds": 3.2,
                "latency_ms": 412.0,
                "emotion": {"dominant": "joy"},
                "quality_report": {"mos": 4.1},
                "language": {"iso3": "hin"},
                "phoneme_timestamps": [
                    {"phoneme": "AA", "viseme": "viseme_aa", "startMs": 0, "endMs": 100}
                ],
            },
        )
        self.assertEqual(response.model_used, "mms-tts")
        self.assertEqual(response.emotion["dominant"], "joy")
        self.assertEqual(response.quality_report["mos"], 4.1)
        self.assertEqual(response.language["iso3"], "hin")
        self.assertEqual(response.latency_ms, 412.0)
        self.assertEqual(len(response.phoneme_timestamps), 1)

    def test_partial_result_does_not_raise(self):
        response = app_module._job_response("task-2", "SUCCESS", {"model": "kokoro"})
        self.assertIsNone(response.output_path)
        self.assertIsNone(response.quality_report)


if __name__ == "__main__":
    unittest.main()
