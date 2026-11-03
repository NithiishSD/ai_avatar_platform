"""
Unit and integration tests for user-consented voice cloning and demo integration.

Verifies:
1. User voice reference registration with 'speaker-recorded' consent provenance.
2. Audio format readiness (24 kHz mono PCM WAV for zero-shot XTTS-v2).
3. Preflight consent validation on the VoiceEngineRouter.
4. Speaker embedding extraction and similarity metrics via ECAPA-TDNN.
5. Presence and validity of demo assets and manifest configuration.
"""

import json
import unittest
from pathlib import Path

import audio_utils
import provenance
from quality_auditor import SpeechQualityAuditor
from voice_engine import VoiceEngineRouter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class UserVoiceProvenanceTests(unittest.TestCase):
    """Checks provenance and admissibility for the user-provided voice recordings."""

    def test_user_voice_reference_wav_has_admissible_provenance(self):
        wav_path = INPUTS_DIR / "user_voice_reference.wav"
        self.assertTrue(wav_path.exists(), f"Missing {wav_path}")
        desc = provenance.describe(wav_path)
        self.assertTrue(desc["hasRecord"], "Missing provenance sidecar for user_voice_reference.wav")
        self.assertTrue(desc["admissible"], f"User voice reference must be admissible: {desc['reason']}")
        self.assertEqual(desc["source"], provenance.HUMAN)
        self.assertEqual(desc["consentBasis"], "speaker-recorded")

    def test_user_voice_mp3_has_provenance(self):
        mp3_path = INPUTS_DIR / "Record (online-voice-recorder.com).mp3"
        self.assertTrue(mp3_path.exists(), f"Missing {mp3_path}")
        desc = provenance.describe(mp3_path)
        self.assertTrue(desc["hasRecord"], "Missing provenance sidecar for mp3 recording")
        self.assertTrue(desc["admissible"])
        self.assertEqual(desc["consentBasis"], "speaker-recorded")


class UserVoiceAudioReadinessTests(unittest.TestCase):
    """Checks that the user audio is properly formatted and ready for zero-shot cloning."""

    def test_user_reference_ready_for_cloning(self):
        wav_path = INPUTS_DIR / "user_voice_reference.wav"
        info = audio_utils.probe_audio(wav_path)
        self.assertTrue(info.ready_for_cloning, "user_voice_reference.wav is not ready for cloning")
        self.assertEqual(info.sample_rate, 24000)
        self.assertEqual(info.channels, 1)
        self.assertGreaterEqual(info.duration_seconds, 30.0)

    def test_user_voice_listed_in_samples(self):
        samples = audio_utils.list_voice_samples(INPUTS_DIR)
        filenames = [s.filename for s in samples]
        self.assertIn("user_voice_reference.wav", filenames)
        self.assertIn("Record (online-voice-recorder.com).mp3", filenames)


class UserVoiceEnginePreflightTests(unittest.TestCase):
    """Checks that the router preflight accepts the user's voice reference."""

    def setUp(self):
        self.router = VoiceEngineRouter(device="cpu")
        self.ref_path = INPUTS_DIR / "user_voice_reference.wav"

    def test_preflight_passes_for_xtts(self):
        # Should not raise VoiceConsentRequired
        self.router.preflight("xtts-v2", str(self.ref_path))

    def test_preflight_passes_for_openvoice(self):
        # Should not raise VoiceConsentRequired
        self.router.preflight("openvoice-v2", str(self.ref_path))


class UserVoiceAuditorTests(unittest.TestCase):
    """Tests speaker similarity audit reporting on the user voice."""

    def test_auditor_handles_user_voice_similarity_reporting(self):
        auditor = SpeechQualityAuditor(device="cpu")
        report = auditor.speaker_similarity(
            INPUTS_DIR / "user_voice_reference.wav",
            OUTPUTS_DIR / "demo" / "audio" / "clone_xtts_v2_english.wav",
        )
        self.assertIsNotNone(report.similarity)
        self.assertTrue(report.is_ecapa)
        self.assertGreater(report.similarity, 0.2)
        self.assertAlmostEqual(report.to_dict()["similarityPercent"], 24.5, places=0)



class UserVoiceDemoIntegrationTests(unittest.TestCase):
    """Checks that the demo manifest and cloned audio artifacts exist and are valid."""

    def test_demo_cloning_audio_files_exist(self):
        expected_files = [
            "clone_reference_voice.wav",
            "clone_xtts_v2_english.wav",
            "clone_xtts_v2_spanish.wav",
            "clone_xtts_v2_french.wav",
            "clone_xtts_v2_hindi.wav",
        ]
        demo_audio_dir = OUTPUTS_DIR / "demo" / "audio"
        for name in expected_files:
            target = demo_audio_dir / name
            self.assertTrue(target.exists(), f"Demo audio file missing: {target}")
            self.assertGreater(target.stat().st_size, 1000, f"Demo file too small or empty: {target}")

    def test_demo_manifest_references_user_clones(self):
        manifest_path = OUTPUTS_DIR / "demo" / "manifest.json"
        self.assertTrue(manifest_path.exists(), "outputs/demo/manifest.json missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cloning = manifest.get("audio", {}).get("cloning", [])
        self.assertGreaterEqual(len(cloning), 5, "Expected at least 5 cloning items in manifest")
        ids = [item["id"] for item in cloning]
        self.assertIn("ref_speaker", ids)
        self.assertIn("clone_en", ids)
        self.assertIn("clone_es", ids)
        self.assertIn("clone_fr", ids)
        self.assertIn("clone_hi", ids)


if __name__ == "__main__":
    unittest.main()
