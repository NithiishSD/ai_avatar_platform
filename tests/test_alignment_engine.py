"""
Unit tests for ForcedAligner, PhonemeToVisemeMapper, and Alignment API routes (Phase 2).
"""

import os
import sys
import unittest
from pathlib import Path

import torch
from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from alignment_engine import ForcedAligner, PhonemeToVisemeMapper, PHONEME_TO_VISEME
from app import app
from contracts import (
    AlignmentRequest,
    AlignmentResponse,
    AudioSynthesisRequest,
    PhonemeTimestamp,
    RenderQuality,
    AvatarRenderJob,
    EmotionVector,
)


class PhonemeToVisemeMapperTests(unittest.TestCase):
    """Tests for phoneme-to-viseme conversion accuracy and robustness."""

    def test_canonical_visemes_list(self):
        visemes = PhonemeToVisemeMapper.get_supported_visemes()
        self.assertEqual(len(visemes), 15)
        self.assertIn("viseme_sil", visemes)
        self.assertIn("viseme_aa", visemes)
        self.assertIn("viseme_PP", visemes)
        self.assertIn("viseme_E", visemes)
        self.assertIn("viseme_O", visemes)

    def test_arpabet_mapping(self):
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("AA"), "viseme_aa")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("P"), "viseme_PP")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("B"), "viseme_PP")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("M"), "viseme_PP")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("F"), "viseme_FF")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("V"), "viseme_FF")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("TH"), "viseme_TH")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("DH"), "viseme_TH")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("S"), "viseme_SS")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("Z"), "viseme_SS")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("CH"), "viseme_CH")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("SH"), "viseme_CH")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("K"), "viseme_kk")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("G"), "viseme_kk")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("R"), "viseme_RR")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("ER"), "viseme_RR")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("UW"), "viseme_U")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("OW"), "viseme_O")

    def test_stress_digit_stripping(self):
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("AA1"), "viseme_aa")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("EH0"), "viseme_E")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("IY2"), "viseme_I")

    def test_silence_and_empty_mapping(self):
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme(""), "viseme_sil")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("SIL"), "viseme_sil")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme("SP"), "viseme_sil")
        self.assertEqual(PhonemeToVisemeMapper.map_phoneme(" "), "viseme_sil")


class ForcedAlignerTests(unittest.TestCase):
    """Tests for forced alignment generation and interval consistency."""

    def setUp(self):
        self.aligner = ForcedAligner(device="cpu")

    def test_acoustic_alignment_generation(self):
        dummy_waveform = torch.zeros(1, 24000 * 2)  # 2.0 seconds audio
        transcript = "Hello world, welcome to AI Avatar!"
        timestamps = self.aligner.align(
            dummy_waveform,
            transcript=transcript,
            sample_rate=24000,
        )

        self.assertTrue(len(timestamps) > 0)
        previous_end = 0
        for ts in timestamps:
            self.assertIsInstance(ts, PhonemeTimestamp)
            self.assertGreaterEqual(ts.start_ms, 0)
            self.assertGreater(ts.end_ms, ts.start_ms)
            self.assertGreaterEqual(ts.start_ms, previous_end)
            self.assertTrue(ts.viseme.startswith("viseme_"))
            previous_end = ts.start_ms

        # Final end timestamp should match approximately 2000ms
        self.assertLessEqual(timestamps[-1].end_ms, 2050)

    def test_empty_transcript_handling(self):
        dummy_waveform = torch.zeros(1, 24000)
        timestamps = self.aligner.align(dummy_waveform, transcript="", sample_rate=24000)
        self.assertEqual(len(timestamps), 1)
        self.assertEqual(timestamps[0].viseme, "viseme_sil")
        self.assertEqual(timestamps[0].start_ms, 0)
        self.assertEqual(timestamps[0].end_ms, 1000)

    def test_compatibility_with_avatar_render_job(self):
        """Verify generated timestamps satisfy AvatarRenderJob validator constraints."""
        dummy_waveform = torch.zeros(1, 24000 * 3)  # 3.0 seconds
        transcript = "Testing Avatar Render Job Integration."
        timestamps = self.aligner.align(dummy_waveform, transcript=transcript, sample_rate=24000)

        # Build complete AvatarRenderJob payload
        job = AvatarRenderJob(
            jobId="ALIGN-TEST-001",
            avatarId="avatar_model_01",
            audioUrl="http://localhost:8000/outputs/test.wav",
            sampleRate=24000,
            durationSeconds=3.0,
            phonemeTimestamps=timestamps,
            emotionVector=EmotionVector(happy=0.5, neutral=0.5, eyeblinkRate=1.0),
            renderQuality=RenderQuality.PREVIEW,
            targetFps=30,
        )
        self.assertEqual(job.job_id, "ALIGN-TEST-001")
        self.assertEqual(len(job.phoneme_timestamps), len(timestamps))


class AlignmentAPIRouteTests(unittest.TestCase):
    """Tests for FastAPI alignment endpoint."""

    def setUp(self):
        self.client = TestClient(app)

    def test_align_endpoint_missing_file(self):
        response = self.client.post(
            "/api/v1/audio/align",
            json={
                "audioPath": "non_existent_file.wav",
                "transcript": "Hello world",
                "sampleRate": 24000,
                "language": "en",
            },
        )
        self.assertEqual(response.status_code, 404)

    def test_synthesis_request_with_prosody_and_alignment(self):
        req = AudioSynthesisRequest(
            text="Hello with pitch and speed!",
            speed=1.2,
            pitch=1.1,
            returnAlignment=True,
        )
        self.assertEqual(req.speed, 1.2)
        self.assertEqual(req.pitch, 1.1)
        self.assertTrue(req.return_alignment)


if __name__ == "__main__":
    unittest.main()
