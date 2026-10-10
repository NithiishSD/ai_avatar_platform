import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contracts import AvatarRenderJob, RenderQuality


VALID_JOB = {
    "jobId": "AVT-9821-X",
    "avatarId": "AVATAR_FEMALE_04",
    "audioUrl": "s3://assets/audio/generated_tts_9821.wav",
    "sampleRate": 24000,
    "durationSeconds": 14.5,
    "phonemeTimestamps": [
        {"phoneme": "HH", "viseme": "viseme_sil", "startMs": 0, "endMs": 120},
        {"phoneme": "EH", "viseme": "viseme_E", "startMs": 120, "endMs": 280},
        {"phoneme": "L", "viseme": "viseme_L", "startMs": 280, "endMs": 410},
        {"phoneme": "OW", "viseme": "viseme_O", "startMs": 410, "endMs": 620},
    ],
    "emotionVector": {"happy": 0.8, "neutral": 0.2, "eyeblinkRate": 1.2},
    "renderQuality": "1080P_HQ",
    "targetFps": 30,
}


class AvatarRenderContractTests(unittest.TestCase):
    def test_roadmap_payload_validates(self):
        job = AvatarRenderJob.model_validate(VALID_JOB)

        self.assertEqual(job.job_id, "AVT-9821-X")
        self.assertEqual(job.render_quality, RenderQuality.HD_1080P)
        self.assertEqual(job.phoneme_timestamps[1].start_ms, 120)

    def test_rejects_timestamp_outside_audio_duration(self):
        invalid_job = {**VALID_JOB, "phonemeTimestamps": [{"phoneme": "AA", "viseme": "viseme_A", "startMs": 0, "endMs": 15000}]}

        with self.assertRaises(ValidationError):
            AvatarRenderJob.model_validate(invalid_job)

    def test_rejects_unordered_or_invalid_interval(self):
        invalid_job = {**VALID_JOB, "phonemeTimestamps": [
            {"phoneme": "AA", "viseme": "viseme_A", "startMs": 300, "endMs": 200},
            {"phoneme": "BB", "viseme": "viseme_B", "startMs": 100, "endMs": 150},
        ]}

        with self.assertRaises(ValidationError):
            AvatarRenderJob.model_validate(invalid_job)

    def test_rejects_invalid_media_bounds_and_url(self):
        invalid_job = {**VALID_JOB, "sampleRate": 1000, "targetFps": 0, "audioUrl": "audio.wav"}

        with self.assertRaises(ValidationError):
            AvatarRenderJob.model_validate(invalid_job)


if __name__ == "__main__":
    unittest.main()

class MotionIntensityContractTests(unittest.TestCase):
    def test_motion_intensity_is_optional_and_bounded(self):
        from pydantic import ValidationError

        from contracts import AvatarRenderJob

        base = {"jobId": "M1", "avatarId": "demo", "audioUrl": "file:///x.wav", "sampleRate": 24000, "durationSeconds": 1.0,
                "phonemeTimestamps": [{"phoneme": "AA", "viseme": "viseme_aa", "startMs": 0, "endMs": 400}],
                "emotionVector": {"happy": 0, "neutral": 1, "eyeblinkRate": 1}, "renderQuality": "PREVIEW", "targetFps": 25}
        self.assertIsNone(AvatarRenderJob.model_validate(base).motion_intensity)  # older jobs unchanged
        self.assertEqual(AvatarRenderJob.model_validate({**base, "motionIntensity": 1.5}).motion_intensity, 1.5)
        with self.assertRaises(ValidationError):
            AvatarRenderJob.model_validate({**base, "motionIntensity": 3})
