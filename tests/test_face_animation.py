"""Frame timeline: phoneme timestamps -> per-frame blendshape weights (G2-03)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from contracts import PhonemeTimestamp
from face_animation import (
    BLINK_CLOSE_S,
    BLINK_OPEN_S,
    MAX_HOLD_MS,
    MIN_BLINK_INTERVAL_S,
    blink_schedule,
    build_animation,
    frame_count_for,
    hold_until_next,
    normalise_timestamps,
    seed_from_job_id,
    speech_energy_envelope,
)


def ts(viseme, start, end):
    return {"phoneme": "X", "viseme": viseme, "startMs": start, "endMs": end}


class FrameCountTests(unittest.TestCase):
    def test_frame_count_is_duration_times_fps(self):
        self.assertEqual(frame_count_for(2.0, 25), 50)
        self.assertEqual(frame_count_for(14.5, 30), 435)
        self.assertEqual(frame_count_for(0.001, 25), 1)
        # Rounded up: the video must cover the audio.
        self.assertEqual(frame_count_for(3.017, 30), 91)
        self.assertEqual(frame_count_for(1.49, 1), 2)

    def test_track_has_exactly_that_many_frames(self):
        for duration, fps in ((1.0, 25), (2.48, 30), (0.5, 60)):
            track = build_animation([ts("viseme_aa", 0, 300)], duration, fps)
            self.assertEqual(track.frame_count, frame_count_for(duration, fps))
            self.assertEqual(track.weights.shape[1], len(track.names))

    def test_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            build_animation([], 1.0, 0)
        with self.assertRaises(ValueError):
            build_animation([], 0.0, 25)


class TimestampTests(unittest.TestCase):
    def test_accepts_contract_objects_and_dicts(self):
        model = PhonemeTimestamp(phoneme="AA", viseme="viseme_aa", startMs=0, endMs=100)
        self.assertEqual(
            normalise_timestamps([model, ts("viseme_O", 100, 200), {"viseme": "viseme_E", "start_ms": 200, "end_ms": 300}]),
            [("viseme_aa", 0.0, 100.0), ("viseme_O", 100.0, 200.0), ("viseme_E", 200.0, 300.0)],
        )

    def test_hold_fills_short_gaps_but_not_pauses(self):
        segments = [("viseme_aa", 0.0, 20.0), ("viseme_O", 100.0, 120.0), ("viseme_E", 2000.0, 2020.0)]
        held = hold_until_next(segments, duration_ms=2100.0)
        self.assertEqual(held[0], ("viseme_aa", 0.0, 100.0))          # held to the next start
        self.assertEqual(held[1], ("viseme_O", 100.0, 120.0 + MAX_HOLD_MS))  # a real pause: capped
        self.assertEqual(held[2], ("viseme_E", 2000.0, 2100.0))       # last one: to the clip end

    def test_touching_segments_pass_through_unchanged(self):
        segments = [("viseme_aa", 0.0, 100.0), ("viseme_O", 100.0, 200.0)]
        self.assertEqual(hold_until_next(segments, 200.0), segments)

    def test_short_ctc_spans_still_open_the_mouth(self):
        # What the aligner really emits: 20 ms blips with gaps between them.
        sparse = [ts("viseme_aa", 100 + 120 * i, 120 + 120 * i) for i in range(8)]
        track = build_animation(sparse, 1.2, 25)
        self.assertGreater(track.column("jawOpen")[3:25].mean(), 0.4)


class MouthMotionTests(unittest.TestCase):
    def test_open_vowel_opens_then_silence_closes(self):
        track = build_animation([ts("viseme_aa", 0, 500), ts("viseme_sil", 500, 1000)], 1.0, 25)
        jaw = track.column("jawOpen")
        self.assertGreater(jaw[5], 0.55)
        self.assertLess(jaw[-2], 0.02)

    def test_motion_is_smooth_not_stepped(self):
        track = build_animation([ts("viseme_sil", 0, 400), ts("viseme_aa", 400, 800)], 0.8, 50)
        jaw = track.column("jawOpen")
        self.assertLess(np.abs(np.diff(jaw)).max(), 0.35)
        self.assertTrue(np.all(np.diff(jaw[15:25]) >= -1e-6))

    def test_bilabial_between_vowels_closes_the_lips(self):
        visemes = [ts("viseme_aa", 0, 300), ts("viseme_PP", 300, 380), ts("viseme_aa", 380, 700)]
        track = build_animation(visemes, 0.7, 100)
        jaw = track.column("jawOpen")
        self.assertLess(jaw[34], 0.12)
        self.assertGreater(jaw[15], 0.5)
        self.assertGreater(jaw[55], 0.5)

    def test_later_segment_owns_an_overlap_with_a_closure(self):
        visemes = [ts("viseme_PP", 0, 600), ts("viseme_aa", 200, 800)]
        jaw = build_animation(visemes, 0.8, 100).column("jawOpen")
        self.assertGreater(jaw[40], 0.5)   # inside the overlap: the vowel wins
        self.assertLess(jaw[8], 0.05)      # before it: lips closed

    def test_unknown_visemes_are_counted_and_rendered_as_rest(self):
        with self.assertLogs("face_animation", level="WARNING"):
            track = build_animation([ts("viseme_B", 0, 500), ts("viseme_B", 500, 900)], 1.0, 25)
        self.assertEqual(track.unknown_visemes, {"viseme_B": 2})
        self.assertEqual(float(track.column("jawOpen").max()), 0.0)

    def test_weights_stay_in_range(self):
        visemes = [ts(v, i * 100, (i + 1) * 100) for i, v in enumerate(["viseme_aa", "viseme_O", "viseme_U", "viseme_E"])]
        track = build_animation(visemes, 0.4, 30, emotion_vector={"joy": 1.0, "excitement": 1.0, "eyeblinkRate": 3.0})
        self.assertGreaterEqual(float(track.weights.min()), 0.0)
        self.assertLessEqual(float(track.weights.max()), 1.0)


class EnergyGateTests(unittest.TestCase):
    def test_silent_audio_keeps_the_mouth_shut(self):
        visemes = [ts("viseme_aa", 0, 1000)]
        envelope = np.concatenate([np.ones(100), np.zeros(100)]).astype(np.float32)
        track = build_animation(visemes, 1.0, 25, energy_envelope=envelope)
        jaw = track.column("jawOpen")
        self.assertTrue(track.energy_gated)
        self.assertGreater(jaw[5], 0.5)
        self.assertLess(jaw[-3], 0.02)

    def test_mostly_silent_clip_is_not_normalised_to_zero(self):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pause.wav"
            tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(1600) / 16000.0)  # 0.1 s of speech
            sf.write(path, np.concatenate([tone, np.zeros(16000 * 4)]).astype(np.float32), 16000)
            envelope = speech_energy_envelope(path)
        self.assertGreater(envelope[:18].min(), 0.9)

    def test_truly_silent_clip_disables_the_gate_instead_of_closing_the_mouth(self):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "silent.wav"
            sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
            envelope = speech_energy_envelope(path)
        self.assertEqual(len(envelope), 0)
        track = build_animation([ts("viseme_aa", 0, 1000)], 1.0, 25, energy_envelope=envelope)
        self.assertFalse(track.energy_gated)
        self.assertGreater(track.column("jawOpen").max(), 0.5)

    def test_envelope_from_a_wav_is_normalised(self):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav"
            tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(8000) / 16000.0)
            sf.write(path, np.concatenate([tone, np.zeros(8000)]).astype(np.float32), 16000)
            envelope = speech_energy_envelope(path)
        self.assertEqual(len(envelope), 200)  # 1 s on a 5 ms grid
        self.assertGreater(envelope[:95].min(), 0.9)
        self.assertEqual(float(envelope[105:].max()), 0.0)


class BlinkTests(unittest.TestCase):
    def test_schedule_is_deterministic_per_seed(self):
        self.assertEqual(blink_schedule(20.0, 1.0, 7), blink_schedule(20.0, 1.0, 7))
        self.assertNotEqual(blink_schedule(20.0, 1.0, 7), blink_schedule(20.0, 1.0, 8))

    def test_rate_scales_blinks_and_zero_disables_them(self):
        calm = blink_schedule(60.0, 0.6, 1)
        angry = blink_schedule(60.0, 2.4, 1)
        self.assertGreater(len(angry), 2 * len(calm))
        self.assertTrue(10 <= len(blink_schedule(60.0, 1.0, 1)) <= 25)
        self.assertEqual(blink_schedule(60.0, 0.0, 1), [])

    def test_blinks_fit_in_the_clip_and_are_spaced(self):
        times = blink_schedule(30.0, 10.0, 3)
        self.assertLessEqual(times[-1] + BLINK_CLOSE_S + BLINK_OPEN_S, 30.0)
        self.assertGreaterEqual(min(np.diff(times)), MIN_BLINK_INTERVAL_S - 1e-6)

    def test_blinks_close_both_eyes_in_the_track(self):
        track = build_animation([ts("viseme_sil", 0, 9000)], 10.0, 50, emotion_vector={"eyeblinkRate": 1.5}, seed=11)
        self.assertTrue(track.blink_times)
        left, right = track.column("eyeBlinkLeft"), track.column("eyeBlinkRight")
        np.testing.assert_array_equal(left, right)
        self.assertGreater(left.max(), 0.9)
        self.assertLess(np.median(left), 0.05)

    def test_job_seed_is_stable(self):
        self.assertEqual(seed_from_job_id("AVT-9821-X"), seed_from_job_id("AVT-9821-X"))
        self.assertNotEqual(seed_from_job_id("a"), seed_from_job_id("b"))
        self.assertLess(seed_from_job_id("anything"), 2**32)


class EmotionTrackTests(unittest.TestCase):
    def test_joy_and_sorrow_are_visibly_different(self):
        visemes = [ts("viseme_sil", 0, 1000)]
        joy = build_animation(visemes, 1.0, 25, emotion_vector={"joy": 1.0, "eyeblinkRate": 0})
        sorrow = build_animation(visemes, 1.0, 25, emotion_vector={"sorrow": 1.0, "eyeblinkRate": 0})
        self.assertGreater(joy.column("mouthSmileLeft").mean(), 0.4)
        self.assertEqual(float(sorrow.column("mouthSmileLeft").max()), 0.0)
        self.assertGreater(sorrow.column("browInnerUp").mean(), 0.4)

    def test_frame_returns_only_active_shapes(self):
        track = build_animation([ts("viseme_sil", 0, 400)], 0.4, 25, emotion_vector={"eyeblinkRate": 0})
        self.assertEqual(track.frame(0), {})


if __name__ == "__main__":
    unittest.main()
