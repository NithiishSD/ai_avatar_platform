"""Mouth shapes estimated from sound (I-02): each sound family maps to its viseme, runs are merged."""

import unittest

import numpy as np

import audio_visemes

RATE = 16000


def tone(freq, seconds=0.2, level=0.3):
    t = np.arange(int(RATE * seconds)) / RATE
    return (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class ClassifyTests(unittest.TestCase):
    def test_each_sound_family_gets_its_mouth_shape(self):
        # White noise differenced: energy pushed high, a sign change at most samples, like an "s".
        hiss = np.diff(np.random.default_rng(0).normal(scale=0.2, size=int(RATE * 0.2)), prepend=0).astype(np.float32)
        cases = {
            "viseme_sil": np.zeros(int(RATE * 0.2), np.float32),
            "viseme_O": tone(300),        # energy kept low: rounded vowel
            "viseme_aa": tone(1000),      # middle: open vowel
            "viseme_E": tone(2200),       # higher: front vowel
            "viseme_SS": hiss,            # noise, mostly high, many zero crossings: fricative
        }
        for expected, sound in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(audio_visemes.classify(sound, RATE), expected)

    def test_room_noise_below_the_gate_is_silence(self):
        quiet = (np.random.default_rng(1).normal(scale=0.002, size=RATE)).astype(np.float32)
        self.assertEqual(audio_visemes.classify(quiet, RATE), "viseme_sil")


class TimestampTests(unittest.TestCase):
    def test_runs_are_merged_and_cover_the_chunk(self):
        audio = np.concatenate([np.zeros(int(RATE * 0.2), np.float32), tone(300, 0.4), tone(2200, 0.4)])
        segments = audio_visemes.estimate_timestamps(audio, RATE)
        self.assertEqual([s["viseme"] for s in segments], ["viseme_sil", "viseme_O", "viseme_E"])
        self.assertEqual(segments[0]["startMs"], 0)
        self.assertAlmostEqual(segments[-1]["endMs"], 1000.0, places=3)
        for a, b in zip(segments, segments[1:], strict=False):
            self.assertEqual(a["endMs"], b["startMs"])  # no gaps, no overlaps


if __name__ == "__main__":
    unittest.main()
