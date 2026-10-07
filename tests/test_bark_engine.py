"""
Tests for Bark dialogue assembly.

The model is faked, so nothing is downloaded or loaded. What is tested is our
side: splitting a script into speaker turns, keeping each generation short
enough for Bark, giving each speaker their own preset, joining the turns, and
a load failure that names its fix.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from bark_engine import (
    DIALOGUE_VOICES,
    MAX_CHARS_PER_GENERATION,
    TURN_GAP_SECONDS,
    BarkEngine,
    BarkUnavailable,
    chunk_turn,
    split_dialogue,
)


class SplitDialogueTests(unittest.TestCase):
    def test_turns_follow_the_tags(self):
        self.assertEqual(
            split_dialogue("[S1] Hi there. [S2] Hello! [S1] Bye."),
            [(1, "Hi there."), (2, "Hello!"), (1, "Bye.")],
        )

    def test_untagged_text_is_speaker_one(self):
        self.assertEqual(split_dialogue("Just one voice."), [(1, "Just one voice.")])

    def test_text_before_the_first_tag_is_kept(self):
        self.assertEqual(split_dialogue("Intro. [S2] Reply."), [(1, "Intro."), (2, "Reply.")])

    def test_lowercase_tags_and_empty_turns(self):
        self.assertEqual(split_dialogue("[s1] [S2] Only two speaks."), [(2, "Only two speaks.")])


class ChunkTurnTests(unittest.TestCase):
    def test_short_turns_are_one_generation(self):
        self.assertEqual(chunk_turn("A short line."), ["A short line."])

    def test_long_turns_split_at_sentence_ends_within_the_limit(self):
        sentence = "This sentence is about sixty characters long, more or less ok."
        pieces = chunk_turn(" ".join([sentence] * 8))
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(p) <= MAX_CHARS_PER_GENERATION for p in pieces))
        self.assertEqual(" ".join(pieces), " ".join([sentence] * 8))  # nothing dropped


class DialogueSynthesisTests(unittest.TestCase):
    def test_each_speaker_gets_their_preset_and_turns_are_joined(self):
        engine = BarkEngine(device="cpu")
        calls = []

        def fake_generate(text, voice, seed):
            calls.append((text, voice, seed))
            return np.full(24000, 0.1, dtype=np.float32), 24000  # one second each

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(engine, "_generate", side_effect=fake_generate):
            out = Path(tmp) / "dialogue.wav"
            rate, duration, turns = engine.synthesize_dialogue("[S1] Hi. [S2] Hello.", out)
            written = sf.info(str(out))
        self.assertEqual([c[1] for c in calls], [DIALOGUE_VOICES[1], DIALOGUE_VOICES[2]])
        self.assertEqual(len({c[2] for c in calls}), 2)  # a different seed per piece
        self.assertEqual((rate, turns), (24000, 2))
        self.assertAlmostEqual(duration, 2 + TURN_GAP_SECONDS, places=3)
        self.assertEqual(written.samplerate, 24000)

    def test_a_script_with_only_tags_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                BarkEngine(device="cpu").synthesize_dialogue("[S1] [S2]", Path(tmp) / "x.wav")


class LoadTests(unittest.TestCase):
    def test_a_load_failure_names_the_fix_is_cached_and_stays_offline(self):
        engine = BarkEngine(device="cpu")
        with mock.patch("huggingface_hub.snapshot_download", side_effect=FileNotFoundError("none")) as fetch:
            with self.assertRaises(BarkUnavailable) as first:
                engine._load()
            with self.assertRaises(BarkUnavailable):
                engine._load()
        self.assertIn("fetch_models.py --only bark", str(first.exception))
        self.assertEqual(fetch.call_count, 1)
        self.assertTrue(fetch.call_args.kwargs["local_files_only"])

    def test_a_missing_preset_names_the_fix(self):
        engine = BarkEngine(device="cpu")
        with tempfile.TemporaryDirectory() as tmp:
            engine._snapshot = Path(tmp)
            with self.assertRaises(BarkUnavailable) as caught:
                engine._preset("v2/en_speaker_6")
        self.assertIn("fetch_models.py --only bark", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
