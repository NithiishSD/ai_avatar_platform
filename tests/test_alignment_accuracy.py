"""
Tests for the CTC forced-alignment path.

The MMS_FA branch used to compute a CTC alignment and then throw it away,
emitting evenly-spaced timestamps identical to the offline fallback. Developer
2's >95% lip-sync target depends entirely on these timings, so these tests pin
the properties that distinguish a real alignment from an even split.
"""

import unittest
from unittest import mock

import torch

from alignment_engine import (
    DIGRAPH_TO_PHONEME,
    ForcedAligner,
    PhonemeToVisemeMapper,
    graphemes_to_phonemes,
)
from contracts import AvatarRenderJob, PhonemeTimestamp


class GraphemeToPhonemeTests(unittest.TestCase):
    def test_character_counts_always_cover_the_whole_word(self):
        """Spans and phonemes must never drift apart."""
        for word in ["hello", "ship", "time", "nothing", "phone", "queue", "a", "rhythm"]:
            with self.subTest(word=word):
                groups = graphemes_to_phonemes(word)
                self.assertEqual(sum(count for _, count in groups), len(word))

    def test_digraphs_become_one_phoneme(self):
        self.assertEqual(graphemes_to_phonemes("ship")[0], ("SH", 2))
        self.assertEqual(graphemes_to_phonemes("thin")[0], ("TH", 2))
        self.assertEqual(graphemes_to_phonemes("chip")[0], ("CH", 2))

    def test_silent_final_e_folds_into_the_previous_phoneme(self):
        groups = graphemes_to_phonemes("time")
        self.assertEqual(groups[-1], ("M", 2))
        self.assertEqual(len(groups), 3)

    def test_every_phoneme_maps_to_a_canonical_viseme(self):
        valid = set(PhonemeToVisemeMapper.get_supported_visemes())
        for word in ["hello", "world", "shrimp", "though", "quickly"]:
            for phoneme, _ in graphemes_to_phonemes(word):
                with self.subTest(word=word, phoneme=phoneme):
                    self.assertIn(PhonemeToVisemeMapper.map_phoneme(phoneme), valid)

    def test_digraph_table_only_holds_two_character_keys(self):
        for key in DIGRAPH_TO_PHONEME:
            self.assertEqual(len(key), 2)

    def test_empty_word_yields_nothing(self):
        self.assertEqual(graphemes_to_phonemes(""), [])


class AlignableWordTests(unittest.TestCase):
    def test_punctuation_and_digits_are_stripped(self):
        words = ForcedAligner._alignable_words("Hello, world! 42 times.")
        self.assertEqual(words, ["hello", "world", "times"])

    def test_apostrophes_are_kept_inside_words(self):
        self.assertEqual(ForcedAligner._alignable_words("don't"), ["don't"])

    def test_non_latin_text_yields_no_alignable_words(self):
        """Raw non-Latin script is unalignable, which is why it gets romanized first."""
        self.assertEqual(ForcedAligner._alignable_words("नमस्ते"), [])


class RomanizationForAlignmentTests(unittest.TestCase):
    """
    Phase 3 ships MMS-TTS audio in 1000+ languages, but both aligners work over
    a Latin alphabet. Without transliteration a Hindi or Tamil transcript
    reduces to zero alignable words and the timings stop describing the speech,
    so Developer 2 cannot lip-sync anything but English.
    """

    def test_ascii_text_is_returned_untouched(self):
        self.assertEqual(
            ForcedAligner._prepare_for_alignment("hello world", "en"),
            "hello world",
        )

    def test_empty_text_is_returned_untouched(self):
        self.assertEqual(ForcedAligner._prepare_for_alignment("", "hin"), "")

    def test_non_latin_text_becomes_alignable(self):
        with mock.patch(
            "alignment_engine.romanize", return_value="namaste duniyaa"
        ) as romanize:
            prepared = ForcedAligner._prepare_for_alignment("नमस्ते दुनिया", "hin")
        self.assertEqual(prepared, "namaste duniyaa")
        self.assertEqual(ForcedAligner._alignable_words(prepared), ["namaste", "duniyaa"])
        romanize.assert_called_once_with("नमस्ते दुनिया", lcode="hin")

    def test_language_code_is_normalized_to_iso3_for_the_romanizer(self):
        """uroman transliterates more accurately with an ISO-639-3 hint."""
        with mock.patch("alignment_engine.romanize", return_value="namaste") as romanize:
            ForcedAligner._prepare_for_alignment("नमस्ते", "hi")
        self.assertEqual(romanize.call_args.kwargs["lcode"], "hin")

    def test_missing_romanizer_falls_back_to_the_original_text(self):
        """Without uroman the aligner must degrade, not crash."""
        with mock.patch("alignment_engine.romanize", return_value=None):
            self.assertEqual(
                ForcedAligner._prepare_for_alignment("नमस्ते", "hin"), "नमस्ते"
            )

    def test_blank_romanization_is_rejected(self):
        with mock.patch("alignment_engine.romanize", return_value="   "):
            self.assertEqual(
                ForcedAligner._prepare_for_alignment("नमस्ते", "hin"), "नमस्ते"
            )


class Span:
    """Stands in for torchaudio's TokenSpan."""

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end


class CtcSpanTests(unittest.TestCase):
    """The MMS branch with stubbed torchaudio, so no model is downloaded."""

    def setUp(self):
        self.aligner = ForcedAligner(device="cpu")
        self.transcript = "hi there"
        # "hi there" is 7 alignable characters -> 7 CTC spans. The frame counts
        # are deliberately uneven so an even split is distinguishable from a
        # real alignment, and there is a silent gap between the two words.
        self.spans = [
            Span(0, 5), Span(5, 8),                          # h, i
            Span(20, 24), Span(24, 26), Span(26, 40),        # t, h, e
            Span(40, 42), Span(42, 50),                      # r, e
        ]

    def run_align(self, duration_ms: int = 1000, spans=None):
        spans = spans if spans is not None else self.spans
        emission = torch.zeros(1, 100, 29)
        model = mock.Mock(return_value=(emission, None))
        tokenizer = mock.Mock(
            side_effect=lambda words: [[1] * len(word) for word in words]
        )
        import torchaudio.functional as AF

        # Patch the three torchaudio calls in place rather than replacing the
        # module, so the real import inside _align_mms still resolves.
        with mock.patch.object(AF, "resample", side_effect=lambda w, **kw: w), \
             mock.patch.object(
                 AF,
                 "forced_align",
                 return_value=(
                     torch.zeros(1, 100, dtype=torch.int32),
                     torch.zeros(1, 100),
                 ),
             ), \
             mock.patch.object(AF, "merge_tokens", return_value=spans):
            return self.aligner._align_mms(
                waveform=torch.zeros(1, 16000),
                sample_rate=16000,
                transcript=self.transcript,
                model=model,
                tokenizer=tokenizer,
                duration_ms=duration_ms,
            )

    def test_timestamps_are_not_evenly_spaced(self):
        """The regression this path was written to fix."""
        stamps = self.run_align()
        durations = {stamp.end_ms - stamp.start_ms for stamp in stamps}
        self.assertGreater(len(durations), 1)

    def test_timestamps_follow_the_ctc_frame_spans(self):
        stamps = self.run_align(duration_ms=1000)
        # 100 frames over 1000 ms -> 10 ms per frame.
        first = next(s for s in stamps if s.phoneme != "SIL")
        self.assertEqual(first.start_ms, 0)
        self.assertEqual(first.end_ms, 50)  # span 0..5 frames

    def test_a_gap_between_words_becomes_an_explicit_silence(self):
        stamps = self.run_align(duration_ms=1000)
        # "hi" ends at frame 8 (80 ms), "there" starts at frame 20 (200 ms).
        silences = [s for s in stamps if s.phoneme == "SIL"]
        self.assertTrue(silences)
        self.assertTrue(any(s.viseme == "viseme_sil" for s in silences))

    def test_result_is_ordered_and_non_overlapping(self):
        stamps = self.run_align()
        for previous, current in zip(stamps, stamps[1:], strict=False):
            self.assertLessEqual(previous.end_ms, current.start_ms)
            self.assertLess(current.start_ms, current.end_ms)

    def test_result_never_exceeds_the_audio_duration(self):
        stamps = self.run_align(duration_ms=400)
        self.assertLessEqual(stamps[-1].end_ms, 400)

    def test_span_count_mismatch_raises_instead_of_guessing(self):
        with self.assertRaises(RuntimeError):
            self.run_align(spans=self.spans[:3])

    def test_output_builds_a_valid_render_job(self):
        """The whole point: Developer 2 must be able to consume this directly."""
        stamps = self.run_align(duration_ms=1000)
        job = AvatarRenderJob.model_validate(
            {
                "jobId": "JOB-ALIGN-1",
                "avatarId": "AVATAR_01",
                "audioUrl": "https://example.com/a.wav",
                "sampleRate": 24000,
                "durationSeconds": 1.0,
                "phonemeTimestamps": [s.model_dump(by_alias=True) for s in stamps],
                "emotionVector": {"happy": 0.5, "neutral": 0.5, "eyeblinkRate": 1.0},
                "renderQuality": "1080P_HQ",
                "targetFps": 30,
            }
        )
        self.assertEqual(len(job.phoneme_timestamps), len(stamps))


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        self.aligner = ForcedAligner(device="cpu")

    def test_overlaps_are_removed(self):
        raw = [
            PhonemeTimestamp(phoneme="AA", viseme="viseme_aa", startMs=0, endMs=200),
            PhonemeTimestamp(phoneme="B", viseme="viseme_PP", startMs=100, endMs=300),
        ]
        fixed = self.aligner._normalize_timestamps(raw, 1000)
        self.assertGreaterEqual(fixed[1].start_ms, fixed[0].end_ms)

    def test_empty_input_yields_a_single_silence(self):
        fixed = self.aligner._normalize_timestamps([], 500)
        self.assertEqual(len(fixed), 1)
        self.assertEqual(fixed[0].phoneme, "SIL")

    def test_trailing_gap_is_padded_with_silence(self):
        raw = [PhonemeTimestamp(phoneme="AA", viseme="viseme_aa", startMs=0, endMs=100)]
        fixed = self.aligner._normalize_timestamps(raw, 1000)
        self.assertEqual(fixed[-1].phoneme, "SIL")
        self.assertEqual(fixed[-1].end_ms, 1000)


if __name__ == "__main__":
    unittest.main()
