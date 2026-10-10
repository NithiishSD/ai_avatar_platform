"""
Tests for the shared romanizer's result check.

uroman is typed to return either text or a list of edges. We only ever ask
for text, so anything else must fail loudly instead of reaching a tokenizer.
"""

import unittest
from unittest import mock

import romanizer


class AsTextTests(unittest.TestCase):
    def test_text_passes_through_unchanged(self):
        self.assertEqual(romanizer.as_text("namaste"), "namaste")

    def test_a_non_string_result_is_rejected(self):
        with self.assertRaises(TypeError):
            romanizer.as_text([("edge", 0, 1)])


class RomanizeTests(unittest.TestCase):
    def tearDown(self):
        romanizer.reset_cache()

    def test_a_list_from_uroman_raises_rather_than_returning_it(self):
        fake = mock.Mock()
        fake.romanize_string.return_value = ["not", "text"]
        with mock.patch.object(romanizer, "get_romanizer", return_value=fake):
            with self.assertRaises(TypeError):
                romanizer.romanize("नमस्ते", lcode="hin")
        # Exactly one call: the bad result is rejected, not mistaken for an
        # old uroman signature and retried without the language hint.
        self.assertEqual(fake.romanize_string.call_count, 1)

    def test_old_uroman_without_lcode_still_works(self):
        fake = mock.Mock()
        fake.romanize_string.side_effect = [TypeError("no lcode"), "namaste"]
        with mock.patch.object(romanizer, "get_romanizer", return_value=fake):
            self.assertEqual(romanizer.romanize("नमस्ते", lcode="hin"), "namaste")
        self.assertEqual(fake.romanize_string.call_count, 2)

    def test_missing_romanizer_returns_none(self):
        with mock.patch.object(romanizer, "get_romanizer", return_value=None):
            self.assertIsNone(romanizer.romanize("नमस्ते"))
