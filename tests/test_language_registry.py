"""Tests for the MMS-TTS language registry. No model downloads."""

import unittest

import language_registry


class LanguageResolutionTests(unittest.TestCase):
    def test_catalogue_covers_over_one_thousand_languages(self):
        """The roadmap's Phase 3 target is 1000+ languages."""
        self.assertGreater(language_registry.supported_count(), 1000)

    def test_iso639_1_codes_resolve_to_iso639_3(self):
        for code, iso3 in [("en", "eng"), ("hi", "hin"), ("ta", "tam"), ("sw", "swa")]:
            with self.subTest(code=code):
                self.assertEqual(language_registry.to_iso3(code), iso3)

    def test_region_subtags_are_stripped(self):
        self.assertEqual(language_registry.to_iso3("pt-BR"), "por")
        self.assertEqual(language_registry.to_iso3("en_US"), "eng")

    def test_iso639_3_codes_pass_through(self):
        self.assertEqual(language_registry.to_iso3("swh"), "swh")

    def test_unknown_code_resolves_without_raising(self):
        info = language_registry.resolve("zz-nope")
        self.assertFalse(info.mms_supported)
        self.assertIsNone(info.mms_model)

    def test_supported_language_carries_a_model_id(self):
        info = language_registry.resolve("hi")
        self.assertTrue(info.mms_supported)
        self.assertEqual(info.mms_model, "facebook/mms-tts-hin")
        self.assertEqual(info.name, "Hindi")

    def test_languages_outside_mms_are_reported_honestly(self):
        """MMS-TTS genuinely has no Japanese, Mandarin or Italian checkpoint."""
        for code in ("ja", "zh", "it"):
            with self.subTest(code=code):
                info = language_registry.resolve(code)
                self.assertFalse(info.mms_supported)
                self.assertIsNone(info.mms_model)

    def test_english_detection(self):
        for code in ("en", "en-US", "en_GB", "eng"):
            with self.subTest(code=code):
                self.assertTrue(language_registry.is_english(code))
        self.assertFalse(language_registry.is_english("hi"))


class LanguageSearchTests(unittest.TestCase):
    def test_search_matches_names_case_insensitively(self):
        results = language_registry.search("tamil")
        self.assertTrue(any(info.iso3 == "tam" for info in results))

    def test_search_matches_codes(self):
        results = language_registry.search("swh")
        self.assertTrue(any(info.iso3 == "swh" for info in results))

    def test_search_respects_limit(self):
        self.assertEqual(len(language_registry.search("", limit=7)), 7)

    def test_search_results_are_name_sorted(self):
        names = [info.name for info in language_registry.search("", limit=25)]
        self.assertEqual(names, sorted(names))

    def test_unmatched_query_returns_nothing(self):
        self.assertEqual(language_registry.search("zzzzz-not-a-language"), [])


if __name__ == "__main__":
    unittest.main()
