import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from voice_engine import VoiceEngineRouter


class RouterSelectionTests(unittest.TestCase):
    """Tests for the routing decision matrix — no models loaded."""

    def setUp(self):
        self.router = VoiceEngineRouter(device="cpu")

    # ------------------------------------------------------------------ Kokoro
    def test_fast_english_routes_to_kokoro(self):
        self.assertEqual(self.router.select_model(mode="fast", language="en"), "kokoro")

    def test_fast_en_us_routes_to_kokoro(self):
        self.assertEqual(self.router.select_model(mode="fast", language="en-US"), "kokoro")

    def test_fast_en_gb_routes_to_kokoro(self):
        self.assertEqual(self.router.select_model(mode="fast", language="en-gb"), "kokoro")

    # ------------------------------------------------------------------ XTTS-v2
    def test_clone_mode_routes_to_xtts(self):
        self.assertEqual(self.router.select_model(mode="clone"), "xtts-v2")

    def test_clone_mode_overrides_language(self):
        # Clone always goes to XTTS regardless of language
        self.assertEqual(self.router.select_model(mode="clone", language="fr"), "xtts-v2")

    # ------------------------------------------------------------------ Higgs TTS 2
    def test_high_quality_mode_routes_to_higgs(self):
        self.assertEqual(self.router.select_model(mode="high_quality"), "higgs-tts-2")

    def test_quality_high_routes_to_higgs(self):
        self.assertEqual(self.router.select_model(mode="fast", language="en", quality="high"), "higgs-tts-2")

    def test_non_english_with_mms_coverage_routes_to_mms(self):
        """Phase 3: MMS-TTS covers these, so it wins over Higgs for plain synthesis."""
        for code in ("es", "fr", "de", "hi", "ta", "swh"):
            with self.subTest(language=code):
                self.assertEqual(
                    self.router.select_model(mode="fast", language=code), "mms-tts"
                )

    def test_non_english_without_mms_coverage_routes_to_higgs(self):
        """Japanese, Mandarin and Italian have no MMS-TTS checkpoint."""
        for code in ("ja", "zh", "it"):
            with self.subTest(language=code):
                self.assertEqual(
                    self.router.select_model(mode="fast", language=code), "higgs-tts-2"
                )

    # ------------------------------------------------------------------ Dia-1.6B
    def test_dialogue_mode_routes_to_dia(self):
        self.assertEqual(self.router.select_model(mode="dialogue"), "dia-1.6b")

    def test_style_dialogue_routes_to_dia(self):
        self.assertEqual(self.router.select_model(mode="fast", language="en", style="dialogue"), "dia-1.6b")

    def test_speaker_tags_route_to_dia(self):
        self.assertEqual(
            self.router.select_model(mode="fast", language="en", text="[S1] Hello [S2] Hi"),
            "dia-1.6b",
        )

    # ------------------------------------------------------------------ Fallbacks
    def test_higgs_failed_falls_back_to_xtts_for_high_quality(self):
        self.router._higgs_failed = True
        self.assertEqual(self.router.select_model(mode="high_quality"), "xtts-v2")

    def test_higgs_failed_still_routes_mms_covered_language_to_mms(self):
        self.router._higgs_failed = True
        self.assertEqual(self.router.select_model(mode="fast", language="es"), "mms-tts")

    def test_higgs_failed_raises_for_language_without_mms_coverage(self):
        self.router._higgs_failed = True
        with self.assertRaises(ValueError):
            self.router.select_model(mode="fast", language="ja")

    def test_both_multilingual_backends_failed_raises(self):
        self.router._higgs_failed = True
        self.router._mms_failed = True
        with self.assertRaises(ValueError):
            self.router.select_model(mode="fast", language="es")

    def test_dia_failed_falls_back_to_kokoro_for_dialogue(self):
        self.router._dia_failed = True
        self.assertEqual(self.router.select_model(mode="dialogue"), "kokoro")

    # ------------------------------------------------------------------ Errors
    def test_unsupported_mode_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.router.select_model(mode="invalid_mode")


class KokoroSynthesisTests(unittest.TestCase):
    """Tests for Kokoro fast synthesis path (model mocked)."""

    def setUp(self):
        self.router = VoiceEngineRouter(device="cpu")
        samples = np.zeros(2400, dtype=np.float32)
        self.router.kokoro_pipeline = lambda text, voice, speed: iter([(None, None, samples)])
        self.output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_phase1"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        import voice_engine
        self._orig_dir = voice_engine.OUTPUT_DIR
        voice_engine.OUTPUT_DIR = self.output_dir

    def tearDown(self):
        import voice_engine
        voice_engine.OUTPUT_DIR = self._orig_dir

    def test_fast_synthesis_returns_structured_metadata(self):
        result = self.router.synthesize("Test speech")
        self.assertEqual(result.model, "kokoro")
        self.assertEqual(result.mode, "fast")
        self.assertEqual(result.sample_rate, 24000)
        self.assertAlmostEqual(result.duration_seconds, 0.1)
        self.assertTrue(Path(result.output_path).exists())
        self.assertEqual(sf.info(result.output_path).samplerate, 24000)

    def test_high_quality_with_higgs_failed_raises_without_speaker_wav(self):
        """When Higgs fails, routing to XTTS-v2; without speaker_wav it raises FileNotFoundError."""
        self.router._higgs_failed = True
        with self.assertRaises(FileNotFoundError):
            self.router.synthesize("High quality test", mode="high_quality")

    def test_synthesis_rejects_empty_text(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.router.synthesize("   ")

    def test_dialogue_with_dia_failed_falls_back_to_kokoro(self):
        """Dia is unavailable; dialogue falls back to Kokoro (speaker tags stripped)."""
        self.router._dia_failed = True
        result = self.router.synthesize("[S1] Hello there [S2] Good morning", mode="dialogue")
        self.assertEqual(result.model, "kokoro")


class HiggsLoadingTests(unittest.TestCase):
    """Tests for Higgs model loading guard."""

    def test_higgs_load_failure_marks_flag(self):
        router = VoiceEngineRouter(device="cpu")
        # Patch transformers pipeline to raise so load fails
        with patch("voice_engine.VoiceEngineRouter.load_higgs", return_value=False):
            router._higgs_failed = True
            result = router.select_model(mode="high_quality")
            self.assertEqual(result, "xtts-v2")


class DiaLoadingTests(unittest.TestCase):
    """Tests for Dia model loading guard."""

    def test_dia_load_failure_falls_back_to_kokoro_routing(self):
        router = VoiceEngineRouter(device="cpu")
        router._dia_failed = True
        result = router.select_model(mode="dialogue")
        self.assertEqual(result, "kokoro")


if __name__ == "__main__":
    unittest.main()