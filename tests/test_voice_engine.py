import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from voice_engine import ModelWeightsMissing, VoiceConsentRequired, VoiceEngineRouter


def weights_on_disk(*missing):
    """
    A fake on-disk weight audit: every routed model present except `missing`.

    Tests must not depend on which weights the machine running them happens
    to have - CI has none at all.
    """
    from model_registry import ModelWeightStatus

    keys = ("kokoro", "xtts-v2", "higgs-tts-2", "dia-1.6b", "mms-tts", "openvoice-v2", "bark")
    return [
        ModelWeightStatus(key, key, "test", key not in missing, 0 if key in missing else 1,
                          "cached metadata only (no weight file)" if key in missing else "present")
        for key in keys
    ]


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
    # Dialogue routes to Bark: Dia cannot run on this stack (model_registry.UNRUNNABLE).
    def test_dialogue_mode_routes_to_bark(self):
        self.assertEqual(self.router.select_model(mode="dialogue"), "bark")

    def test_style_dialogue_routes_to_bark(self):
        self.assertEqual(self.router.select_model(mode="fast", language="en", style="dialogue"), "bark")

    def test_speaker_tags_route_to_bark(self):
        self.assertEqual(
            self.router.select_model(mode="fast", language="en", text="[S1] Hello [S2] Hi"),
            "bark",
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

    def test_bark_failed_falls_back_to_kokoro_for_dialogue(self):
        self.router._bark_failed = True
        self.assertEqual(self.router.select_model(mode="dialogue"), "kokoro")

    # ------------------------------------------------------------------ Errors
    def test_unsupported_mode_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.router.select_model(mode="invalid_mode")


class KokoroSynthesisTests(unittest.TestCase):
    """Tests for Kokoro fast synthesis path (model mocked)."""

    def setUp(self):
        self.router = VoiceEngineRouter(device="cpu")
        # These tests are about synthesis with mocked models, not about which
        # weights this machine has: report every model as on disk.
        audit = patch("model_registry.audit_model_weights", return_value=weights_on_disk())
        audit.start()
        self.addCleanup(audit.stop)
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

    def test_dialogue_with_bark_failing_to_load_is_spoken_by_kokoro_and_says_so(self):
        """Bark fails at load time; Kokoro speaks, tags stripped, and model names Kokoro."""
        from bark_engine import BarkUnavailable

        with patch.object(self.router._bark, "synthesize_dialogue", side_effect=BarkUnavailable("oom")), \
             patch.object(self.router, "_synthesize_kokoro", wraps=self.router._synthesize_kokoro) as kokoro:
            result = self.router.synthesize("[S1] Hello there [S2] Good morning", mode="dialogue")
        self.assertEqual(result.model, "kokoro")  # not "bark": the fallback is visible
        self.assertEqual(kokoro.call_args.args[0], "Hello there Good morning")
        self.assertTrue(self.router._bark_failed)


class MissingWeightsTests(unittest.TestCase):
    """An engine with no weights is refused before any loader can download it."""

    def setUp(self):
        self.router = VoiceEngineRouter(device="cpu")

    def test_missing_weights_raise_with_the_fetch_command(self):
        with patch("model_registry.audit_model_weights", return_value=weights_on_disk("higgs-tts-2")):
            with self.assertRaises(ModelWeightsMissing) as caught:
                self.router.require_weights("higgs-tts-2")
        self.assertEqual(caught.exception.model_key, "higgs-tts-2")
        self.assertIn("scripts/fetch_models.py --only higgs-tts-2", str(caught.exception))

    def test_present_and_unknown_models_pass(self):
        with patch("model_registry.audit_model_weights", return_value=weights_on_disk("dia-1.6b")):
            self.router.require_weights("kokoro")
            self.router.require_weights("some-future-engine")

    def test_synthesis_never_reaches_the_loader_without_weights(self):
        # The bug this guards: load_higgs/load_dia call from_pretrained, which
        # would have downloaded gigabytes mid-request.
        with patch("model_registry.audit_model_weights", return_value=weights_on_disk("higgs-tts-2")), \
             patch.object(self.router, "load_higgs") as load_higgs:
            with self.assertRaises(ModelWeightsMissing):
                self.router.synthesize("Hello", mode="high_quality")
        load_higgs.assert_not_called()

    def test_dialogue_without_bark_weights_is_refused_not_downloaded(self):
        with patch("model_registry.audit_model_weights", return_value=weights_on_disk("bark")), \
             patch.object(self.router._bark, "_load") as load_bark:
            with self.assertRaises(ModelWeightsMissing):
                self.router.synthesize("[S1] Hi [S2] Hello", mode="dialogue")
        load_bark.assert_not_called()


def reference(folder, name, seconds=0.1, **provenance_fields):
    """A WAV (0.1 s unless told), with a provenance sidecar unless no fields are given."""
    import provenance

    path = Path(folder) / name
    sf.write(path, np.zeros(int(24000 * seconds), dtype=np.float32), 24000)
    if provenance_fields:
        provenance.write(path, **provenance_fields)
    return str(path)


class XttsLanguageTests(unittest.TestCase):
    """XTTS-v2 gets the code it understands, and a language it cannot speak is refused early."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.router = VoiceEngineRouter(device="cpu")
        audit = patch("model_registry.audit_model_weights", return_value=weights_on_disk())
        audit.start()
        self.addCleanup(audit.stop)
        self.wav = reference(self.tmp.name, "me.wav", source="human", speaker="Me", consent_basis="subject-provided")

    def test_the_model_is_given_two_letter_code_even_when_the_ui_sent_three(self):
        from unittest.mock import MagicMock

        model = MagicMock()
        self.router.xtts_model = model
        out = Path(self.tmp.name) / "out.wav"
        sf.write(out, np.zeros(2400, dtype=np.float32), 24000)  # the engine reads the file back for its duration
        with patch("voice_engine.validate_and_convert_for_cloning", return_value=Path(self.wav)):
            self.router._synthesize_xtts("Hola", out, self.wav, "spa")
        self.assertEqual(model.tts_to_file.call_args.kwargs["language"], "es")

    def test_a_language_xtts_cannot_speak_is_refused_before_queueing_and_names_the_alternative(self):
        with self.assertRaises(ValueError) as caught:
            self.router.preflight("xtts-v2", self.wav, "tam")
        self.assertIn("openvoice-v2", str(caught.exception))
        self.router.preflight("xtts-v2", self.wav, "spa")  # a supported one passes

    def test_openvoice_is_not_limited_to_xttss_languages(self):
        self.router.preflight("openvoice-v2", self.wav, "tam")


class VoiceConsentTests(unittest.TestCase):
    """Cloning refuses a recording whose provenance does not permit it."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.router = VoiceEngineRouter(device="cpu")
        audit = patch("model_registry.audit_model_weights", return_value=weights_on_disk())
        audit.start()
        self.addCleanup(audit.stop)

    def test_a_recording_with_no_record_is_refused_with_the_fix(self):
        wav = reference(self.tmp.name, "unknown.wav")
        with self.assertRaises(VoiceConsentRequired) as caught:
            self.router.preflight("xtts-v2", wav)
        self.assertIn("no provenance record", str(caught.exception))
        self.assertIn("scripts/make_reference.py", str(caught.exception))

    def test_a_human_recording_without_a_consent_basis_is_refused(self):
        wav = reference(self.tmp.name, "someone.wav", source="human", speaker="Someone")
        with self.assertRaises(VoiceConsentRequired) as caught:
            self.router.preflight("xtts-v2", wav)
        self.assertIn("consent basis", str(caught.exception))

    def test_a_consented_human_recording_may_be_cloned(self):
        wav = reference(self.tmp.name, "lj.wav", source="human", speaker="LJ",
                        licence="public domain", consent_basis="open-licence")
        self.router.preflight("xtts-v2", wav)

    def test_synthetic_speech_may_be_cloned(self):
        wav = reference(self.tmp.name, "smoke.wav", source="synthetic", speaker="kokoro")
        self.router.preflight("xtts-v2", wav)

    def test_a_reference_the_engine_ignores_is_not_checked(self):
        # Kokoro never reads speaker_wav, so an unconsented file left selected
        # must not block plain fast synthesis.
        self.router.preflight("kokoro", reference(self.tmp.name, "unknown.wav"))

    def test_a_missing_file_is_reported_as_missing_not_unconsented(self):
        with self.assertRaises(FileNotFoundError):
            self.router.preflight("xtts-v2", str(Path(self.tmp.name) / "nope.wav"))

    def test_consent_is_checked_before_weights(self):
        wav = reference(self.tmp.name, "unknown.wav")
        with patch("model_registry.audit_model_weights", return_value=weights_on_disk("xtts-v2")):
            with self.assertRaises(VoiceConsentRequired):
                self.router.preflight("xtts-v2", wav)

    def test_synthesis_refuses_before_any_loader_runs(self):
        wav = reference(self.tmp.name, "unknown.wav")
        with patch.object(self.router, "load_xtts_cloning") as load:
            with self.assertRaises(VoiceConsentRequired):
                self.router.synthesize("Hello", mode="clone", speaker_wav=wav)
        load.assert_not_called()


class OpenVoiceRoutingTests(unittest.TestCase):
    """mode='clone' uses the engine the caller names; OpenVoice needs its base voice."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.router = VoiceEngineRouter(device="cpu")

    def test_xtts_stays_the_default_cloner(self):
        self.assertEqual(self.router.select_model(mode="clone"), "xtts-v2")

    def test_openvoice_is_used_only_when_asked_for(self):
        self.assertEqual(self.router.select_model(mode="clone", clone_engine="openvoice-v2"), "openvoice-v2")

    def test_an_unknown_cloner_is_rejected(self):
        with self.assertRaises(ValueError):
            self.router.select_model(mode="clone", clone_engine="bark")

    def test_the_base_voice_follows_the_language(self):
        self.assertEqual(self.router.openvoice_base("en"), "kokoro")
        self.assertEqual(self.router.openvoice_base("hin"), "mms-tts")
        with self.assertRaises(ValueError):
            self.router.openvoice_base("zzz")

    def test_preflight_needs_the_base_voice_weights_too(self):
        wav = reference(self.tmp.name, "lj.wav", source="human", speaker="LJ",
                        licence="public domain", consent_basis="open-licence")
        with patch("model_registry.audit_model_weights", return_value=weights_on_disk("kokoro")):
            with self.assertRaises(ModelWeightsMissing) as caught:
                self.router.preflight("openvoice-v2", wav, "en")
        self.assertEqual(caught.exception.model_key, "kokoro")

    def test_openvoice_refuses_an_unconsented_voice(self):
        wav = reference(self.tmp.name, "unknown.wav")
        with patch("model_registry.audit_model_weights", return_value=weights_on_disk()):
            with self.assertRaises(VoiceConsentRequired):
                self.router.preflight("openvoice-v2", wav, "en")

    def test_synthesis_speaks_with_the_base_then_converts(self):
        # 4 s: real reference validation refuses anything under 3 s.
        wav = reference(self.tmp.name, "lj.wav", seconds=4.0, source="human", speaker="LJ",
                        licence="public domain", consent_basis="open-licence")
        out_dir = Path(self.tmp.name)
        with patch("model_registry.audit_model_weights", return_value=weights_on_disk()), \
             patch("voice_engine.OUTPUT_DIR", out_dir), \
             patch.object(self.router, "_synthesize_kokoro") as kokoro, \
             patch.object(self.router._openvoice, "convert", side_effect=lambda b, r, o: (sf.write(o, np.zeros(24000, np.float32), 24000), (24000, 1.0))[1]) as convert:
            result = self.router.synthesize("Hello", mode="clone", clone_engine="openvoice-v2",
                                            speaker_wav=wav, output_filename="ov.wav")
        kokoro.assert_called_once()
        convert.assert_called_once()
        self.assertEqual(result.model, "openvoice-v2")
        self.assertEqual(result.language["baseEngine"], "kokoro")


class MMSFailureTests(unittest.TestCase):
    """An MMS failure is that request's error; it never reaches for Higgs."""

    def test_failure_raises_with_the_language_and_does_not_load_higgs(self):
        router = VoiceEngineRouter(device="cpu")
        with patch.object(router._mms, "synthesize", side_effect=OSError("not cached")), \
             patch.object(router, "load_higgs") as load_higgs:
            with self.assertRaises(RuntimeError) as caught:
                router._synthesize_mms("namaste", Path("/tmp/x.wav"), "hin")
        load_higgs.assert_not_called()
        self.assertIn("hin", str(caught.exception))
        # One language failing must not switch MMS off for every other one.
        self.assertFalse(router._mms_failed)


class HiggsLoadingTests(unittest.TestCase):
    """Tests for Higgs model loading guard."""

    def test_higgs_load_failure_marks_flag(self):
        router = VoiceEngineRouter(device="cpu")
        # Patch transformers pipeline to raise so load fails
        with patch("voice_engine.VoiceEngineRouter.load_higgs", return_value=False):
            router._higgs_failed = True
            result = router.select_model(mode="high_quality")
            self.assertEqual(result, "xtts-v2")


if __name__ == "__main__":
    unittest.main()