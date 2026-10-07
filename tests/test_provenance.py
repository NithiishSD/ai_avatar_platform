"""
Tests for voice reference provenance and the benchmark's admissibility guard.

The hole these close: ``benchmark_similarity`` read ``inputs/`` and measured
whatever it found first, so a reference concatenated from our own Kokoro output
would have been scored against the >85% cloning threshold and written into
docs/benchmarks/ as acceptance evidence. Synthetic speech is an artificially
easy cloning target, so that number would have looked good and meant nothing.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import provenance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_phase3 import build_matrix  # noqa: E402


class SidecarTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.audio = Path(self._tmp.name) / "voice.wav"
        self.audio.write_bytes(b"RIFF")

    def test_sidecar_sits_beside_the_audio_keeping_the_extension(self):
        """inputs/ is scanned by extension, so the record must not shadow a .wav."""
        path = provenance.sidecar_path(self.audio)
        self.assertEqual(path.name, "voice.wav.provenance.json")
        self.assertEqual(path.parent, self.audio.parent)

    def test_missing_record_is_not_admissible(self):
        info = provenance.describe(self.audio)
        self.assertFalse(info["hasRecord"])
        self.assertFalse(info["admissible"])
        self.assertIn("no provenance record", info["reason"])

    def test_corrupt_record_reads_as_missing_rather_than_raising(self):
        provenance.sidecar_path(self.audio).write_text("{not json")
        self.assertIsNone(provenance.load(self.audio))
        self.assertFalse(provenance.describe(self.audio)["admissible"])

    def test_roundtrip_preserves_the_consent_basis(self):
        provenance.write(
            self.audio,
            source=provenance.HUMAN,
            speaker="LJSpeech",
            licence="Public domain",
            consent_basis="open-licence",
        )
        record = provenance.load(self.audio)
        self.assertEqual(record.speaker, "LJSpeech")
        self.assertEqual(record.consent_basis, "open-licence")
        self.assertFalse(record.is_synthetic)

    def test_written_record_is_readable_json(self):
        provenance.write(self.audio, source=provenance.SYNTHETIC)
        raw = json.loads(provenance.sidecar_path(self.audio).read_text())
        self.assertEqual(raw["source"], "synthetic")
        self.assertIn("created", raw)

    def test_an_unknown_source_is_rejected_at_write_time(self):
        with self.assertRaises(ValueError):
            provenance.write(self.audio, source="probably-fine")


class AdmissibilityTests(unittest.TestCase):
    def test_synthetic_reference_is_never_admissible(self):
        record = provenance.VoiceProvenance(source=provenance.SYNTHETIC)
        admissible, reason = record.admissibility()
        self.assertFalse(admissible)
        self.assertIn("artificially easy", reason)

    def test_human_reference_without_a_consent_basis_is_not_admissible(self):
        """A recording with no recorded consent cannot back a published claim."""
        record = provenance.VoiceProvenance(source=provenance.HUMAN, speaker="someone")
        self.assertFalse(record.admissibility()[0])

    def test_human_reference_with_an_unrecognized_basis_is_not_admissible(self):
        record = provenance.VoiceProvenance(
            source=provenance.HUMAN, consent_basis="they seemed fine with it"
        )
        self.assertFalse(record.admissibility()[0])

    def test_each_recognized_consent_basis_admits_a_human_reference(self):
        for basis in provenance.CONSENT_BASES:
            with self.subTest(basis=basis):
                record = provenance.VoiceProvenance(
                    source=provenance.HUMAN, consent_basis=basis
                )
                admissible, reason = record.admissibility()
                self.assertTrue(admissible)
                self.assertIn(basis, reason)


class BenchmarkGuardTests(unittest.TestCase):
    """The acceptance matrix must not count an inadmissible similarity score."""

    def row(self, report: dict) -> dict:
        matrix = build_matrix(report)
        return next(r for r in matrix if "cloning similarity" in r["requirement"])

    def test_admissible_ecapa_score_is_counted(self):
        row = self.row(
            {
                "similarity": {
                    "status": "MEASURED",
                    "similarity": 0.91,
                    "similarityPercent": 91.0,
                    "isEcapa": True,
                    "admissible": True,
                    "method": "ecapa-tdnn",
                }
            }
        )
        self.assertEqual(row["measured"], 0.91)
        self.assertEqual(row["verdict"], "PASS")

    def test_synthetic_reference_scoring_above_the_threshold_still_fails(self):
        """The exact trap: a flattering score from a synthetic reference."""
        row = self.row(
            {
                "similarity": {
                    "status": "PIPELINE_TEST",
                    "similarity": 0.97,
                    "similarityPercent": 97.0,
                    "isEcapa": True,
                    "admissible": False,
                    "method": "ecapa-tdnn",
                    "reason": "reference is synthetic (TTS output)",
                }
            }
        )
        self.assertIsNone(row["measured"])
        self.assertNotEqual(row["verdict"], "PASS")
        self.assertIn("not counted", row["method"])
        # The number is still disclosed, just not credited.
        self.assertIn("97.0%", row["method"])

    def test_non_ecapa_score_is_not_counted_even_when_admissible(self):
        row = self.row(
            {
                "similarity": {
                    "status": "MEASURED",
                    "similarity": 0.93,
                    "similarityPercent": 93.0,
                    "isEcapa": False,
                    "admissible": True,
                    "method": "spectral proxy",
                }
            }
        )
        self.assertIsNone(row["measured"])
        self.assertNotEqual(row["verdict"], "PASS")

    def test_absent_similarity_section_reports_not_run(self):
        row = self.row({})
        self.assertIsNone(row["measured"])
        self.assertEqual(row["method"], "not run")


class SmokeReferenceTests(unittest.TestCase):
    def test_the_checked_in_smoke_reference_is_labelled_synthetic(self):
        """
        Guards the live inputs/ directory: if a smoke reference is present it
        must carry a synthetic record, or the benchmark could credit it.
        """
        smoke = PROJECT_ROOT / "inputs" / "smoke_reference_kokoro.wav"
        if not smoke.exists():
            self.skipTest("no smoke reference in inputs/")
        record = provenance.load(smoke)
        self.assertIsNotNone(record, "smoke reference has no provenance record")
        self.assertTrue(record.is_synthetic)
        self.assertFalse(record.admissibility()[0])


if __name__ == "__main__":
    unittest.main()
