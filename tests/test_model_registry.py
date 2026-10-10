"""
Tests for the model weight audit.

These pin the exact failure the audit exists to catch: a HuggingFace snapshot
holding ``config.json`` and nothing else. Higgs TTS 2 and Dia-1.6B sat in that
state for three phases while the router's graceful fallbacks reported success,
the test suite mocked past the load, and the Phase 3 benchmark measured Kokoro
in their place.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_registry import (
    MIN_CHECKPOINT_BYTES,
    ModelWeightStatus,
    audit_summary,
    check_coqui_model,
    check_hf_repo,
    check_mms,
    log_weight_audit,
)


def write(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)


class HFRepoAuditTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch("model_registry._hf_cache_root", return_value=self.cache)
        patcher.start()
        self.addCleanup(patcher.stop)

    def snapshot(self, repo_id: str) -> Path:
        return (
            self.cache
            / f"models--{repo_id.replace('/', '--')}"
            / "snapshots"
            / "deadbeef"
        )

    def test_uncached_repo_is_absent(self):
        status = check_hf_repo("higgs-tts-2", "Higgs", "bosonai/higgs-tts-2-3b-base")
        self.assertFalse(status.present)
        self.assertEqual(status.size_bytes, 0)
        self.assertIn("never downloaded", status.detail)

    def test_config_only_snapshot_is_absent(self):
        """The real Higgs/Dia failure mode: metadata cached, no weights."""
        write(self.snapshot("nari-labs/Dia-1.6B") / "config.json", 900)
        status = check_hf_repo("dia-1.6b", "Dia", "nari-labs/Dia-1.6B")
        self.assertFalse(status.present)
        self.assertIn("metadata only", status.detail)

    def test_snapshot_with_a_checkpoint_is_present(self):
        write(
            self.snapshot("hexgrad/Kokoro-82M") / "kokoro-v1_0.pth",
            MIN_CHECKPOINT_BYTES * 3,
        )
        status = check_hf_repo("kokoro", "Kokoro", "hexgrad/Kokoro-82M")
        self.assertTrue(status.present)
        self.assertEqual(status.size_bytes, MIN_CHECKPOINT_BYTES * 3)

    def test_small_weight_files_do_not_count_as_a_checkpoint(self):
        """A tokenizer vocabulary is a .bin too, so size is what separates them."""
        write(self.snapshot("acme/tiny") / "vocab.bin", 4096)
        status = check_hf_repo("tiny", "Tiny", "acme/tiny")
        self.assertFalse(status.present)

    def test_symlinked_blobs_are_resolved(self):
        """
        The hub cache stores weights in blobs/ and symlinks them into
        snapshots/, so measuring the link itself would report a few bytes.
        """
        repo = self.cache / "models--acme--linked"
        blob = repo / "blobs" / "abc123"
        write(blob, MIN_CHECKPOINT_BYTES * 2)
        snap = repo / "snapshots" / "deadbeef"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "model.safetensors").symlink_to(blob)

        status = check_hf_repo("linked", "Linked", "acme/linked")
        self.assertTrue(status.present)
        self.assertEqual(status.size_bytes, MIN_CHECKPOINT_BYTES * 2)

    def test_mms_reports_no_language_when_nothing_is_cached(self):
        status = check_mms()
        self.assertFalse(status.present)

    def test_mms_counts_only_languages_with_weights(self):
        write(
            self.cache
            / "models--facebook--mms-tts-hin"
            / "snapshots"
            / "a"
            / "model.safetensors",
            MIN_CHECKPOINT_BYTES * 2,
        )
        # Cached config with no weights must not inflate the count.
        write(
            self.cache
            / "models--facebook--mms-tts-tam"
            / "snapshots"
            / "b"
            / "config.json",
            500,
        )
        status = check_mms()
        self.assertTrue(status.present)
        self.assertIn("1 language(s)", status.detail)
        self.assertIn("hin", status.detail)
        self.assertNotIn("tam", status.detail)


class CoquiAuditTests(unittest.TestCase):
    def test_absent_model_names_every_path_it_searched(self):
        """
        The VS Code snap rewrites XDG_DATA_HOME, so Coqui's download directory
        differs between an IDE terminal and a plain shell. The detail has to
        say where it looked or the miss is unactionable.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "model_registry._coqui_roots", return_value=[Path(tmp)]
            ):
                status = check_coqui_model(
                    "xtts-v2", "XTTS-v2", "tts_models/multilingual/multi-dataset/xtts_v2"
                )
        self.assertFalse(status.present)
        self.assertIn(tmp, status.detail)

    XTTS = "tts_models/multilingual/multi-dataset/xtts_v2"

    def coqui_folder(self, tmp, *, config=True, checkpoint="complete"):
        """A Coqui download folder: complete, or broken the way real ones break."""
        import zipfile

        folder = Path(tmp) / "tts_models--multilingual--multi-dataset--xtts_v2"
        folder.mkdir(parents=True)
        if config:
            (folder / "config.json").write_text("{}")
        path = folder / "model.pth"
        with zipfile.ZipFile(path, "w") as archive:  # what torch.save writes
            archive.writestr("archive/data.pkl", b"x" * MIN_CHECKPOINT_BYTES * 2)
        if checkpoint == "truncated":
            data = path.read_bytes()
            path.write_bytes(data[: len(data) * 7 // 10])  # stopped at 70%
        return folder

    def audit(self, tmp):
        with mock.patch("model_registry._coqui_roots", return_value=[Path(tmp)]):
            return check_coqui_model("xtts-v2", "XTTS-v2", self.XTTS)

    def test_present_model_is_found_under_a_searched_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.coqui_folder(tmp)
            status = self.audit(tmp)
        self.assertTrue(status.present)

    def test_a_checkpoint_without_its_config_is_an_unfinished_download(self):
        # This fixture used to *be* the "present" case: model.pth alone.
        with tempfile.TemporaryDirectory() as tmp:
            self.coqui_folder(tmp, config=False)
            status = self.audit(tmp)
        self.assertFalse(status.present)
        self.assertIn("config.json is missing", status.detail)

    def test_a_truncated_checkpoint_is_not_counted_as_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.coqui_folder(tmp, checkpoint="truncated")
            status = self.audit(tmp)
        self.assertFalse(status.present)
        self.assertIn("model.pth is truncated", status.detail)


class SizeLabelTests(unittest.TestCase):
    def make(self, size: int) -> ModelWeightStatus:
        return ModelWeightStatus(
            key="k", name="n", source="s", present=True, size_bytes=size, detail=""
        )

    def test_labels_scale_with_magnitude(self):
        self.assertEqual(self.make(0).size_label, "0 B")
        self.assertEqual(self.make(327_000_000).size_label, "327 MB")
        self.assertEqual(self.make(6_500_000_000).size_label, "6.5 GB")


class AuditReportingTests(unittest.TestCase):
    def test_summary_shape_matches_the_health_contract(self):
        summary = audit_summary()
        self.assertEqual(
            set(summary), {"available", "total", "missing", "models"}
        )
        self.assertEqual(summary["total"], len(summary["models"]))
        self.assertEqual(
            summary["available"], sum(1 for m in summary["models"] if m["present"])
        )

    def test_every_missing_model_is_warned_about_individually(self):
        statuses = [
            ModelWeightStatus("a", "A", "a", True, MIN_CHECKPOINT_BYTES, "ok"),
            ModelWeightStatus("b", "B", "b", False, 0, "gone"),
            ModelWeightStatus("c", "C", "c", False, 0, "gone"),
        ]
        with self.assertLogs("model_registry", level="WARNING") as captured:
            log_weight_audit(statuses)
        warnings = [line for line in captured.output if "WARNING" in line]
        self.assertEqual(len(warnings), 2)
        # Worded "unavailable", not "weights missing": for an engine this
        # stack cannot run, the weights are not the problem.
        self.assertTrue(all("MODEL UNAVAILABLE" in w for w in warnings))
        self.assertTrue(all("Fix:" in w for w in warnings))

    def test_a_full_house_logs_no_warning(self):
        statuses = [
            ModelWeightStatus("a", "A", "a", True, MIN_CHECKPOINT_BYTES, "ok"),
        ]
        with self.assertLogs("model_registry", level="INFO") as captured:
            log_weight_audit(statuses)
        self.assertFalse(any("WARNING" in line for line in captured.output))


if __name__ == "__main__":
    unittest.main()


class UnrunnableEngineTests(unittest.TestCase):
    """Higgs and Dia cannot load on this stack; nothing may suggest fetching them."""

    def test_audit_reports_them_unavailable_with_no_fetch_advice(self):
        from model_registry import audit_model_weights

        statuses = {s.key: s for s in audit_model_weights()}
        for key in ("higgs-tts-2", "dia-1.6b"):
            self.assertFalse(statuses[key].present)
            self.assertIn("Downloading it would not help", statuses[key].detail)
            self.assertNotIn("fetch_models.py", statuses[key].fix)

    def test_a_fetchable_model_still_gets_the_fetch_command(self):
        from model_registry import audit_model_weights

        with mock.patch("model_registry.check_mms", return_value=ModelWeightStatus(
            "mms-tts", "MMS", "x", False, 0, "nothing cached"
        )):
            statuses = {s.key: s for s in audit_model_weights()}
        self.assertEqual(statuses["mms-tts"].fix, "python scripts/fetch_models.py --only mms-tts")

    def test_the_refusal_message_does_not_say_fetch(self):
        from voice_engine import ModelWeightsMissing
        from model_registry import UNRUNNABLE

        message = str(ModelWeightsMissing("dia-1.6b", UNRUNNABLE["dia-1.6b"], "none on this stack"))
        self.assertNotIn("Fetch it with", message)
        self.assertIn("Fix: none on this stack", message)

    def test_the_fetcher_refuses_instead_of_downloading(self):
        import importlib.util
        import io
        from contextlib import redirect_stderr

        spec = importlib.util.spec_from_file_location(
            "fetch_models", Path(__file__).resolve().parents[1] / "scripts" / "fetch_models.py"
        )
        fetcher = importlib.util.module_from_spec(spec)
        # @dataclass resolves annotations through sys.modules[__module__], so a
        # module loaded from a file path has to be registered while it runs.
        with mock.patch.dict(sys.modules, {"fetch_models": fetcher}):
            spec.loader.exec_module(fetcher)
        err = io.StringIO()
        with mock.patch("sys.argv", ["fetch_models.py", "--only", "higgs-tts-2"]), \
             mock.patch.object(fetcher, "_fetch_hf") as download, redirect_stderr(err):
            self.assertEqual(fetcher.main(), 1)
        download.assert_not_called()
        self.assertIn("Downloading it would not help", err.getvalue())


class OpenVoiceAuditTests(unittest.TestCase):
    def test_a_missing_package_is_reported_with_the_install_command(self):
        from model_registry import check_openvoice

        with mock.patch("importlib.util.find_spec", return_value=None):
            status = check_openvoice()
        self.assertFalse(status.present)
        self.assertIn("pip install --no-deps", status.fix)

    def test_present_only_with_both_converter_files(self):
        from model_registry import check_openvoice

        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snapshots" / "abc" / "converter"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            with mock.patch("model_registry._hf_repo_dir", return_value=Path(tmp)):
                self.assertFalse(check_openvoice().present)  # checkpoint missing
                (snapshot / "checkpoint.pth").write_bytes(b"x" * 10)
                self.assertTrue(check_openvoice().present)
