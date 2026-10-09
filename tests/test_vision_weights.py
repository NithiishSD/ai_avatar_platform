"""Vision weight audit and its fetch script (G1-03). Nothing is downloaded."""

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import ClassVar
from unittest import mock

import model_registry
from model_registry import ModelWeightStatus, audit_vision_weights, check_local_file, vision_audit_summary

_SPEC = importlib.util.spec_from_file_location(
    "fetch_vision_models", Path(__file__).resolve().parents[1] / "scripts" / "fetch_vision_models.py"
)
fetch = importlib.util.module_from_spec(_SPEC)
sys.modules["fetch_vision_models"] = fetch
_SPEC.loader.exec_module(fetch)


class LocalFileAuditTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_present_file(self):
        path = self.tmp / "m.bin"
        path.write_bytes(b"\0" * 2000)
        status = check_local_file("m", "Model", path, min_bytes=1000)
        self.assertTrue(status.present)
        self.assertEqual(status.size_bytes, 2000)

    def test_missing_file_names_the_fetch_command(self):
        status = check_local_file("syncnet", "SyncNet", self.tmp / "nope")
        self.assertFalse(status.present)
        self.assertIn("fetch_vision_models.py --only syncnet", status.detail)

    def test_truncated_download_is_not_counted_as_present(self):
        path = self.tmp / "m.bin"
        path.write_bytes(b"<html>404</html>")
        status = check_local_file("m", "Model", path, min_bytes=1000)
        self.assertFalse(status.present)
        self.assertIn("truncated", status.detail)

    def test_first_present_alternative_wins(self):
        second = self.tmp / "b.pth"
        second.write_bytes(b"\0" * 5000)
        status = check_local_file("w", "W", [self.tmp / "a.pth", second], min_bytes=1000)
        self.assertTrue(status.present)
        self.assertEqual(status.source, str(second))


class VisionAuditTests(unittest.TestCase):
    def test_audit_lists_every_vision_model(self):
        keys = [s.key for s in audit_vision_weights()]
        self.assertEqual(
            keys,
            ["videoseal", "face-landmarker", "selfie-segmenter", "multiclass-segmenter", "wav2lip", "sadtalker", "syncnet", "sface", "realesrgan",
             "avatar-diffusion"],
        )

    def test_fetch_script_and_audit_cover_the_same_models(self):
        self.assertEqual({s.key for s in fetch.SPECS}, {s.key for s in audit_vision_weights()})

    def test_missing_wav2lip_explains_licence_and_the_engine_that_still_works(self):
        with mock.patch.object(model_registry, "WAV2LIP_CHECKPOINTS", (Path("/nonexistent/w.pth"),)):
            status = next(s for s in audit_vision_weights() if s.key == "wav2lip")
        self.assertFalse(status.present)
        self.assertIn("--accept-licence wav2lip", status.detail)
        self.assertIn("blendshape", status.detail)

    def test_summary_shape_for_health(self):
        statuses = [
            ModelWeightStatus("a", "A", "x", True, 10, "weights present"),
            ModelWeightStatus("b", "B", "y", False, 0, "missing"),
        ]
        with mock.patch("model_registry.audit_vision_weights", return_value=statuses):
            summary = vision_audit_summary()
        self.assertEqual((summary["available"], summary["total"], summary["missing"]), (1, 2, ["b"]))


class FetchScriptTests(unittest.TestCase):
    def test_default_selection_excludes_licence_gated_models(self):
        keys = {s.key for s in fetch.select_specs(None, include_all=False)}
        self.assertNotIn("wav2lip", keys)
        self.assertIn("face-landmarker", keys)
        self.assertIn("wav2lip", {s.key for s in fetch.select_specs(None, include_all=True)})

    def test_only_selects_named_keys_and_rejects_unknown(self):
        self.assertEqual([s.key for s in fetch.select_specs("syncnet, sface", False)], ["syncnet", "sface"])
        with self.assertRaises(ValueError):
            fetch.select_specs("syncnet,made-up", False)

    def test_licence_gate(self):
        wav2lip = fetch.select_specs("wav2lip", False)
        self.assertEqual([s.key for s in fetch.licence_blocked(wav2lip, set())], ["wav2lip"])
        self.assertEqual(fetch.licence_blocked(wav2lip, {"wav2lip"}), [])

    def _run(self, argv, present=()):
        statuses = [
            ModelWeightStatus(s.key, s.name, "x", s.key in present, 0, "") for s in fetch.SPECS
        ]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(fetch, "audit_vision_weights", return_value=statuses), \
             redirect_stdout(out), redirect_stderr(err):
            code = fetch.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_restricted_model_is_never_fetched_without_acceptance(self):
        called = mock.Mock()
        spec = next(s for s in fetch.SPECS if s.key == "wav2lip")
        with mock.patch.object(fetch, "SPECS", [spec.__class__(**{**spec.__dict__, "fetch": called})]):
            code, _, err = self._run(["--only", "wav2lip", "--yes"])
        self.assertEqual(code, 2)
        self.assertIn("--accept-licence wav2lip", err)
        called.assert_not_called()

    def test_nothing_to_fetch_when_everything_is_present(self):
        code, out, _ = self._run(["--yes"], present={s.key for s in fetch.SPECS})
        self.assertEqual(code, 0)
        self.assertIn("Nothing to fetch", out)

    def test_dry_run_downloads_nothing(self):
        with mock.patch.object(fetch, "download_file") as download:
            code, out, _ = self._run(["--dry-run", "--only", "syncnet"])
        self.assertEqual(code, 0)
        self.assertIn("syncnet", out)
        download.assert_not_called()

    def test_refuses_to_fill_the_disk(self):
        with mock.patch.object(fetch, "free_gb", return_value=5.5):
            code, _, err = self._run(["--only", "avatar-diffusion", "--yes"])
        self.assertEqual(code, 1)
        self.assertIn("Refusing to download", err)

    def test_download_is_atomic_and_size_checked(self):
        class Response(io.BytesIO):
            headers: ClassVar[dict] = {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "sub" / "model.bin"
            with mock.patch.object(fetch.urllib.request, "urlopen", return_value=Response(b"tiny")):
                with self.assertRaises(RuntimeError) as ctx:
                    fetch.download_file("http://x/model.bin", dest, min_bytes=1000)
            self.assertIn("expected at least", str(ctx.exception))
            self.assertEqual(list(dest.parent.iterdir()), [])  # no file, no .part

            with mock.patch.object(fetch.urllib.request, "urlopen", return_value=Response(b"x" * 2000)):
                with self.assertRaises(RuntimeError) as ctx:
                    fetch.download_file("http://x/model.bin", dest, sha256="0" * 64)
            self.assertIn("checksum mismatch", str(ctx.exception))
            self.assertFalse(dest.exists())

            with mock.patch.object(fetch.urllib.request, "urlopen", return_value=Response(b"x" * 2000)):
                fetch.download_file("http://x/model.bin", dest, min_bytes=1000)
            self.assertEqual(dest.stat().st_size, 2000)


if __name__ == "__main__":
    unittest.main()
