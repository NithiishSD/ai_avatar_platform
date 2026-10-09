"""
SadTalker engine (I-01, option 2): what it needs, how it is launched, how failures read.

The child process is never started: ``subprocess.run`` is replaced, so no model is loaded and
nothing is downloaded. The real run is recorded in docs/12-PROGRESS.md.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import sadtalker_engine
from sadtalker_engine import SadTalkerError

# Two tiny stand-in weight files with known sizes, so "complete" can be built on disk in a temp folder.
FAKE_FILES = {"checkpoints/a.safetensors": ("https://x/a", "0" * 64, 5), "gfpgan/weights/b.pth": ("https://x/b", "1" * 64, 3)}


class EngineCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        for name, value in (("SADTALKER_DIR", self.root), ("SADTALKER_FILES", FAKE_FILES)):
            patcher = mock.patch.object(sadtalker_engine, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def install(self, sizes=None):
        """Lay out a complete checkout (inference.py + every file at its size)."""
        (self.root / "inference.py").write_text("")
        for relative, (_url, _sha, size) in FAKE_FILES.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * (sizes or {}).get(relative, size))


class MissingTests(EngineCase):
    def test_empty_folder_names_the_code_every_file_and_the_fetch_command(self):
        problems = sadtalker_engine.missing()
        self.assertTrue(any("not checked out" in p for p in problems))
        for relative in FAKE_FILES:
            self.assertTrue(any(relative in p and "missing" in p for p in problems))
        self.assertTrue(all(sadtalker_engine.FETCH_COMMAND in p or "pip install" in p for p in problems))
        self.assertFalse(sadtalker_engine.available())

    def test_a_truncated_file_does_not_count(self):
        self.install(sizes={"checkpoints/a.safetensors": 4})
        with mock.patch("importlib.util.find_spec", return_value=object()):
            problems = sadtalker_engine.missing()
        self.assertEqual(len(problems), 1)
        self.assertIn("4 bytes, expected 5", problems[0])

    def test_missing_package_names_the_pip_command(self):
        self.install()
        with mock.patch("importlib.util.find_spec", side_effect=lambda m: None if m == "kornia" else object()):
            problems = sadtalker_engine.missing()
        self.assertEqual(len(problems), 1)
        self.assertIn("kornia", problems[0])
        self.assertIn("--no-deps facexlib", problems[0])

    def test_complete_install_is_available(self):
        self.install()
        with mock.patch("importlib.util.find_spec", return_value=object()):
            self.assertEqual(sadtalker_engine.missing(), [])


class PoseStyleTests(unittest.TestCase):
    def test_stable_and_within_the_trained_styles(self):
        self.assertEqual(sadtalker_engine.pose_style_for(12345), sadtalker_engine.pose_style_for(12345))
        self.assertTrue(all(0 <= sadtalker_engine.pose_style_for(s) < 46 for s in range(500)))


class RenderTests(EngineCase):
    def setUp(self):
        super().setUp()
        self.install()
        patcher = mock.patch("importlib.util.find_spec", return_value=object())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.image = np.zeros((64, 64, 3), dtype=np.uint8)
        self.audio = self.root / "speech.wav"
        self.audio.write_bytes(b"")

    def fake_child(self, returncode=0, stdout="SADTALKER_DEVICE=cuda\nSADTALKER_PEAK_VRAM_MB=2345\n", stderr="", write_video=True):
        """A stand-in for subprocess.run that records the command and writes the child's output video."""
        seen = {}

        def run(command, cwd, env, capture_output, text, timeout):
            seen.update(command=command, cwd=cwd)
            if write_video:
                out = Path(command[command.index("--result_dir") + 1])
                out.mkdir(parents=True, exist_ok=True)
                (out / "2026_10_09.mp4").write_bytes(b"video")
            return subprocess.CompletedProcess(command, returncode, stdout, stderr)

        return run, seen

    def test_launches_the_child_in_its_folder_and_reads_its_report(self):
        run, seen = self.fake_child()
        with mock.patch("subprocess.run", side_effect=run), mock.patch("gpu_utils.cuda_available", return_value=True), \
             mock.patch("gpu_utils.ensure_vram") as ensure:
            result = sadtalker_engine.render(self.image, self.audio, pose_style=7, still=False)
        self.addCleanup(sadtalker_engine.cleanup, result)
        command = seen["command"]
        self.assertEqual(seen["cwd"], self.root)
        self.assertEqual(command[2], "--child")
        self.assertEqual(command[command.index("--preprocess") + 1], "full")
        self.assertEqual(command[command.index("--pose_style") + 1], "7")
        self.assertNotIn("--still", command)
        self.assertNotIn("--cpu", command)
        ensure.assert_called_once_with(sadtalker_engine.REQUIRED_VRAM_MB, "SadTalker")
        self.assertEqual((result.device, result.peak_vram_mb), ("cuda", 2345))
        self.assertTrue(result.video_path.is_file())
        sadtalker_engine.cleanup(result)
        self.assertFalse(result.video_path.exists())

    def test_still_and_cpu_flags(self):
        run, seen = self.fake_child(stdout="SADTALKER_DEVICE=cpu\n")
        with mock.patch("subprocess.run", side_effect=run), mock.patch("gpu_utils.cuda_available", return_value=False):
            result = sadtalker_engine.render(self.image, self.audio, pose_style=0, still=True)
        self.addCleanup(sadtalker_engine.cleanup, result)
        self.assertIn("--still", seen["command"])
        self.assertIn("--cpu", seen["command"])
        self.assertIsNone(result.peak_vram_mb)

    def test_child_failure_raises_with_its_last_error_line(self):
        run, _ = self.fake_child(returncode=1, stderr="Traceback\nRuntimeError: CUDA out of memory", write_video=False)
        with mock.patch("subprocess.run", side_effect=run), mock.patch("gpu_utils.cuda_available", return_value=False):
            with self.assertRaises(SadTalkerError) as ctx:
                sadtalker_engine.render(self.image, self.audio, pose_style=0, still=False)
        self.assertIn("CUDA out of memory", str(ctx.exception))

    def test_no_face_found_says_so(self):
        # inference.py returns normally (exit 0) without a video when it cannot fit the face.
        run, _ = self.fake_child(stdout="Can't get the coeffs of the input\n", write_video=False)
        with mock.patch("subprocess.run", side_effect=run), mock.patch("gpu_utils.cuda_available", return_value=False):
            with self.assertRaises(SadTalkerError) as ctx:
                sadtalker_engine.render(self.image, self.audio, pose_style=0, still=False)
        self.assertIn("no face", str(ctx.exception))

    def test_refuses_before_launching_when_something_is_missing(self):
        (self.root / "inference.py").unlink()
        with mock.patch("subprocess.run") as run:
            with self.assertRaises(SadTalkerError) as ctx:
                sadtalker_engine.render(self.image, self.audio, pose_style=0, still=False)
        run.assert_not_called()
        self.assertIn(sadtalker_engine.FETCH_COMMAND, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
