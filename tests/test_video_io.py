"""Video plumbing (G2-04). The round trips use the system ffmpeg, not a model."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import video_io
from video_io import FFmpegNotFound, VideoEncodeError, VideoWriter, even, fit_within, parse_probe

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


class GeometryTests(unittest.TestCase):
    def test_even_rounds_down(self):
        self.assertEqual([even(v) for v in (511, 512, 513.9, 1)], [510, 512, 512, 2])

    def test_fit_within_preserves_aspect_and_never_upscales(self):
        self.assertEqual(fit_within(3840, 2160, 1920, 1080), (1920, 1080))
        self.assertEqual(fit_within(1000, 2000, 1920, 1080), (540, 1080))
        self.assertEqual(fit_within(512, 512, 1920, 1080), (512, 512))


class CommandTests(unittest.TestCase):
    def test_missing_binary_names_the_fix(self):
        with mock.patch("video_io.shutil.which", return_value=None):
            with self.assertRaises(FFmpegNotFound) as ctx:
                video_io.require_binary("ffmpeg")
        self.assertIn("apt install ffmpeg", str(ctx.exception))

    def test_odd_dimensions_are_refused(self):
        with self.assertRaises(ValueError):
            VideoWriter("x.mp4", 511, 512, 25)

    def test_command_is_browser_playable_and_muxes_audio(self):
        with mock.patch("video_io.shutil.which", return_value="/usr/bin/ffmpeg"):
            silent = VideoWriter("x.mp4", 64, 64, 25).command()
            voiced = VideoWriter("x.mp4", 64, 64, 25, audio_path="a.wav").command()
        for flag in ("libx264", "yuv420p", "+faststart"):
            self.assertIn(flag, silent)
        self.assertNotIn("-shortest", silent)
        for flag in ("a.wav", "aac", "-shortest"):
            self.assertIn(flag, voiced)

    def test_missing_audio_file_fails_before_encoding(self):
        with self.assertRaises(FileNotFoundError):
            with VideoWriter("x.mp4", 64, 64, 25, audio_path="/nonexistent/a.wav"):
                pass


class ProbeParsingTests(unittest.TestCase):
    PAYLOAD = {
        "format": {"duration": "2.000"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 64, "height": 48,
             "avg_frame_rate": "25/1", "nb_frames": "50", "duration": "2.0"},
            {"codec_type": "audio", "codec_name": "aac", "duration": "1.96"},
        ],
    }

    def test_parses_both_streams(self):
        info = parse_probe("x.mp4", self.PAYLOAD)
        self.assertTrue(info.has_video and info.has_audio)
        self.assertEqual((info.width, info.height, info.frame_count), (64, 48, 50))
        self.assertAlmostEqual(info.fps, 25.0)
        self.assertAlmostEqual(info.duration_gap, 0.04)
        self.assertEqual(info.to_dict()["videoCodec"], "h264")

    def test_tolerates_missing_streams_and_bad_numbers(self):
        info = parse_probe("x.wav", {"streams": [{"codec_type": "audio", "codec_name": "pcm"}], "format": {"duration": "3"}})
        self.assertFalse(info.has_video)
        self.assertEqual(info.audio_duration, 3.0)
        self.assertEqual(info.duration_gap, 0.0)
        broken = parse_probe("x", {"streams": [{"codec_type": "video", "avg_frame_rate": "0/0", "nb_frames": "N/A"}]})
        self.assertEqual((broken.fps, broken.frame_count), (0.0, 0))


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg / ffprobe not installed")
class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _wav(self, seconds: float) -> Path:
        import soundfile as sf

        path = self.tmp / "tone.wav"
        t = np.arange(int(16000 * seconds)) / 16000.0
        sf.write(path, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000)
        return path

    def test_frames_and_audio_become_one_mp4_of_equal_length(self):
        out = self.tmp / "clip.mp4"
        with VideoWriter(out, 64, 48, 25, audio_path=self._wav(2.0)) as writer:
            for index in range(50):
                writer.write(np.full((48, 64, 3), index * 5, dtype=np.uint8))
        info = video_io.probe(out)
        self.assertTrue(info.has_video and info.has_audio)
        self.assertEqual((info.width, info.height, info.frame_count), (64, 48, 50))
        self.assertLess(info.duration_gap, 0.08)

        frames = list(video_io.read_frames(out))
        self.assertEqual(len(frames), 50)
        self.assertEqual(frames[0].shape, (48, 64, 3))
        self.assertLess(frames[0].mean(), frames[-1].mean())
        audio = video_io.read_audio(out, 16000)
        self.assertAlmostEqual(len(audio) / 16000.0, 2.0, delta=0.1)

    def test_wrong_frame_shape_is_refused_and_leaves_no_file(self):
        out = self.tmp / "bad.mp4"
        with self.assertRaises(ValueError):
            with VideoWriter(out, 64, 48, 25) as writer:
                writer.write(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertFalse(out.exists())

    def test_empty_render_is_an_error_not_an_empty_file(self):
        out = self.tmp / "empty.mp4"
        with self.assertRaises(VideoEncodeError):
            with VideoWriter(out, 64, 48, 25):
                pass
        self.assertFalse(out.exists())

    def test_probe_of_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            video_io.probe(self.tmp / "nope.mp4")


if __name__ == "__main__":
    unittest.main()
