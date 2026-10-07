"""
Render worker (G2-05 / G2-06): AvatarRenderJob -> MP4.

The face landmarker is replaced by the synthetic-face fixture; ffmpeg is the
real system binary. No model weights are touched.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import provenance
import render_engine
import video_io
from avatar_store import AvatarConsentError, AvatarNotFound, AvatarStore
from contracts import AvatarRenderJob, RenderQuality
from render_engine import RenderError
from vision_fixtures import FakeFaceEngine, gradient_image

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


class RenderCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.outputs = self.tmp / "outputs"
        self.inputs = self.tmp / "inputs"
        self.outputs.mkdir()
        self.inputs.mkdir()
        for name, value in (
            ("OUTPUTS_DIR", self.outputs),
            ("INPUTS_DIR", self.inputs),
            ("RENDERS_DIR", self.outputs / "renders"),
        ):
            patcher = mock.patch.object(render_engine, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.engine = FakeFaceEngine()
        patcher = mock.patch("render_engine.shared_face_engine", return_value=self.engine)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.store = AvatarStore(root=self.inputs / "faces", engine=self.engine)
        self.store.register(gradient_image(512), avatar_id="demo", source=provenance.SYNTHETIC)

    def wav(self, seconds=1.0, name="speech.wav") -> Path:
        import soundfile as sf

        path = self.outputs / name
        t = np.arange(int(24000 * seconds)) / 24000.0
        sf.write(path, (0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 24000)
        return path

    def job(self, **overrides) -> AvatarRenderJob:
        payload = {
            "jobId": "JOB-1",
            "avatarId": "demo",
            "audioUrl": "http://localhost:8000/outputs/speech.wav",
            "sampleRate": 24000,
            "durationSeconds": 1.0,
            "phonemeTimestamps": [
                {"phoneme": "AA", "viseme": "viseme_aa", "startMs": 0, "endMs": 400},
                {"phoneme": "M", "viseme": "viseme_PP", "startMs": 400, "endMs": 500},
                {"phoneme": "OW", "viseme": "viseme_O", "startMs": 500, "endMs": 900},
            ],
            "emotionVector": {"happy": 0.0, "neutral": 1.0, "eyeblinkRate": 1.0},
            "renderQuality": "PREVIEW",
            "targetFps": 25,
        }
        payload.update(overrides)
        return AvatarRenderJob.model_validate(payload)


class AudioUrlTests(RenderCase):
    def test_outputs_url_maps_to_the_local_file(self):
        path = self.wav()
        self.assertEqual(render_engine.resolve_audio_url("http://localhost:8000/outputs/speech.wav"), path.resolve())
        self.assertEqual(render_engine.resolve_audio_url(path.resolve().as_uri()), path.resolve())

    def test_remote_and_foreign_urls_are_refused_with_the_fix(self):
        for url in ("s3://assets/audio/x.wav", "http://example.com/x.wav", "ftp://host/outputs/x.wav"):
            with self.assertRaises(RenderError) as ctx:
                render_engine.resolve_audio_url(url)
            self.assertIn("/outputs/", str(ctx.exception))

    def test_cannot_escape_the_media_folders(self):
        secret = self.tmp / "secret.wav"
        secret.write_bytes(b"x")
        for url in ("http://localhost/outputs/../secret.wav", secret.as_uri(), "file:///etc/passwd"):
            with self.assertRaises(RenderError):
                render_engine.resolve_audio_url(url)

    def test_nul_byte_is_a_render_error_not_a_crash(self):
        for url in ("http://h/outputs/a%00b.wav", "file:///home/x%00.wav"):
            with self.assertRaises(RenderError):
                render_engine.resolve_audio_url(url)

    def test_missing_file_says_to_synthesize_first(self):
        with self.assertRaises(RenderError) as ctx:
            render_engine.resolve_audio_url("http://localhost/outputs/nope.wav")
        self.assertIn("synthesize", str(ctx.exception).lower())


class PlanningTests(RenderCase):
    def test_output_name_is_a_safe_filename(self):
        self.assertEqual(render_engine.output_path_for("AVT-9821-X").name, "AVT-9821-X.mp4")
        tricky = render_engine.output_path_for("../../etc/passwd")
        self.assertEqual(tricky.parent, self.outputs / "renders")
        self.assertNotIn("..", tricky.name)
        self.assertTrue(render_engine.output_path_for("///").name.startswith("job-"))

    def test_different_job_ids_never_share_a_file(self):
        ids = ["a b", "a_b", "a/b", "_a_b_", "x" * 80 + "1", "x" * 80 + "2", "///", "//"]
        names = {render_engine.output_path_for(job_id).name for job_id in ids}
        self.assertEqual(len(names), len(ids))
        for job_id in ids:
            self.assertEqual(render_engine.output_path_for(job_id).parent, self.outputs / "renders")

    def test_quality_caps_size_without_upscaling(self):
        self.assertEqual(render_engine.output_size(2048, 2048, RenderQuality.PREVIEW), (512, 512))
        self.assertEqual(render_engine.output_size(2048, 2048, RenderQuality.HD_1080P), (1080, 1080))
        self.assertEqual(render_engine.output_size(400, 300, RenderQuality.HD_1080P), (400, 300))

    def test_engine_names(self):
        self.assertEqual(render_engine.validate_engine(None), "blendshape")
        self.assertEqual(render_engine.validate_engine(" Wav2Lip "), "wav2lip")
        with self.assertRaises(RenderError):
            render_engine.validate_engine("sadtalker")
        with mock.patch.dict("os.environ", {"RENDER_ENGINE": "wav2lip"}):
            self.assertEqual(render_engine.validate_engine(None), "wav2lip")


class PreflightTests(RenderCase):
    def test_passes_for_a_renderable_job_without_loading_a_model(self):
        path = self.wav()
        self.assertEqual(render_engine.preflight(self.job(), store=self.store), path.resolve())
        self.assertEqual(self.engine.calls, 1)  # only the registration in setUp

    def test_unknown_avatar(self):
        self.wav()
        with self.assertRaises(AvatarNotFound):
            render_engine.preflight(self.job(avatarId="ghost"), store=self.store)

    def test_face_without_consent_is_never_rendered(self):
        self.wav()
        provenance.sidecar_path(self.store.get("demo").path).unlink()
        with self.assertRaises(AvatarConsentError):
            render_engine.preflight(self.job(), store=self.store)
        with self.assertRaises(AvatarConsentError):
            render_engine.render_job(self.job(), store=self.store)

    def test_wav2lip_without_weights_fails_loudly_instead_of_falling_back(self):
        self.wav()
        fake = mock.Mock(available=False)
        with mock.patch("wav2lip_engine.shared_wav2lip_engine", return_value=fake):
            with self.assertRaises(RenderError) as ctx:
                render_engine.preflight(self.job(), engine="wav2lip", store=self.store)
        self.assertIn("--accept-licence wav2lip", str(ctx.exception))


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg / ffprobe not installed")
class RenderJobTests(RenderCase):
    def test_renders_an_mp4_with_both_streams_and_the_right_frame_count(self):
        self.wav(1.0)
        ticks = []
        result = render_engine.render_job(
            self.job(), store=self.store, progress=lambda done, total: ticks.append((done, total))
        )
        self.assertEqual(result.engine, "blendshape")
        self.assertEqual(result.frame_count, 25)
        self.assertEqual((result.width, result.height), (512, 512))
        self.assertEqual(result.video_url, "/outputs/renders/JOB-1.mp4")
        self.assertEqual(ticks[-1], (25, 25))
        self.assertTrue(result.energy_gated)
        self.assertIsNone(result.peak_vram_mb)
        self.assertEqual(result.warnings, [])

        info = video_io.probe(result.output_path)
        self.assertTrue(info.has_video and info.has_audio)
        self.assertEqual(info.frame_count, 25)
        self.assertLess(info.duration_gap, 0.08)

        payload = result.to_dict()
        self.assertEqual(payload["media"]["videoCodec"], "h264")
        self.assertGreater(payload["realtimeFactor"], 0)

    def test_mouth_moves_between_frames(self):
        self.wav(1.0)
        result = render_engine.render_job(self.job(), store=self.store, label=False)
        frames = list(video_io.read_frames(result.output_path))
        differences = [np.abs(frames[i].astype(int) - frames[0].astype(int)).mean() for i in range(1, len(frames))]
        self.assertGreater(max(differences), 0.05)

    def test_label_is_burned_in_by_default(self):
        self.wav(1.0)
        plain = render_engine.render_job(self.job(jobId="plain"), store=self.store, label=False)
        labelled = render_engine.render_job(self.job(jobId="labelled"), store=self.store)
        a = next(video_io.read_frames(plain.output_path)).astype(int)
        b = next(video_io.read_frames(labelled.output_path)).astype(int)
        corner = np.abs(a - b)[-60:, :200].mean()
        elsewhere = np.abs(a - b)[:200, 300:].mean()
        self.assertGreater(corner, elsewhere * 5)

    def test_video_follows_the_audio_when_the_contract_duration_is_off(self):
        self.wav(2.0)
        result = render_engine.render_job(self.job(durationSeconds=1.0), store=self.store)
        self.assertEqual(result.frame_count, 50)
        self.assertIn("follows the audio", result.warnings[0])

    def test_unknown_visemes_are_reported_in_the_result(self):
        self.wav(1.0)
        job = self.job(phonemeTimestamps=[{"phoneme": "B", "viseme": "viseme_B", "startMs": 0, "endMs": 500}])
        with self.assertLogs("face_animation", level="WARNING"):
            result = render_engine.render_job(job, store=self.store)
        self.assertEqual(result.unknown_visemes, {"viseme_B": 1})
        self.assertTrue(any("viseme_B" in w for w in result.warnings))

    def test_quality_tier_caps_resolution(self):
        self.wav(0.4)
        big = AvatarStore(root=self.inputs / "big", engine=self.engine)
        big.register(gradient_image(1024), avatar_id="demo", source=provenance.SYNTHETIC)
        job = self.job(durationSeconds=0.4, phonemeTimestamps=[{"phoneme": "AA", "viseme": "viseme_aa", "startMs": 0, "endMs": 300}])
        preview = render_engine.render_job(job, store=big)
        self.assertEqual((preview.width, preview.height), (512, 512))
        full = render_engine.render_job(self.job(jobId="hq", renderQuality="1080P_HQ", durationSeconds=0.4,
                                                 phonemeTimestamps=job.model_dump(by_alias=True)["phonemeTimestamps"]), store=big)
        self.assertEqual((full.width, full.height), (1024, 1024))

    def test_wav2lip_engine_gets_mouthless_frames_and_reports_itself(self):
        self.wav(1.0)
        seen = {}

        class FakeWav2Lip:
            available = True

            def sync_frames(self, frames, face_box, audio_path, fps, frame_count):
                seen.update(face_box=face_box, fps=fps, frame_count=frame_count)
                for frame in frames:
                    yield frame

        with mock.patch("wav2lip_engine.shared_wav2lip_engine", return_value=FakeWav2Lip()):
            with mock.patch("render_engine._warp_frames", wraps=render_engine._warp_frames) as warp:
                result = render_engine.render_job(self.job(), engine="wav2lip", store=self.store)
        self.assertEqual(result.engine, "wav2lip")
        self.assertEqual(seen["frame_count"], 25)
        self.assertEqual(len(seen["face_box"]), 4)
        self.assertFalse(warp.call_args.kwargs["include_mouth"])

    def test_failed_render_leaves_no_partial_file(self):
        self.wav(1.0)
        with mock.patch("render_engine.PortraitAnimator.render", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                render_engine.render_job(self.job(), store=self.store)
        self.assertEqual(list((self.outputs / "renders").glob("*")), [])

    def test_failed_rerender_keeps_the_earlier_finished_video(self):
        self.wav(1.0)
        first = render_engine.render_job(self.job(), store=self.store)
        before = Path(first.output_path).read_bytes()
        with mock.patch("render_engine.PortraitAnimator.render", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                render_engine.render_job(self.job(), store=self.store)
        self.assertEqual(Path(first.output_path).read_bytes(), before)

    def test_every_rendered_frame_is_in_the_file_and_no_audio_is_cut(self):
        # 3.017 s at 30 fps used to write 91 frames and keep 90 (-shortest).
        self.wav(3.017)
        result = render_engine.render_job(self.job(targetFps=30, durationSeconds=3.017), store=self.store)
        self.assertEqual(result.frame_count, 91)
        self.assertEqual(result.media["frameCount"], 91)
        self.assertAlmostEqual(result.media["audioDuration"], 3.017, delta=0.03)
        self.assertEqual(result.warnings, [])

    def test_clip_shorter_than_a_frame_still_has_a_video_stream(self):
        self.wav(0.6)
        job = self.job(targetFps=1, durationSeconds=0.6,
                       phonemeTimestamps=[{"phoneme": "AA", "viseme": "viseme_aa", "startMs": 0, "endMs": 400}])
        result = render_engine.render_job(job, store=self.store)
        info = video_io.probe(result.output_path)
        self.assertTrue(info.has_video and info.has_audio)


if __name__ == "__main__":
    unittest.main()
