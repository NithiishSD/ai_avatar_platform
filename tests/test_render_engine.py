"""
Render worker (G2-05 / G2-06): AvatarRenderJob -> MP4.

The face landmarker is replaced by the synthetic-face fixture; ffmpeg is the
real system binary. No model weights are touched.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import manifest as manifest_module
import provenance
import render_engine
import watermark_engine
import video_io
from avatar_store import AvatarConsentError, AvatarNotFound, AvatarStore
from contracts import AvatarRenderJob, BackgroundSpec, RenderQuality
from face_engine import BackgroundSegmenter
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
            render_engine.validate_engine("liveportrait")
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

    def test_sadtalker_without_its_files_fails_loudly_instead_of_falling_back(self):
        self.wav()
        with mock.patch("sadtalker_engine.missing", return_value=["the SadTalker code is not checked out (fetch it)"]):
            with self.assertRaises(RenderError) as ctx:
                render_engine.preflight(self.job(), engine="sadtalker", store=self.store)
        self.assertIn("not checked out", str(ctx.exception))
        self.assertIn("blendshape", str(ctx.exception))


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

    def fake_sadtalker(self, frames=20, size=64, device="cuda"):
        """Replace the SadTalker child with one that writes a real 25 fps MP4 of grey frames, and the
        Real-ESRGAN sharpener with a 4x pixel repeat, so no model is loaded."""
        import sadtalker_engine

        calls = {"upscaled": 0}

        def render(image, audio_path, pose_style, still):
            calls.update(shape=image.shape, pose_style=pose_style, still=still)
            folder = self.tmp / "st" / "out"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "result.mp4"
            with video_io.VideoWriter(path, size, size, 25) as writer:
                for _ in range(frames):
                    writer.write(np.full((size, size, 3), 128, dtype=np.uint8))
            return sadtalker_engine.SadTalkerResult(path, device, 2345 if device == "cuda" else None, 1.0)

        class FakeResolver:
            available = True

            def upscale(self, frame):
                calls["upscaled"] += 1
                return np.repeat(np.repeat(frame, 4, axis=0), 4, axis=1)

        patcher = mock.patch("render_engine._frame_resolver", return_value=FakeResolver())
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls, render

    def test_sadtalker_frames_replace_the_warp_and_report_themselves(self):
        self.wav(1.0)
        # 20 frames at 25 fps for 1 s of audio: the last frame must be held to reach 25.
        calls, render = self.fake_sadtalker(frames=20)
        with mock.patch("sadtalker_engine.missing", return_value=[]), \
             mock.patch("sadtalker_engine.render", side_effect=render), \
             mock.patch("render_engine._warp_frames") as warp:
            result = render_engine.render_job(self.job(), engine="sadtalker", store=self.store)
        warp.assert_not_called()
        self.assertEqual(result.engine, "sadtalker")
        self.assertEqual(result.frame_count, 25)
        self.assertEqual((result.width, result.height), (512, 512))
        self.assertEqual(result.peak_vram_mb, 2345)
        # The warp's blink model did not draw this video, so it reports none.
        self.assertEqual(result.blink_count, 0)
        self.assertEqual(calls["shape"], (512, 512, 3))
        self.assertFalse(calls["still"])
        self.assertEqual(calls["upscaled"], 20)  # every SadTalker frame is sharpened
        self.assertFalse((self.tmp / "st").exists())  # the child's folder is cleaned up

    def test_sadtalker_still_mode_and_honest_warnings(self):
        self.wav(1.0)
        calls, render = self.fake_sadtalker(frames=25, device="cpu")
        job = self.job(motionIntensity=0.0, emotionVector={"happy": 0.0, "neutral": 0.5, "eyeblinkRate": 1.0, "joy": 0.8})
        with mock.patch("sadtalker_engine.missing", return_value=[]), mock.patch("sadtalker_engine.render", side_effect=render):
            result = render_engine.render_job(job, engine="sadtalker", store=self.store)
        self.assertTrue(calls["still"])
        # SadTalker's small frames are scaled to the job's 512 px frame, not passed through at the wrong size.
        self.assertEqual((result.width, result.height), (512, 512))
        text = " ".join(result.warnings)
        self.assertIn("no emotion input", text)
        self.assertIn("ran on the cpu", text)

    def test_sadtalker_without_real_esrgan_says_the_frames_were_not_sharpened(self):
        self.wav(1.0)
        _, render = self.fake_sadtalker(frames=25)
        with mock.patch("sadtalker_engine.missing", return_value=[]), mock.patch("sadtalker_engine.render", side_effect=render), \
             mock.patch("render_engine._frame_resolver", return_value=mock.Mock(available=False)):
            result = render_engine.render_job(self.job(), engine="sadtalker", store=self.store)
        self.assertIn("without sharpening", " ".join(result.warnings))

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

    def test_a_clip_whose_last_frame_starts_just_before_the_end_keeps_it(self):
        # 3.25 s at 25 fps: frame 82 starts at 3.24 s. ffmpeg used to cut it
        # (the owner's host render printed "82 frames were rendered but the file holds 81").
        self.wav(3.25)
        result = render_engine.render_job(self.job(durationSeconds=3.25), store=self.store, label=False)
        self.assertEqual(result.frame_count, 82)
        self.assertEqual(result.media["frameCount"], 82)
        self.assertEqual(result.warnings, [])
        self.assertAlmostEqual(result.media["audioDuration"], 3.25, delta=0.03)  # audio not cut

    def test_clip_shorter_than_a_frame_still_has_a_video_stream(self):
        self.wav(0.6)
        job = self.job(targetFps=1, durationSeconds=0.6,
                       phonemeTimestamps=[{"phoneme": "AA", "viseme": "viseme_aa", "startMs": 0, "endMs": 400}])
        result = render_engine.render_job(job, store=self.store)
        info = video_io.probe(result.output_path)
        self.assertTrue(info.has_video and info.has_audio)


class FakeSegmenter(BackgroundSegmenter):
    """Real compositing code, with the model replaced by a centred rectangle."""

    def __init__(self, person_fraction: float = 0.5, available: bool = True):
        super().__init__(model_path=Path("/nonexistent.tflite"))
        self.person_fraction = person_fraction
        self._available = available
        self.calls = 0

    @property
    def available(self) -> bool:
        return self._available

    def foreground_mask(self, image):
        self.calls += 1
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        mh, mw = int(h * self.person_fraction), int(w * self.person_fraction)
        top, left = (h - mh) // 2, (w - mw) // 2
        mask[top : top + mh, left : left + mw] = 1.0
        return mask


class BackgroundSpecTests(unittest.TestCase):
    def test_needs_exactly_one_source(self):
        for payload in ({}, {"color": "#112233", "imageUrl": "file:///x.png"}):
            with self.assertRaises(ValueError):
                BackgroundSpec.model_validate(payload)

    def test_colour_must_be_six_digit_hex(self):
        for bad in ("red", "#fff", "#12345g", "112233"):
            with self.assertRaises(ValueError):
                BackgroundSpec.model_validate({"color": bad})
        self.assertEqual(BackgroundSpec.model_validate({"color": "#0a0B0c"}).color, "#0a0B0c")

    def test_cover_fit_fills_without_stretching(self):
        wide = np.zeros((100, 400, 3), dtype=np.uint8)
        wide[:, 150:250] = 255  # a white band in the middle
        fitted = render_engine._cover_fit(wide, 200, 200)
        self.assertEqual(fitted.shape, (200, 200, 3))
        # Scaled by 2, the middle 100 px of the source (the band) spans the
        # whole 200 px crop; a stretch would have shown black either side.
        self.assertGreater(fitted.mean(), 250)


class BackgroundRenderTests(RenderCase):
    def setUp(self):
        super().setUp()
        self.segmenter = FakeSegmenter()
        patcher = mock.patch("render_engine.shared_segmenter", return_value=self.segmenter)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_colour_replaces_the_background_but_keeps_the_subject(self):
        photo = self.store.load_image("demo")
        out, warnings = render_engine.apply_background(photo, BackgroundSpec(color="#ff0000"))
        self.assertEqual(warnings, [])
        h, w = photo.shape[:2]
        self.assertEqual(tuple(out[2, 2]), (255, 0, 0))               # corner: new colour
        self.assertTrue(np.array_equal(out[h // 2, w // 2], photo[h // 2, w // 2]))  # centre: untouched

    def test_image_background_is_read_from_outputs_only(self):
        import cv2

        cv2.imwrite(str(self.outputs / "bg.png"), np.full((64, 128, 3), (0, 0, 255), dtype=np.uint8))  # BGR red
        spec = BackgroundSpec(imageUrl="http://localhost/outputs/bg.png")
        out, _ = render_engine.apply_background(self.store.load_image("demo"), spec)
        self.assertEqual(tuple(out[2, 2]), (255, 0, 0))
        for url in ("file:///etc/passwd", "http://localhost/other/bg.png"):
            with self.assertRaises(RenderError):
                render_engine._check_background(BackgroundSpec(imageUrl=url))

    def test_undecodable_background_image_is_an_error(self):
        (self.outputs / "bg.png").write_bytes(b"not an image")
        with self.assertRaises(RenderError) as ctx:
            render_engine.apply_background(
                self.store.load_image("demo"), BackgroundSpec(imageUrl="http://localhost/outputs/bg.png")
            )
        self.assertIn("decoded", str(ctx.exception))

    def test_missing_segmenter_refuses_instead_of_keeping_the_old_background(self):
        self.segmenter._available = False
        self.wav(1.0)
        with self.assertRaises(RenderError) as ctx:
            render_engine.preflight(self.job(background={"color": "#101820"}), store=self.store)
        self.assertIn("fetch_vision_models", str(ctx.exception))

    def test_a_missed_subject_is_reported(self):
        self.segmenter.person_fraction = 0.1  # 1% of the frame
        _, warnings = render_engine.apply_background(self.store.load_image("demo"), BackgroundSpec(color="#000000"))
        self.assertEqual(len(warnings), 1)
        self.assertIn("almost no person", warnings[0])

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg / ffprobe not installed")
    def test_rendered_video_carries_the_new_background_and_says_so(self):
        self.wav(1.0)
        result = render_engine.render_job(
            self.job(background={"color": "#00c800"}), store=self.store, label=False
        )
        self.assertEqual(result.background, "color #00c800")
        frame = next(video_io.read_frames(result.output_path)).astype(int)
        self.assertLess(np.abs(frame[4, 4] - np.array([0, 200, 0])).max(), 30)  # H.264 is lossy

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg / ffprobe not installed")
    def test_job_without_background_never_touches_the_segmenter(self):
        self.wav(1.0)
        result = render_engine.render_job(self.job(), store=self.store, label=False)
        self.assertIsNone(result.background)
        self.assertEqual(self.segmenter.calls, 0)


class FakeVideoMarker:
    """Stands in for VideoSeal: passes frames through, remembers the id, reads back what it is told to."""

    def __init__(self, detected=True):
        self.detected = detected
        self.embedded_ids = []
        self.frames_seen = 0

    def embed_stream(self, frames, manifest_id):
        self.embedded_ids.append(manifest_id)
        for frame in frames:
            self.frames_seen += 1
            yield frame

    def detect_video(self, path, sample_frames=16):
        from video_watermark import VideoWatermarkReport

        ident = self.embedded_ids[-1].hex() if self.embedded_ids else None
        return VideoWatermarkReport(self.detected, 128 if self.detected else 40, 1.0 if self.detected else 0.3,
                                    ident if self.detected else None, sample_frames)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg / ffprobe not installed")
class RenderWatermarkAndManifestTests(RenderCase):
    def setUp(self):
        super().setUp()
        self.marker = FakeVideoMarker()
        for patcher in (
            mock.patch("watermark_engine.enabled", return_value=True),
            mock.patch("watermark_engine.signing_key", return_value=b"render-test-key"),
            mock.patch("video_watermark.VideoWatermarker.files_present", return_value=True),
            mock.patch("video_watermark.shared_video_watermarker", return_value=self.marker),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        watermark_engine.reset_key_cache()
        self.addCleanup(watermark_engine.reset_key_cache)

    def render(self, **kw):
        self.wav(1.0)
        return render_engine.render_job(self.job(**kw), store=self.store, label=False)

    def load_manifest(self, result):
        return json.loads(Path(result.manifest["path"]).read_text())

    def test_a_render_marks_the_frames_and_ships_a_manifest_that_verifies_against_the_file(self):
        result = self.render()
        self.assertEqual(self.marker.frames_seen, result.frame_count)           # every frame went through the marker
        self.assertTrue(result.watermark["applied"] and result.watermark["verified"])
        document = self.load_manifest(result)
        report = manifest_module.verify(document, result.output_path)
        self.assertTrue(report.trustworthy, report.problems)
        # one id ties the three together: what was embedded, what was read back, what the manifest says
        embedded = self.marker.embedded_ids[-1].hex()
        self.assertEqual({result.watermark["embeddedManifestId"], result.watermark["manifestId"],
                          result.manifest["manifestId"], document["manifestId"]}, {embedded})
        self.assertTrue(result.manifest["url"].endswith(".mp4.manifest.json"))

    def test_the_manifest_records_what_went_into_the_video(self):
        result = self.render()
        document = self.load_manifest(result)
        self.assertEqual(document["inputs"]["avatar"]["avatarId"], "demo")
        self.assertEqual(document["inputs"]["avatar"]["source"], "synthetic")
        self.assertEqual(document["processing"]["renderEngine"], "blendshape")
        self.assertEqual(document["processing"]["quality"], "PREVIEW")
        self.assertEqual(document["content"]["frameCount"], result.frame_count)
        self.assertEqual(document["inputs"]["audio"]["speechRecord"]["status"], "unrecorded")  # this wav came from no synthesis

    def test_a_speech_record_for_the_exact_audio_is_carried_into_the_manifest(self):
        path = self.wav(1.0)
        manifest_module.write_speech_record(path, model="kokoro", mode="fast", language="en", speaker_wav=None, clone_engine=None,
                                            emotion=None, alignment_method="mms_fa", duration_seconds=1.0, watermark={"applied": True})
        result = render_engine.render_job(self.job(), store=self.store, label=False)
        record = self.load_manifest(result)["inputs"]["audio"]["speechRecord"]
        self.assertEqual((record["status"], record["model"], record["alignmentMethod"]), ("matched", "kokoro", "mms_fa"))

    def test_a_mark_that_cannot_be_read_back_fails_the_job_and_leaves_no_video(self):
        self.marker.detected = False
        self.wav(1.0)
        with self.assertRaisesRegex(RenderError, "could not be read back"):
            render_engine.render_job(self.job(), store=self.store, label=False)
        self.assertEqual(list((self.outputs / "renders").glob("*")), [])        # no video, no partial, no manifest

    def test_switching_the_mark_off_is_visible_and_the_manifest_is_still_issued(self):
        with mock.patch("watermark_engine.enabled", return_value=False):
            result = self.render()
        self.assertEqual(result.watermark["applied"], False)
        self.assertIn("WATERMARK_ENABLED", result.watermark["reason"])
        self.assertEqual(self.marker.frames_seen, 0)
        document = self.load_manifest(result)
        self.assertTrue(manifest_module.verify(document, result.output_path).trustworthy)
        self.assertEqual(document["watermarks"]["video"]["applied"], False)    # the manifest does not claim a mark it lacks
        self.assertIsNone(document["models"]["videoWatermark"])

    def test_a_render_writes_the_face_use_and_the_manifest_to_the_audit_trail(self):
        from audit_log import AuditLog

        log = AuditLog(":memory:")
        with mock.patch("audit_log.shared_audit", return_value=log):
            result = self.render()
        (issued,) = log.query(event="manifest_issued")
        (used,) = log.query(event="face_use")
        self.assertEqual(issued["subject"], result.manifest["manifestId"])
        self.assertEqual(issued["details"]["video_sha256"], result.manifest["videoSha256"])
        self.assertEqual((used["subject"], used["basis"], used["details"]["manifest"]), ("demo", "synthetic", result.manifest["manifestId"]))
        self.assertTrue(log.verify_chain()["valid"])

    def test_two_renders_never_share_a_manifest_id(self):
        first = self.render(jobId="A")
        second = self.render(jobId="B")
        self.assertNotEqual(first.manifest["manifestId"], second.manifest["manifestId"])

    def test_preflight_refuses_to_queue_a_job_that_cannot_be_marked(self):
        self.wav(1.0)
        with mock.patch("video_watermark.VideoWatermarker.files_present", return_value=False):
            with self.assertRaisesRegex(RenderError, "fetch_vision_models.py --only videoseal"):
                render_engine.preflight(self.job(), store=self.store)
            with mock.patch("watermark_engine.enabled", return_value=False):
                render_engine.preflight(self.job(), store=self.store)           # the opt-out needs no model


if __name__ == "__main__":
    unittest.main()
