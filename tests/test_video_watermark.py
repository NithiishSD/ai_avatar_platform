"""
Invisible video watermark (R-33): the message layout, the decision rule, streaming in windows, and
(when the weights are on disk) the real VideoSeal round trip through an H.264 encode.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import video_watermark as vw
import watermark_engine

KEY = b"video-test-key"
HAVE_FFMPEG = bool(shutil.which("ffmpeg"))


def frames(count=5, size=64, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(count)]


class FakeModel:
    """Embeds by adding 1 to every pixel; reads back whatever logits it is told to."""

    def __init__(self, logits=None):
        self.windows = []
        self.messages = []
        self.logits = logits

    def embed(self, batch, message, is_video=True, lowres_attenuation=False):
        self.windows.append(batch.shape[0])
        self.messages.append(message.clone())
        return {"imgs_w": (batch + 1.0 / 255.0).clamp(0, 1)}

    def detect(self, batch, is_video=True):
        n = batch.shape[0]
        logits = self.logits if self.logits is not None else torch.zeros(vw.MESSAGE_BITS)
        return {"preds": torch.cat([torch.ones(n, 1), logits.repeat(n, 1)], dim=1)}


def watermarker(model):
    marker = vw.VideoWatermarker()
    marker._model = model
    return marker


def logits_for(tag_errors=0, id_bytes=bytes(range(16)), id_errors=0):
    """What a detector would output for our tag with ``tag_errors`` wrong tag bits."""
    tag = vw.platform_bits(KEY)
    for i in range(tag_errors):
        tag[i] = 1 - tag[i]
    ident = vw.bits_of(id_bytes)
    for i in range(id_errors):
        ident[i] = 1 - ident[i]
    return torch.tensor([1.0 if b else -1.0 for b in tag + ident])


class PatchedKey(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("watermark_engine.signing_key", return_value=KEY)
        patcher.start()
        self.addCleanup(patcher.stop)


class BitsAndOddsTests(PatchedKey):
    def test_bits_round_trip_and_the_tag_is_128_bits_that_depend_on_the_key(self):
        self.assertEqual(vw.bytes_of(vw.bits_of(b"\x00\xffAB")), b"\x00\xffAB")
        tag = vw.platform_bits(KEY)
        self.assertEqual(len(tag), vw.TAG_BITS)
        self.assertNotEqual(tag, vw.platform_bits(b"other-key"))
        self.assertNotEqual(vw.platform_bits(KEY), watermark_engine.platform_tag(KEY) * 8)  # not the audio tag repeated

    def test_the_acceptance_threshold_is_as_unlikely_by_chance_as_claimed(self):
        self.assertLess(vw.chance_of_matching(vw.MIN_TAG_BITS), 1e-8)
        self.assertAlmostEqual(vw.chance_of_matching(64), 0.5, delta=0.05)  # half the bits is what chance gives
        self.assertEqual(vw.chance_of_matching(0), 1.0)


class EmbedStreamTests(PatchedKey):
    def test_frames_stream_through_in_windows_with_shape_dtype_and_order_kept(self):
        model = FakeModel()
        source = frames(70)
        out = list(watermarker(model).embed_stream(iter(source), bytes(range(16))))
        self.assertEqual(model.windows, [32, 32, 6])          # bounded memory: never more than a window
        self.assertEqual(len(out), 70)
        for before, after in zip(source, out, strict=True):
            self.assertEqual((before.shape, after.dtype), (after.shape, np.uint8))
            self.assertFalse(np.array_equal(before, after))   # every frame was marked
            self.assertLessEqual(int(np.abs(after.astype(int) - before.astype(int)).max()), 1)

    def test_the_message_is_the_tag_then_the_manifest_id_and_the_same_for_every_window(self):
        model = FakeModel()
        ident = bytes(range(16))
        list(watermarker(model).embed_stream(iter(frames(40)), ident))
        for message in model.messages:
            bits = [int(b) for b in message[0].tolist()]
            self.assertEqual(bits[:vw.TAG_BITS], vw.platform_bits(KEY))
            self.assertEqual(vw.bytes_of(bits[vw.TAG_BITS:]), ident)
            self.assertEqual(len(bits), vw.MESSAGE_BITS)

    def test_no_frames_means_no_work_and_the_model_is_never_loaded(self):
        marker = vw.VideoWatermarker()
        self.assertEqual(list(marker.embed_stream(iter([]), bytes(16))), [])
        self.assertIsNone(marker._model)

    def test_a_manifest_id_of_the_wrong_length_is_refused(self):
        with self.assertRaises(ValueError):
            watermarker(FakeModel()).message(b"too short")


class DetectTests(PatchedKey):
    def detect(self, logits, count=12):
        return watermarker(FakeModel(logits)).detect_frames(frames(count))

    def test_our_tag_is_detected_and_the_manifest_id_is_read_out(self):
        report = self.detect(logits_for())
        self.assertTrue(report.detected)
        self.assertEqual(report.tag_bits_matching, 128)
        self.assertEqual(report.manifest_id, bytes(range(16)).hex())

    def test_bit_errors_from_compression_are_tolerated_up_to_the_threshold(self):
        self.assertTrue(self.detect(logits_for(tag_errors=32)).detected)       # 96 of 128: exactly the bar
        self.assertFalse(self.detect(logits_for(tag_errors=33)).detected)      # 95 of 128: just under it

    def test_a_manifest_id_with_errors_is_still_reported_for_nearest_match_lookup(self):
        report = self.detect(logits_for(id_errors=10))
        self.assertTrue(report.detected)
        self.assertNotEqual(report.manifest_id, bytes(range(16)).hex())       # it carries the errors it was read with

    def test_someone_elses_mark_a_random_video_and_silence_are_not_ours(self):
        other = torch.tensor([1.0 if b else -1.0 for b in vw.platform_bits(b"another-platform") + [0] * 128])
        self.assertFalse(self.detect(other).detected)
        rng = np.random.default_rng(3)
        for _ in range(20):
            noise = torch.tensor(rng.standard_normal(vw.MESSAGE_BITS), dtype=torch.float32)
            self.assertFalse(self.detect(noise).detected)
        self.assertFalse(self.detect(torch.zeros(vw.MESSAGE_BITS)).detected)
        self.assertIsNone(self.detect(other).manifest_id)                      # nothing is read out of a mark that is not ours

    def test_no_frames_and_very_few_frames_say_so(self):
        empty = watermarker(FakeModel()).detect_frames([])
        self.assertFalse(empty.detected)
        self.assertIn("no frames", " ".join(empty.warnings))
        self.assertIn("fewer than 8", " ".join(self.detect(logits_for(), count=3).warnings))

    def test_the_report_carries_its_method(self):
        self.assertIn("videoseal", self.detect(logits_for()).to_dict()["method"])

    def test_detect_video_samples_evenly_through_the_file(self):
        seen = []

        def fake_read(path):
            for i in range(300):
                seen.append(i)
                yield frames(1, seed=i)[0]

        marker = watermarker(FakeModel(logits_for()))
        with mock.patch("video_io.probe", return_value=mock.Mock(frame_count=300)), mock.patch("video_io.read_frames", fake_read):
            report = marker.detect_video("x.mp4", sample_frames=10)
        self.assertEqual(report.frames_analysed, 10)
        self.assertGreater(max(seen), 250)                                      # spread to the end, not the first ten frames


class LoadTests(unittest.TestCase):
    def test_missing_files_raise_with_the_fix_and_the_failure_is_cached(self):
        marker = vw.VideoWatermarker()
        with mock.patch.object(vw.VideoWatermarker, "files_present", return_value=False):
            with self.assertRaises(vw.VideoWatermarkUnavailable) as caught:
                marker.detect_frames(frames(10))
        self.assertIn("--only videoseal", str(caught.exception))
        with self.assertRaises(vw.VideoWatermarkUnavailable):
            marker.detect_frames(frames(10))

    def test_a_checkpoint_that_fails_its_pinned_hash_is_refused(self):
        marker = vw.VideoWatermarker()
        with mock.patch.object(vw.VideoWatermarker, "files_present", return_value=True), \
                mock.patch("video_watermark._sha256", return_value="0" * 64):
            with self.assertRaises(vw.VideoWatermarkUnavailable) as caught:
                marker.detect_frames(frames(10))
        self.assertIn("pinned SHA-256", str(caught.exception))


# --------------------------------------------------------------------------- the real model
@unittest.skipUnless(vw.VideoWatermarker.files_present(), "VideoSeal weights are not on disk (scripts/fetch_vision_models.py --only videoseal)")
class RealVideoSealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patcher = mock.patch("watermark_engine.signing_key", return_value=KEY)
        cls.patcher.start()
        cls.marker = vw.VideoWatermarker()
        cls.ident = bytes(range(16, 32))
        # a smooth, face-like picture with some texture: the mark needs content to hide in
        yy, xx = np.mgrid[0:256, 0:256]
        base = np.stack([(120 + 60 * np.sin(xx / 29.0)), (110 + 50 * np.cos(yy / 23.0)), 130 + 40 * np.sin((xx + yy) / 41.0)], axis=-1)
        rng = np.random.default_rng(1)
        cls.clip = [np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8) for _ in range(40)]
        cls.marked = list(cls.marker.embed_stream(iter(cls.clip), cls.ident))

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_marked_frames_carry_our_tag_and_id_and_the_originals_do_not(self):
        report = self.marker.detect_frames(self.marked)
        self.assertTrue(report.detected)
        self.assertEqual(report.tag_bits_matching, 128)
        self.assertEqual(report.manifest_id, self.ident.hex())
        self.assertFalse(self.marker.detect_frames(self.clip).detected)

    def test_the_mark_is_quiet_by_the_numbers(self):
        mse = np.mean((np.stack(self.marked).astype(float) - np.stack(self.clip).astype(float)) ** 2)
        self.assertGreater(10 * np.log10(255**2 / mse), 42)  # PSNR in dB (measured ~46 here); 40 is the usual "invisible" bar

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg is not installed")
    def test_it_survives_the_h264_encode_the_renderer_uses_and_a_harsher_one(self):
        import video_io

        # Measured with this clip at strength 0.3: CRF 20 -> 128/128 bits, CRF 23 -> 102/128 (the rule needs 96).
        for crf, minimum in ((20, 0.95), (23, 0.75)):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "marked.mp4"
                with video_io.VideoWriter(path, 256, 256, 25, crf=crf) as writer:
                    for frame in self.marked:
                        writer.write(frame)
                report = self.marker.detect_video(path, sample_frames=16)
            self.assertTrue(report.detected, f"crf {crf}: {report.to_dict()}")
            self.assertGreaterEqual(report.bit_accuracy, minimum, f"crf {crf}")

    def test_another_secret_does_not_recognise_the_mark(self):
        with mock.patch("watermark_engine.signing_key", return_value=b"a-different-secret"):
            self.assertFalse(self.marker.detect_frames(self.marked).detected)


if __name__ == "__main__":
    unittest.main()
