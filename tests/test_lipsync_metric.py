"""SyncNet LSE-C / LSE-D (G2-08). The network is never loaded here."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import lipsync_metric
from lipsync_metric import (
    MAX_SHIFT,
    LipSyncMetricError,
    LipSyncScore,
    SyncNetScorer,
    SyncNetUnavailable,
    crop_padded,
    sync_scores,
    syncnet_crop_box,
)


def embeddings(windows=120, dim=32, seed=0):
    return np.random.default_rng(seed).normal(size=(windows, dim)).astype(np.float32)


class SyncScoreTests(unittest.TestCase):
    def test_identical_streams_are_in_sync(self):
        lip = embeddings()
        confidence, distance, offset = sync_scores(lip, lip.copy())
        self.assertEqual(offset, 0)
        self.assertAlmostEqual(distance, 0.0, places=5)
        self.assertGreater(confidence, 5.0)

    def test_a_shift_is_recovered_with_its_sign(self):
        base = embeddings(160)
        for shift in (3, 7):
            late = np.roll(base, shift, axis=0)     # audio lags the video
            early = np.roll(base, -shift, axis=0)
            self.assertEqual(sync_scores(base, late)[2], -shift)
            self.assertEqual(sync_scores(base, early)[2], shift)

    def test_unrelated_streams_have_no_confidence(self):
        confidence, distance, _ = sync_scores(embeddings(seed=1), embeddings(seed=2))
        self.assertLess(confidence, 0.5)
        self.assertGreater(distance, 5.0)

    def test_search_window(self):
        self.assertEqual(MAX_SHIFT, 15)  # the published protocol; changing it changes the metric


class CropTests(unittest.TestCase):
    def test_crop_is_square_wider_than_the_face_and_shifted_down(self):
        x0, y0, x1, y1 = syncnet_crop_box(100, 100, 200, 200)
        self.assertEqual(x1 - x0, y1 - y0)
        self.assertEqual((x0, x1), (60, 340))
        self.assertEqual((y0, y1), (100, 380))

    def test_crop_pads_when_it_runs_off_the_frame(self):
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        crop = crop_padded(frame, (-20, 50, 60, 130))
        self.assertEqual(crop.shape, (80, 80, 3))
        self.assertEqual(int(crop[0, 0, 0]), 110)     # padding
        self.assertEqual(int(crop[0, 30, 0]), 200)    # image
        self.assertEqual(int(crop[70, 30, 0]), 110)   # below the frame


class ScorerTests(unittest.TestCase):
    def test_missing_weights_name_the_fetch_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            scorer = SyncNetScorer(model_path=Path(tmp) / "syncnet_v2.model")
            self.assertFalse(scorer.available)
            for _ in range(2):  # the failure is cached, and still actionable
                with self.assertRaises(SyncNetUnavailable) as ctx:
                    scorer._load()
                self.assertIn("fetch_vision_models.py --only syncnet", str(ctx.exception))

    def test_corrupt_weights_are_reported_not_crashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "syncnet_v2.model"
            path.write_bytes(b"not a checkpoint")
            with self.assertRaises(SyncNetUnavailable) as ctx:
                SyncNetScorer(model_path=path, device="cpu")._load()
        self.assertIn("corrupt", str(ctx.exception))

    def test_too_short_a_clip_is_refused(self):
        scorer = SyncNetScorer()
        scorer._model = object()
        with self.assertRaises(LipSyncMetricError) as ctx:
            scorer.embed(np.zeros((8, 224, 224, 3), dtype=np.uint8), np.zeros((13, 40), dtype=np.float32))
        self.assertIn("too short", str(ctx.exception))

    def test_network_shapes_on_cpu(self):
        import torch

        model = lipsync_metric._build_network().eval()
        with torch.no_grad():
            lip = model.forward_lip(torch.zeros(2, 3, 5, 224, 224))
            audio = model.forward_aud(torch.zeros(2, 1, 13, 20))
        self.assertEqual(tuple(lip.shape), (2, 1024))
        self.assertEqual(tuple(audio.shape), (2, 1024))

    def test_embed_windows_video_and_audio_together(self):
        import torch

        scorer = SyncNetScorer(device="cpu")

        class Fake:
            def forward_lip(self, x):
                self.lip_shape = tuple(x.shape)
                return torch.zeros(x.shape[0], 4)

            def forward_aud(self, x):
                self.aud_shape = tuple(x.shape)
                return torch.ones(x.shape[0], 4)

        scorer._model = Fake()
        lip, audio = scorer.embed(
            np.zeros((30, 224, 224, 3), dtype=np.uint8), np.zeros((13, 120), dtype=np.float32), batch_size=25
        )
        self.assertEqual(lip.shape, (25, 4))   # 30 frames - 5-frame window
        self.assertEqual(audio.shape, (25, 4))
        self.assertEqual(scorer._model.lip_shape[1:], (3, 5, 224, 224))
        self.assertEqual(scorer._model.aud_shape[1:], (1, 13, 20))

    def test_mfcc_is_13_coefficients_at_100hz(self):
        features = lipsync_metric.mfcc_features(np.zeros(16000, dtype=np.float32))
        self.assertEqual(features.shape[0], 13)
        self.assertAlmostEqual(features.shape[1], 100, delta=2)


class ScoreVideoTests(unittest.TestCase):
    def test_score_names_its_method(self):
        payload = LipSyncScore(lse_c=6.1234, lse_d=7.9876, offset_frames=-1, windows=100, frames=105).to_dict()
        self.assertEqual(payload["lseC"], 6.123)
        self.assertEqual(payload["offsetMs"], -40)
        self.assertIn("syncnet-v2", payload["method"])

    def test_video_without_audio_cannot_be_scored(self):
        info = mock.Mock(has_video=True, has_audio=False)
        scorer = mock.Mock(available=True)
        with mock.patch("lipsync_metric.video_io.probe", return_value=info):
            with self.assertRaises(LipSyncMetricError):
                lipsync_metric.score_video("x.mp4", scorer=scorer)

    def test_pipeline_with_a_known_face_box(self):
        lip = embeddings(60)
        scorer = mock.Mock(available=True)
        scorer.embed.return_value = (lip, np.roll(lip, 4, axis=0))
        frames = [np.zeros((120, 160, 3), dtype=np.uint8)] * 65
        with mock.patch("lipsync_metric.video_io.probe", return_value=mock.Mock(has_video=True, has_audio=True, video_duration=1.0)), \
             mock.patch("lipsync_metric.video_io.read_frames", return_value=iter(frames)), \
             mock.patch("lipsync_metric.video_io.read_audio", return_value=np.zeros(16000 * 3, dtype=np.float32)):
            score = lipsync_metric.score_video("x.mp4", face_box=(40, 20, 60, 80), scorer=scorer)
        crops, mfcc = scorer.embed.call_args.args
        # 65 frames although the container claimed 1.0 s: the buffer grows.
        self.assertEqual(crops.shape, (65, 224, 224, 3))
        self.assertEqual(crops.dtype, np.uint8)
        self.assertEqual(mfcc.shape[0], 13)
        self.assertEqual(score.offset_frames, -4)
        self.assertEqual(score.frames, 65)
        self.assertTrue(score.warnings)  # 4 frames off is flagged


if __name__ == "__main__":
    unittest.main()
