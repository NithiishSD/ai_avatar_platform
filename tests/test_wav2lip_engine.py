"""Wav2Lip engine (G2-07). No checkpoint is loaded or fetched."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import wav2lip_engine
from wav2lip_engine import (
    IMG_SIZE,
    MEL_STEP,
    Wav2LipEngine,
    Wav2LipUnavailable,
    face_box_from_bbox,
    mel_chunks,
    melspectrogram,
)


class MelTests(unittest.TestCase):
    def test_mel_is_80_bands_at_80hz_and_normalised(self):
        mel = melspectrogram(np.random.default_rng(0).normal(scale=0.1, size=16000).astype(np.float32))
        self.assertEqual(mel.shape[0], 80)
        self.assertAlmostEqual(mel.shape[1], 80, delta=2)
        self.assertLessEqual(float(np.abs(mel).max()), 4.0)

    def test_one_window_per_frame(self):
        mel = np.arange(80 * 200, dtype=np.float32).reshape(80, 200)
        chunks = mel_chunks(mel, fps=25, frame_count=60)
        self.assertEqual(chunks.shape, (60, 80, MEL_STEP))
        np.testing.assert_array_equal(chunks[0], mel[:, :16])
        np.testing.assert_array_equal(chunks[10], mel[:, 32:48])   # frame 10 -> column 10 * 80/25
        np.testing.assert_array_equal(chunks[-1], mel[:, -16:])    # tail reuses the last window

    def test_clip_shorter_than_one_window_is_padded(self):
        self.assertEqual(mel_chunks(np.zeros((80, 5), dtype=np.float32), 25, 3).shape, (3, 80, 16))


class FaceBoxTests(unittest.TestCase):
    def test_box_is_padded_below_the_chin_and_clamped(self):
        self.assertEqual(face_box_from_bbox(100, 100, 200, 200, 512, 512), (100, 100, 300, 312))
        self.assertEqual(face_box_from_bbox(-5, 400, 200, 200, 512, 512), (0, 400, 195, 512))


class EngineTests(unittest.TestCase):
    def test_missing_checkpoint_explains_the_licence_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Wav2LipEngine(checkpoint=Path(tmp) / "wav2lip_gan.pth")
            self.assertFalse(engine.available)
            for _ in range(2):
                with self.assertRaises(Wav2LipUnavailable) as ctx:
                    engine._load()
                message = str(ctx.exception)
                self.assertIn("--accept-licence wav2lip", message)
                self.assertIn("non-commercial", message)

    def test_corrupt_checkpoint_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wav2lip.pth"
            path.write_bytes(b"junk")
            with self.assertRaises(Wav2LipUnavailable) as ctx:
                Wav2LipEngine(checkpoint=path, device="cpu")._load()
        self.assertIn("corrupt", str(ctx.exception))

    def test_network_maps_audio_and_face_to_a_face(self):
        import torch

        model = wav2lip_engine._build_network().eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 1, 80, MEL_STEP), torch.zeros(1, 6, IMG_SIZE, IMG_SIZE))
        self.assertEqual(tuple(out.shape), (1, 3, IMG_SIZE, IMG_SIZE))

    def test_sync_frames_repaints_only_the_face_box(self):
        import torch

        engine = Wav2LipEngine(device="cpu", batch_size=4)

        class White(torch.nn.Module):
            def forward(self, mel, face):
                return torch.ones(face.shape[0], 3, IMG_SIZE, IMG_SIZE)

        engine._model = White()
        engine.audio_windows = lambda path, fps, count: np.zeros((count, 80, MEL_STEP), dtype=np.float32)
        frames = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(6)]
        out = list(engine.sync_frames(iter(frames), (50, 50, 150, 150), "x.wav", 25, 6))
        self.assertEqual(len(out), 6)
        self.assertGreaterEqual(int(out[0][125, 100, 0]), 250)  # mouth area: repainted
        self.assertEqual(int(out[0][10, 10, 0]), 0)          # outside: untouched
        self.assertLess(int(out[0][70, 100, 0]), 8)          # eyes and brows: left at source quality
        self.assertLess(int(out[0][125, 52, 0]), 128)        # feathered edge
        self.assertEqual(int(frames[0].max()), 0)            # inputs not modified

    def test_tiny_face_box_is_refused(self):
        engine = Wav2LipEngine(device="cpu")
        engine._model = object()
        engine.audio_windows = lambda *a: np.zeros((1, 80, MEL_STEP), dtype=np.float32)
        with self.assertRaises(ValueError):
            list(engine.sync_frames(iter([]), (0, 0, 4, 4), "x.wav", 25, 1))


if __name__ == "__main__":
    unittest.main()
