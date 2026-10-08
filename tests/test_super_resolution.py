"""Super-resolution (T8.8): tile assembly, the missing-weights error, and when the renderer uses it."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import render_engine
import super_resolution
from contracts import RenderQuality


class NearestNeighbour(torch.nn.Module):
    """Stands in for the network: 4x nearest-neighbour, so the tiled result has one exact answer."""

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")


class UpscaleTests(unittest.TestCase):
    def resolver(self):
        resolver = super_resolution.SuperResolver(weights=Path("unused.pth"))
        resolver._model = NearestNeighbour()
        return resolver

    def test_tiles_are_reassembled_exactly_with_no_seams_or_gaps(self):
        image = np.random.default_rng(0).integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
        with mock.patch.object(super_resolution, "TILE", 16):     # several tiles, ragged edges
            out = self.resolver().upscale(image)
        np.testing.assert_array_equal(out, image.repeat(4, axis=0).repeat(4, axis=1))

    def test_the_real_network_has_the_published_shape(self):
        net = super_resolution._build_network()
        self.assertEqual(len(net.body), 67)                         # conv, prelu, 32 x (conv, prelu), conv
        self.assertEqual(tuple(net(torch.zeros(1, 3, 8, 8)).shape), (1, 3, 32, 32))

    def test_missing_weights_name_the_fetch_command(self):
        with self.assertRaisesRegex(super_resolution.SuperResolutionUnavailable, "fetch_vision_models.py --only realesrgan"):
            super_resolution.SuperResolver(weights=Path(tempfile.gettempdir()) / "nope.pth").upscale(np.zeros((4, 4, 3), np.uint8))


class RendererUseTests(unittest.TestCase):
    def run_it(self, quality, size, available=True):
        fake = mock.Mock(available=available)
        fake.upscale.side_effect = lambda img: img.repeat(4, axis=0).repeat(4, axis=1)
        warnings = []
        with mock.patch("super_resolution.shared_resolver", return_value=fake):
            image, record = render_engine._maybe_super_resolve(np.zeros((size, size, 3), np.uint8), quality, warnings)
        return image, record, warnings, fake

    def test_preview_never_enlarges(self):
        image, record, warnings, fake = self.run_it(RenderQuality.PREVIEW, 256)
        self.assertEqual((image.shape[0], record, warnings), (256, None, []))
        fake.upscale.assert_not_called()

    def test_1080p_enlarges_a_small_photo_and_says_the_detail_is_synthesised(self):
        image, record, warnings, _ = self.run_it(RenderQuality.HD_1080P, 512)
        self.assertEqual(image.shape[:2], (2048, 2048))
        self.assertEqual((record["photo"], record["enlarged"]), ([512, 512], [2048, 2048]))
        self.assertIn("synthesised", record["method"])
        self.assertTrue(any("synthesised" in w for w in warnings))
        self.assertEqual(render_engine.output_size(2048, 2048, RenderQuality.HD_1080P), (1080, 1080))

    def test_a_photo_already_at_720p_or_more_is_left_alone(self):
        for size in (720, 1024, 1200):
            image, record, _, fake = self.run_it(RenderQuality.HD_1080P, size)
            self.assertIsNone(record)
            fake.upscale.assert_not_called()
        self.assertIsNotNone(self.run_it(RenderQuality.HD_1080P, 718)[1])   # just under 720p is enlarged

    def test_missing_weights_keep_the_photo_size_and_warn_with_the_fix(self):
        image, record, warnings, _ = self.run_it(RenderQuality.HD_1080P, 512, available=False)
        self.assertEqual((image.shape[0], record), (512, None))
        self.assertIn("fetch_vision_models.py --only realesrgan", warnings[0])


if __name__ == "__main__":
    unittest.main()
