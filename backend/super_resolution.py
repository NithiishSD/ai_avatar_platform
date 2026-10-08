"""
Super-resolution of the source photo, so a small photo can still give a 1080p video (R-44, N-16).

The renderer animates one photo, so the photo is enlarged **once**, before animation, rather than
every frame of the video: on a CPU one 512 px photo takes about a second, every frame of a
30 s video would take many minutes. The model is Real-ESRGAN's compact general model
(``realesr-general-x4v3``, BSD-3-Clause, xinntao/Real-ESRGAN): a 4x upscaler trained to restore
real photos. Its network (``SRVGGNetCompact``) is small enough to define here, which avoids the
``realesrgan``/``basicsr`` packages and their pins on old ``torchvision``.

What it is not: the added detail is *synthesised* by the network, not recovered. The render
result and the video's manifest say so whenever it was used.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

SCALE = 4
TILE = 256       # pixels per tile edge; bounds memory on a large photo
TILE_PAD = 10    # overlap so tile seams are not visible
METHOD = "Real-ESRGAN realesr-general-x4v3 (photo upscaled once before animation; added detail is synthesised)"


class SuperResolutionUnavailable(RuntimeError):
    """The weights are missing; the message names the fetch command."""


def _build_network() -> Any:
    """SRVGGNetCompact as published with the weights (64 features, 32 convs, PReLU, 4x pixel shuffle)."""
    import torch.nn as nn

    class SRVGGNetCompact(nn.Module):
        def __init__(self, num_feat: int = 64, num_conv: int = 32, upscale: int = SCALE) -> None:
            super().__init__()
            self.upscale = upscale
            # Module names match the checkpoint: body.0 conv, body.1 PReLU, then conv/PReLU pairs, last conv.
            layers: list = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
            for _ in range(num_conv):
                layers += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
            layers.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
            self.body = nn.ModuleList(layers)
            self.upsampler = nn.PixelShuffle(upscale)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            # The network learns the residual over a plain nearest-neighbour enlargement.
            return self.upsampler(out) + nn.functional.interpolate(x, scale_factor=self.upscale, mode="nearest")

    return SRVGGNetCompact()


class SuperResolver:
    def __init__(self, weights: Optional[Path] = None, device: Optional[str] = None) -> None:
        from model_registry import SUPER_RESOLUTION_MODEL

        self.weights = Path(weights) if weights else SUPER_RESOLUTION_MODEL
        self.device = device or "cpu"
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.weights.is_file()

    def _load(self) -> Any:
        if self._model is None:
            import torch

            if not self.available:
                raise SuperResolutionUnavailable(
                    f"super-resolution weights not found at {self.weights}. Fetch them with: "
                    "PYTHONPATH=backend backend/.conda/bin/python scripts/fetch_vision_models.py --only realesrgan"
                )
            model = _build_network()
            state = torch.load(str(self.weights), map_location="cpu", weights_only=True)
            model.load_state_dict(state.get("params", state))
            self._model = model.eval().to(self.device)
            logger.info("Loaded %s on %s", self.weights.name, self.device)
        return self._model

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """RGB uint8 (H, W, 3) -> RGB uint8 (4H, 4W, 3), processed in overlapping tiles."""
        import torch

        with self._lock:
            model = self._load()
            height, width = image.shape[:2]
            source = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
            output = torch.zeros((1, 3, height * SCALE, width * SCALE))
            with torch.no_grad():
                for top in range(0, height, TILE):
                    for left in range(0, width, TILE):
                        # Each tile is cut with a margin, enlarged, and only its centre is kept.
                        y0, x0 = max(0, top - TILE_PAD), max(0, left - TILE_PAD)
                        y1, x1 = min(height, top + TILE + TILE_PAD), min(width, left + TILE + TILE_PAD)
                        enlarged = model(source[:, :, y0:y1, x0:x1].to(self.device)).cpu()
                        th, tw = min(TILE, height - top), min(TILE, width - left)
                        oy, ox = (top - y0) * SCALE, (left - x0) * SCALE
                        output[:, :, top * SCALE:(top + th) * SCALE, left * SCALE:(left + tw) * SCALE] = \
                            enlarged[:, :, oy:oy + th * SCALE, ox:ox + tw * SCALE]
        return (output.squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


_shared: Optional[SuperResolver] = None


def shared_resolver() -> SuperResolver:
    global _shared
    if _shared is None:
        _shared = SuperResolver()
    return _shared
