"""
Wav2Lip neural lip sync (task G2-07).

The optional, GPU render engine: a network that repaints the lower half of a
face crop from a 0.2 s window of mel spectrogram. It produces more natural
lip shapes than the blendshape warp, at two costs this module is explicit
about:

* **Licence.** The published checkpoints were trained on LRS2 and are for
  research / non-commercial use only. They are never fetched automatically;
  ``scripts/fetch_vision_models.py --accept-licence wav2lip`` is a decision a
  person has to make.
* **Resolution.** The network works on a 96x96 crop, so the mouth it returns
  is soft when pasted into a large frame. The paste is feathered and limited
  to the face box to keep that contained.

The architecture below is the one from Prajwal et al., "A Lip Sync Expert Is
All You Need for Speech to Lip Generation In the Wild" (2020). Module names
match the published checkpoints so their state dicts load unchanged.

There is no fallback in here. If the checkpoint is missing the engine raises
with the command that fetches it; choosing another engine is the caller's
decision and shows up in the render result by name.

Concepts used here, explained once:

**Mel spectrogram.** The audio is cut into short overlapping slices; each slice
becomes 80 numbers describing its energy in 80 frequency bands spaced the way
hearing is (the mel scale). The result is an image of the sound, 80 rows tall
and 80 columns per second.

**Mel window.** For each video frame the network is given 16 mel columns, which
is 0.2 s of sound around that frame. That window is what decides the mouth
shape it paints (``mel_chunks``).

**Face crop and paste-back.** The network never sees the whole frame. The face
box is cut out and shrunk to 96x96, the network returns a new 96x96 face with
the lower half repainted, and that result is enlarged back to the box size and
blended into the original frame through a soft-edged mask (``_feather_mask``),
so no hard seam shows where the new mouth meets the old face.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

import gpu_utils
from model_registry import VISION_FETCH_COMMAND, WAV2LIP_CHECKPOINTS

logger = logging.getLogger(__name__)

# The network's fixed input sizes: a 96x96 face crop and 16 mel columns (0.2 s) per frame,
# from audio at 16 kHz. MELS_PER_SECOND follows from the hop: 16000 / _HOP = 80.
IMG_SIZE = 96
MEL_STEP = 16
MEL_SAMPLE_RATE = 16000
MELS_PER_SECOND = 80.0

# How far each frame's audio window is moved earlier, in seconds.
#
# Measured, not derived (T2.3, 8 Oct 2026). With windows starting exactly at
# each frame - the reference inference convention - SyncNet put the best
# audio-video offset at -3 frames (the mouth ~120 ms ahead of the sound) on
# every clip tried: 3.0 s and 6.4 s of Kokoro English and 5.0 s of MMS Hindi.
# Sweeping the shift moved the offset by exactly one frame per frame of shift,
# and 3 frames earlier gave offset 0 with the best or equal LSE-C/LSE-D on all
# three (e.g. 9.01 -> 10.20 LSE-C). The blendshape engine reads offset 0 on
# the same clips, so the lead is in this path, not in the metric; its cause
# (window convention, STFT centring) was not isolated. A calibration knob on
# purpose: re-measure with scripts or the lipsync-score endpoint if the audio
# front end changes.
AUDIO_LEAD_SECONDS = 0.12
# GPU memory freed (by unloading other models) before Wav2Lip loads.
REQUIRED_VRAM_MB = 1200

# Audio front end, fixed by how the checkpoints were trained.
# n_fft / win: 800 samples (50 ms) per slice; hop: a new slice every 200 samples (12.5 ms).
# Pre-emphasis boosts high frequencies before analysis; the dB limits set the range that is
# then scaled into [-_MAX_ABS, _MAX_ABS]. Changing any of these breaks the trained weights.
_N_FFT = 800
_HOP = 200
_WIN = 800
_N_MELS = 80
_FMIN = 55
_FMAX = 7600
_PREEMPHASIS = 0.97
_REF_LEVEL_DB = 20.0
_MIN_LEVEL_DB = -100.0
_MAX_ABS = 4.0


class Wav2LipUnavailable(RuntimeError):
    """The Wav2Lip checkpoint is missing or could not be loaded."""


def _build_network():
    """Construct the (untrained) Wav2Lip generator."""
    import torch
    from torch import nn

    # The classes are defined inside the function so torch is imported only when Wav2Lip is used.

    # Convolution + batch norm + ReLU, the repeated building block. With residual=True the input
    # is added back to the output (a "skip"), which needs equal input and output channels.
    class Conv2d(nn.Module):
        def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size, stride, padding),
                nn.BatchNorm2d(cout),
            )
            self.act = nn.ReLU()
            self.residual = residual

        def forward(self, x):
            out = self.conv_block(x)
            if self.residual:
                out = out + x
            return self.act(out)

    # A transposed convolution upsamples: with stride 2 it doubles the height and width.
    # The decoder uses it to grow the 1x1 code back to a 96x96 image.
    class Conv2dTranspose(nn.Module):
        def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                nn.BatchNorm2d(cout),
            )
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.conv_block(x))

    def res(channels: int, count: int) -> List[nn.Module]:
        """``count`` residual blocks that keep the channel count and image size."""
        return [
            Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, residual=True)
            for _ in range(count)
        ]

    class Wav2Lip(nn.Module):
        def __init__(self):
            super().__init__()
            # Face encoder: 6 input channels (masked crop + reference crop, 3 colours each),
            # halved in size block by block. The trailing comments give each block's output size.
            self.face_encoder_blocks = nn.ModuleList(
                [
                    nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),  # 96
                    nn.Sequential(Conv2d(16, 32, 3, 2, 1), *res(32, 2)),  # 48
                    nn.Sequential(Conv2d(32, 64, 3, 2, 1), *res(64, 3)),  # 24
                    nn.Sequential(Conv2d(64, 128, 3, 2, 1), *res(128, 2)),  # 12
                    nn.Sequential(Conv2d(128, 256, 3, 2, 1), *res(256, 2)),  # 6
                    nn.Sequential(Conv2d(256, 512, 3, 2, 1), *res(512, 1)),  # 3
                    nn.Sequential(
                        Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1
                        Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                    ),
                ]
            )

            # Audio encoder: squeezes one (80, 16) mel window down to a 512-number code.
            self.audio_encoder = nn.Sequential(
                Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                *res(32, 2),
                Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
                *res(64, 2),
                Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
                *res(128, 2),
                Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
                *res(256, 1),
                Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )

            # Decoder: grows the audio code back to 96x96. Input channel counts (1024, 768, ...)
            # include the matching encoder output joined on in forward(); that is the U-Net
            # pattern, which lets fine face detail skip past the bottleneck.
            self.face_decoder_blocks = nn.ModuleList(
                [
                    nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),
                    nn.Sequential(
                        Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  # 3
                        *res(512, 1),
                    ),
                    nn.Sequential(
                        Conv2dTranspose(1024, 512, 3, 2, 1, output_padding=1),  # 6
                        *res(512, 2),
                    ),
                    nn.Sequential(
                        Conv2dTranspose(768, 384, 3, 2, 1, output_padding=1),  # 12
                        *res(384, 2),
                    ),
                    nn.Sequential(
                        Conv2dTranspose(512, 256, 3, 2, 1, output_padding=1),  # 24
                        *res(256, 2),
                    ),
                    nn.Sequential(
                        Conv2dTranspose(320, 128, 3, 2, 1, output_padding=1),  # 48
                        *res(128, 2),
                    ),
                    nn.Sequential(
                        Conv2dTranspose(160, 64, 3, 2, 1, output_padding=1),  # 96
                        *res(64, 2),
                    ),
                ]
            )

            # Final 3-channel image; Sigmoid squeezes every value into 0..1.
            self.output_block = nn.Sequential(
                Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
            )

        def forward(self, audio_sequences, face_sequences):
            # audio: (B, 1, 80, 16)   face: (B, 6, 96, 96)
            audio_embedding = self.audio_encoder(audio_sequences)  # (B, 512, 1, 1)
            feats = []
            x = face_sequences
            # Keep every encoder output for the skip connections below.
            for block in self.face_encoder_blocks:
                x = block(x)
                feats.append(x)
            x = audio_embedding
            for block in self.face_decoder_blocks:
                x = block(x)
                # pop() takes the last encoder output first: the decoder climbs back up in the
                # reverse order the encoder went down, so the sizes match at each step.
                x = torch.cat((x, feats.pop()), dim=1)
            return self.output_block(x)

    return Wav2Lip()


def melspectrogram(wav: np.ndarray) -> np.ndarray:
    """
    The exact mel front end Wav2Lip was trained with: ``(80, T)`` at 80 Hz.

    ``wav`` is mono float audio at 16 kHz.
    """
    import librosa
    from scipy import signal

    # Array, not tuple: lfilter only returns (y, zf) when passed initial state.
    emphasised = np.asarray(signal.lfilter([1.0, -_PREEMPHASIS], [1.0], wav))
    spectrum = np.abs(
        librosa.stft(y=emphasised, n_fft=_N_FFT, hop_length=_HOP, win_length=_WIN)
    )
    basis = librosa.filters.mel(
        sr=MEL_SAMPLE_RATE, n_fft=_N_FFT, n_mels=_N_MELS, fmin=_FMIN, fmax=_FMAX
    )
    # The filter bank is a matrix: multiplying sums the linear-frequency bins into 80 mel bands.
    mel = basis @ spectrum
    # The amplitude of -100 dB; flooring there avoids log10(0) on silent slices.
    min_level = np.exp(_MIN_LEVEL_DB / 20.0 * np.log(10.0))
    # Amplitude to decibels, relative to the training reference level.
    db = 20.0 * np.log10(np.maximum(min_level, mel)) - _REF_LEVEL_DB
    # Map the dB range linearly onto [-4, 4], the scale the network was trained on.
    normalised = (2.0 * _MAX_ABS) * ((db - _MIN_LEVEL_DB) / (-_MIN_LEVEL_DB)) - _MAX_ABS
    return np.clip(normalised, -_MAX_ABS, _MAX_ABS).astype(np.float32)


def mel_chunks(
    mel: np.ndarray,
    fps: float,
    frame_count: int,
    lead_seconds: float = AUDIO_LEAD_SECONDS,
) -> np.ndarray:
    """
    One ``(80, 16)`` mel window per video frame.

    Frame ``i`` starts at mel column ``(i / fps - lead_seconds) * 80``; with
    ``lead_seconds=0`` that is the reference implementation's ``i * 80 / fps``.
    Windows clamp to the clip: early frames reuse the first 16 columns and
    late ones the last 16, so every frame gets a full window.
    """
    total = mel.shape[1]
    # A clip shorter than one window is padded with the quietest value so a window still exists.
    if total < MEL_STEP:
        mel = np.pad(mel, ((0, 0), (0, MEL_STEP - total)), constant_values=-_MAX_ABS)
        total = MEL_STEP
    # Mel columns per video frame: 80 / 25 = 3.2 at 25 fps, so windows overlap heavily.
    step = MELS_PER_SECOND / float(fps)
    chunks = np.empty((frame_count, _N_MELS, MEL_STEP), dtype=np.float32)
    lead_columns = lead_seconds * MELS_PER_SECOND
    for index in range(frame_count):
        # Floor, like the reference, then clamp into the clip at both ends.
        start = int(math.floor(index * step - lead_columns))
        start = min(max(start, 0), total - MEL_STEP)
        chunks[index] = mel[:, start : start + MEL_STEP]
    return chunks


def face_box_from_bbox(
    x: int, y: int, width: int, height: int, image_width: int, image_height: int
) -> Tuple[int, int, int, int]:
    """
    The crop Wav2Lip sees, as ``(x0, y0, x1, y1)``.

    The landmark bounding box, with a little extra below the chin: the network
    was trained on detector boxes padded at the bottom, and without the chin
    in frame it clips the jaw.
    """
    pad_bottom = int(round(height * 0.06))
    # Clamp every edge to the image, so the box can be used directly as an array slice.
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(image_width, x + width)
    y1 = min(image_height, y + height + pad_bottom)
    return x0, y0, x1, y1


def _feather_mask(height: int, width: int) -> np.ndarray:
    """
    Soft-edged paste mask covering the lower part of the face box.

    The network only generates the lower half; its upper half is a 96 px
    copy of the input. Pasting that back would replace sharp eyes and brows
    with a blurry upsample, so the mask starts just above the midline.
    """
    import cv2

    # 1.0 where the generated pixels are used, 0.0 where the original stays; the blur makes
    # the values in between, which is the feathered edge.
    mask = np.zeros((height, width), dtype=np.float32)
    # Keep 8% of the box clear on the sides and bottom, so the edge of the crop never shows.
    inset_y = max(1, int(height * 0.08))
    inset_x = max(1, int(width * 0.08))
    mask[int(height * 0.46) : height - inset_y, inset_x : width - inset_x] = 1.0
    # Blur width scales with the box, so the feather looks the same at any resolution.
    sigma = max(1.0, min(height, width) * 0.04)
    # Kernel size (0, 0) lets OpenCV derive it from sigma. [:, :, None] adds a channel axis,
    # so the mask multiplies all three colour channels at once.
    return cv2.GaussianBlur(mask, (0, 0), sigma)[:, :, None]


class Wav2LipEngine:
    """Lazy-loading Wav2Lip inference with cached failure."""

    def __init__(
        self,
        checkpoint: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        # An explicit checkpoint path wins; otherwise the known locations are searched.
        self._explicit = Path(checkpoint) if checkpoint else None
        # None means "decide at load time": CUDA when PyTorch sees it, otherwise the CPU.
        self.device = device
        self.batch_size = batch_size
        self._model = None
        # Cached failure: after one failed load, later calls raise the same message at once.
        self._load_error: Optional[str] = None
        # Lets gpu_utils unload Wav2Lip when another heavy model needs the GPU (golden rule 5).
        gpu_utils.register_releaser("wav2lip", self.release)

    @property
    def checkpoint(self) -> Optional[Path]:
        """The checkpoint that would be loaded, or ``None`` if none is on disk."""
        candidates: Sequence[Path] = (
            (self._explicit,) if self._explicit else WAV2LIP_CHECKPOINTS
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    @property
    def available(self) -> bool:
        """True when a checkpoint file is on disk (not a guarantee that it loads)."""
        return self.checkpoint is not None

    def _load(self):
        """Return the loaded network, loading it on first use; raise ``Wav2LipUnavailable`` with the fix."""
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise Wav2LipUnavailable(self._load_error)

        checkpoint = self.checkpoint
        if checkpoint is None:
            self._load_error = (
                "Wav2Lip checkpoint not found in .models/wav2lip/. It is under a "
                "research / non-commercial licence, so a person has to opt in: "
                f"{VISION_FETCH_COMMAND} --only wav2lip --accept-licence wav2lip. "
                "The 'blendshape' engine renders without it."
            )
            raise Wav2LipUnavailable(self._load_error)

        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda":
            gpu_utils.ensure_vram(REQUIRED_VRAM_MB, "Wav2Lip", keep="wav2lip")
        try:
            # weights_only: the file comes from the internet, and a pickle is
            # code. A state dict needs nothing beyond tensors.
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            # Some checkpoints wrap the weights in {"state_dict": ...}; others are the weights.
            state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            # Weights saved from nn.DataParallel carry a "module." prefix on every key; strip it
            # so the names match this single-device network.
            state = {key.replace("module.", "", 1): value for key, value in state.items()}
            model = _build_network()
            model.load_state_dict(state)
            self._model = model.to(device).eval()
            self.device = device
            logger.info("Loaded Wav2Lip from %s on %s", checkpoint.name, device)
            return self._model
        except gpu_utils.InsufficientVRAM:
            # A full GPU can clear later, so it is re-raised without being cached as a failure.
            raise
        except Exception as exc:  # noqa: BLE001
            self._load_error = (
                f"Could not load the Wav2Lip checkpoint {checkpoint.name}: {exc}. "
                f"If the file is corrupt, delete it and re-run: {VISION_FETCH_COMMAND} "
                "--only wav2lip --accept-licence wav2lip"
            )
            raise Wav2LipUnavailable(self._load_error) from exc

    def audio_windows(
        self, audio_path: Union[str, Path], fps: float, frame_count: int
    ) -> np.ndarray:
        """Mel windows for ``frame_count`` frames of ``audio_path``."""
        import librosa

        # librosa resamples to 16 kHz and mixes to mono while reading, as the front end requires.
        wav, _ = librosa.load(str(audio_path), sr=MEL_SAMPLE_RATE, mono=True)
        return mel_chunks(melspectrogram(wav), fps, frame_count)

    def sync_frames(
        self,
        frames: Iterable[np.ndarray],
        face_box: Tuple[int, int, int, int],
        audio_path: Union[str, Path],
        fps: float,
        frame_count: int,
    ) -> Iterator[np.ndarray]:
        """
        Repaint the mouth of each RGB frame to match the audio.

        ``frames`` may already carry blinks and brow motion: the network only
        rewrites the lower half of ``face_box``, and takes the upper half of
        each incoming frame as its identity reference.
        """
        import cv2
        import torch

        model = self._load()
        mels = self.audio_windows(audio_path, fps, frame_count)
        x0, y0, x1, y1 = face_box
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise ValueError(f"face box {face_box} is too small for Wav2Lip")
        # The face box is fixed for the whole clip, so the mask is built once.
        mask = _feather_mask(y1 - y0, x1 - x0)

        # Frames are collected into batches: the GPU is far faster on many crops at once.
        batch: List[np.ndarray] = []
        start = 0

        def flush(items: List[np.ndarray], offset: int) -> Iterator[np.ndarray]:
            """Run one batch through the network and yield the pasted-back frames in order."""
            # Cut the face box out of each frame and shrink it to the network's 96x96.
            crops = np.stack(
                [
                    cv2.resize(frame[y0:y1, x0:x1], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                    for frame in items
                ]
            )
            # The checkpoints were trained on BGR crops.
            crops = crops[:, :, :, ::-1]
            # Black out the lower half: the network must paint the mouth, not copy it.
            masked = crops.copy()
            masked[:, IMG_SIZE // 2 :] = 0
            # Stack masked and full crops into 6 channels, scaled from 0..255 to 0..1.
            faces = np.concatenate([masked, crops], axis=3).astype(np.float32) / 255.0
            # PyTorch wants channels first: (batch, H, W, C) becomes (batch, C, H, W).
            face_tensor = torch.from_numpy(faces.transpose(0, 3, 1, 2)).to(self.device)
            # The windows for exactly these frames, with a channel axis: (batch, 1, 80, 16).
            mel_tensor = torch.from_numpy(
                mels[offset : offset + len(items)][:, None, :, :]
            ).to(self.device)
            with torch.no_grad():
                predicted = model(mel_tensor, face_tensor)
            # Back to (batch, H, W, C), 0..255, and BGR back to RGB.
            predicted = (predicted.cpu().numpy().transpose(0, 2, 3, 1) * 255.0)[:, :, :, ::-1]
            for frame, mouth in zip(items, predicted, strict=True):
                # Enlarge the 96x96 result back to the face box; cubic keeps it a little sharper.
                patch = cv2.resize(
                    mouth.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC
                ).astype(np.float32)
                out = frame.copy()
                region = out[y0:y1, x0:x1].astype(np.float32)
                # Alpha blend: mask 1 takes the generated pixel, 0 keeps the original.
                out[y0:y1, x0:x1] = np.clip(
                    region * (1 - mask) + patch * mask, 0, 255
                ).astype(np.uint8)
                yield out

        # A generator: frames are yielded as each batch finishes, so the whole clip is never held
        # in memory at once. ``start`` tracks which mel windows belong to the current batch.
        for frame in frames:
            batch.append(frame)
            if len(batch) == self.batch_size:
                yield from flush(batch, start)
                start += len(batch)
                batch = []
        # The last, partly filled batch.
        if batch:
            yield from flush(batch, start)

    def release(self) -> None:
        """Drop the model so another heavy model can use the GPU."""
        if self._model is not None:
            self._model = None
            gpu_utils.empty_cache()


# ---------------------------------------------------------------------------
# Process-wide engine: one copy of the network, however many jobs render.
# ---------------------------------------------------------------------------
_shared_engine: Optional[Wav2LipEngine] = None


def shared_wav2lip_engine() -> Wav2LipEngine:
    """The process-wide engine, created on first use on the configured device."""
    global _shared_engine
    if _shared_engine is None:
        _shared_engine = Wav2LipEngine(device=gpu_utils.preferred_device())
    return _shared_engine
