"""
Lip-sync metric: SyncNet LSE-C / LSE-D (task G2-08).

"The mouth follows the speech" is a claim, and golden rule 2 says a claim
needs a number with a method. SyncNet (Chung & Zisserman, "Out of time:
automated lip sync in the wild", 2016) embeds a 0.2 s window of mouth frames
and the matching 0.2 s of audio into one space; the distance between the two
says how well they agree. The two scores the lip-sync literature reports
(Wav2Lip, 2020) are read straight off it:

* **LSE-D** -- mean embedding distance at the best audio/video offset.
  Lower is better. Real talking-head video scores roughly 6.5-7.
* **LSE-C** -- confidence: median distance over all offsets minus the
  minimum. Higher is better. Real video scores roughly 6-8; unsynchronised
  video is near 0-2.
* **offset** -- the shift (in 25 fps frames) at which sync is best. A
  correctly muxed render is within about one frame of zero.

These are *not* a percentage. The roadmap's "> 95% lip sync" has no defined
measurement; LSE-C / LSE-D are what this project can honestly report, and the
result carries its method string so nobody can quote it as anything else.

The procedure mirrors ``syncnet_python``: 25 fps video, a 224x224 crop around
the face, 13 MFCCs at 100 Hz, a +/-15 frame offset search. One deliberate
simplification: the face box comes from the first frame, because this
project's avatars do not move their heads. For footage with head motion that
would under-score, and ``method`` says so.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import gpu_utils
import video_io
from model_registry import SYNCNET_MODEL, VISION_FETCH_COMMAND

logger = logging.getLogger(__name__)

SYNCNET_FPS = 25
SYNCNET_SAMPLE_RATE = 16000
CROP_SIZE = 224
CROP_SCALE = 0.40
WINDOW_FRAMES = 5          # 0.2 s of video
MFCC_PER_FRAME = 4         # 100 Hz MFCC / 25 fps
MAX_SHIFT = 15             # offset search, in frames
REQUIRED_VRAM_MB = 900
MIN_WINDOWS = 10
METHOD = "syncnet-v2 LSE-C/LSE-D, 25 fps, face box from first frame"


class SyncNetUnavailable(RuntimeError):
    """The SyncNet weights could not be loaded; the message names the fix."""


class LipSyncMetricError(RuntimeError):
    """The clip cannot be scored (too short, no face, no audio)."""


@dataclass
class LipSyncScore:
    """One clip's SyncNet result, with the method that produced it."""

    lse_c: float
    lse_d: float
    offset_frames: int
    windows: int
    frames: int
    method: str = METHOD
    face_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "lseC": round(self.lse_c, 3),
            "lseD": round(self.lse_d, 3),
            "offsetFrames": self.offset_frames,
            "offsetMs": int(round(self.offset_frames * 1000.0 / SYNCNET_FPS)),
            "windows": self.windows,
            "frames": self.frames,
            "method": self.method,
            "faceBox": list(self.face_box),
            "warnings": self.warnings,
        }


def _build_network():
    """SyncNet v2. Module names match the published state dict."""
    import torch.nn as nn

    class SyncNet(nn.Module):
        def __init__(self, embedding: int = 1024) -> None:
            super().__init__()
            self.netcnnaud = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
                nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(192), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
                nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(384), nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
                nn.BatchNorm2d(512), nn.ReLU(),
            )
            self.netfcaud = nn.Sequential(
                nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, embedding),
            )
            self.netfclip = nn.Sequential(
                nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, embedding),
            )
            self.netcnnlip = nn.Sequential(
                nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
                nn.BatchNorm3d(96), nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
                nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(256), nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(256), nn.ReLU(inplace=True),
                nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(256), nn.ReLU(inplace=True),
                nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(256), nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
                nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),
                nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            )

        def forward_aud(self, x):
            return self.netfcaud(self.netcnnaud(x).flatten(1))

        def forward_lip(self, x):
            return self.netfclip(self.netcnnlip(x).flatten(1))

    return SyncNet()


def syncnet_crop_box(
    x: int, y: int, width: int, height: int
) -> Tuple[int, int, int, int]:
    """
    The square SyncNet sees, as ``(x0, y0, x1, y1)`` (may exceed the image).

    Same geometry as ``syncnet_python``'s crop: centred on the face
    horizontally, shifted down so the mouth sits in the lower-middle, and
    ``CROP_SCALE`` wider than the detection.
    """
    half = max(width, height) / 2.0
    cx = x + width / 2.0
    cy = y + height / 2.0
    x0 = int(round(cx - half * (1 + CROP_SCALE)))
    x1 = int(round(cx + half * (1 + CROP_SCALE)))
    y0 = int(round(cy - half))
    y1 = int(round(cy + half * (1 + 2 * CROP_SCALE)))
    return x0, y0, x1, y1


def crop_padded(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``box`` from ``frame``, padding with mid-grey where it runs off."""
    x0, y0, x1, y1 = box
    height, width = frame.shape[:2]
    out = np.full((max(1, y1 - y0), max(1, x1 - x0), 3), 110, dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return out


def sync_scores(
    lip: np.ndarray, audio: np.ndarray, max_shift: int = MAX_SHIFT
) -> Tuple[float, float, int]:
    """
    ``(lse_c, lse_d, offset)`` from per-window lip and audio embeddings.

    For every window the lip embedding is compared with the audio embedding
    ``-max_shift .. +max_shift`` windows away; distances are averaged over the
    clip per shift. LSE-D is the smallest mean, LSE-C how far it sits below
    the median, and the offset where it occurred.
    """
    windows = lip.shape[0]
    padded = np.pad(audio, ((max_shift, max_shift), (0, 0)))
    distances = np.empty((windows, 2 * max_shift + 1), dtype=np.float64)
    for shift in range(2 * max_shift + 1):
        distances[:, shift] = np.linalg.norm(lip - padded[shift : shift + windows], axis=1)
    mean = distances.mean(axis=0)
    best = int(mean.argmin())
    return float(np.median(mean) - mean[best]), float(mean[best]), max_shift - best


class SyncNetScorer:
    """Lazy-loading SyncNet with cached failure."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None, device: Optional[str] = None) -> None:
        self.model_path = Path(model_path) if model_path else SYNCNET_MODEL
        self.device = device
        self._model = None
        self._load_error: Optional[str] = None
        gpu_utils.register_releaser("syncnet", self.release)

    @property
    def available(self) -> bool:
        return self.model_path.is_file()

    def _load(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise SyncNetUnavailable(self._load_error)
        if not self.available:
            self._load_error = (
                f"SyncNet weights not found at {self.model_path}. Fetch them with: "
                f"{VISION_FETCH_COMMAND} --only syncnet"
            )
            raise SyncNetUnavailable(self._load_error)

        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda":
            gpu_utils.ensure_vram(REQUIRED_VRAM_MB, "SyncNet", keep="syncnet")
        try:
            state = torch.load(self.model_path, map_location="cpu", weights_only=True)
            model = _build_network()
            model.load_state_dict(state)
            self._model = model.to(device).eval()
            self.device = device
            logger.info("Loaded SyncNet on %s", device)
            return self._model
        except gpu_utils.InsufficientVRAM:
            raise
        except Exception as exc:  # noqa: BLE001
            self._load_error = (
                f"Could not load SyncNet from {self.model_path.name}: {exc}. If the "
                f"file is corrupt, delete it and re-run: {VISION_FETCH_COMMAND} --only syncnet"
            )
            raise SyncNetUnavailable(self._load_error) from exc

    def embed(self, crops: np.ndarray, mfcc: np.ndarray, batch_size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embeddings for every 5-frame window.

        ``crops`` is (frames, 224, 224, 3) uint8 BGR; ``mfcc`` is (13, steps)
        at 100 Hz. Returns ``(lip, audio)``, each (windows, 1024).
        """
        import torch

        model = self._load()
        windows = min(crops.shape[0], mfcc.shape[1] // MFCC_PER_FRAME) - WINDOW_FRAMES
        if windows < MIN_WINDOWS:
            raise LipSyncMetricError(
                f"clip is too short to score: {max(0, windows)} windows, need {MIN_WINDOWS} "
                f"(about {(MIN_WINDOWS + WINDOW_FRAMES) / SYNCNET_FPS:.1f} s)"
            )
        # Kept as uint8 and converted a batch at a time: a float copy of the
        # whole clip is 600 kB per frame. Values stay in 0..255, as trained.
        video = torch.from_numpy(np.ascontiguousarray(crops))
        sound = torch.from_numpy(np.ascontiguousarray(mfcc)).float()
        lip_out: List[np.ndarray] = []
        aud_out: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, windows, batch_size):
                stop = min(windows, start + batch_size)
                # (window, H, W, 3) -> (3, window, H, W)
                lips = torch.stack(
                    [video[i : i + WINDOW_FRAMES].permute(3, 0, 1, 2) for i in range(start, stop)]
                ).float()
                auds = torch.stack(
                    [
                        sound[:, i * MFCC_PER_FRAME : i * MFCC_PER_FRAME + WINDOW_FRAMES * MFCC_PER_FRAME]
                        for i in range(start, stop)
                    ]
                ).unsqueeze(1)
                lip_out.append(model.forward_lip(lips.to(self.device)).cpu().numpy())
                aud_out.append(model.forward_aud(auds.to(self.device)).cpu().numpy())
        return np.concatenate(lip_out), np.concatenate(aud_out)

    def release(self) -> None:
        if self._model is not None:
            self._model = None
            gpu_utils.empty_cache()


_shared_scorer: Optional[SyncNetScorer] = None


def shared_scorer() -> SyncNetScorer:
    global _shared_scorer
    if _shared_scorer is None:
        _shared_scorer = SyncNetScorer(device=gpu_utils.preferred_device())
    return _shared_scorer


def mfcc_features(audio: np.ndarray) -> np.ndarray:
    """13 MFCCs at 100 Hz, shaped (13, steps), from 16 kHz float audio."""
    import python_speech_features

    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    return python_speech_features.mfcc(pcm, SYNCNET_SAMPLE_RATE).T.astype(np.float32)


def score_video(
    video_path: Union[str, Path],
    face_box: Optional[Tuple[int, int, int, int]] = None,
    scorer: Optional[SyncNetScorer] = None,
) -> LipSyncScore:
    """
    Score a rendered clip.

    ``face_box`` is ``(x, y, width, height)`` of the face in the video's own
    pixels; when omitted the face is found in the first frame with MediaPipe.
    """
    import cv2

    scorer = scorer or shared_scorer()
    if not scorer.available:
        scorer._load()  # raises with the fetch command

    info = video_io.probe(video_path)
    if not (info.has_video and info.has_audio):
        raise LipSyncMetricError(f"{Path(video_path).name} needs both a video and an audio stream")

    crop_box: Optional[Tuple[int, int, int, int]] = None
    expected = max(1, int(round(info.video_duration * SYNCNET_FPS)) + 2)
    crops = np.empty((expected, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    count = 0
    for frame in video_io.read_frames(video_path, fps=SYNCNET_FPS):
        if crop_box is None:
            if face_box is None:
                from face_engine import FACE_ENGINE_LOCK, NoFaceDetected, shared_face_engine

                try:
                    with FACE_ENGINE_LOCK:
                        box = shared_face_engine().analyze(frame).bounding_box
                except NoFaceDetected as err:
                    raise LipSyncMetricError(
                        f"no face in the first frame of {Path(video_path).name}"
                    ) from err
                face_box = (box.x, box.y, box.width, box.height)
            crop_box = syncnet_crop_box(*face_box)
        crop = cv2.resize(crop_padded(frame, crop_box), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
        if count == len(crops):  # the container under-reported its length
            crops = np.concatenate([crops, np.empty_like(crops)])
        crops[count] = crop[:, :, ::-1]  # SyncNet was trained on BGR frames
        count += 1
    if count == 0:
        raise LipSyncMetricError(f"{Path(video_path).name} has no decodable frames")
    crops = crops[:count]

    mfcc = mfcc_features(video_io.read_audio(video_path, SYNCNET_SAMPLE_RATE))
    lip, audio = scorer.embed(crops, mfcc)
    lse_c, lse_d, offset = sync_scores(lip, audio)

    warnings: List[str] = []
    if abs(offset) > 2:
        warnings.append(
            f"best sync is {offset:+d} frames from zero; the streams may be misaligned, "
            "or the mouth motion is too weak for SyncNet to lock onto"
        )
    return LipSyncScore(
        lse_c=lse_c,
        lse_d=lse_d,
        offset_frames=offset,
        windows=int(lip.shape[0]),
        frames=count,
        face_box=tuple(int(v) for v in face_box),  # type: ignore[arg-type]
        warnings=warnings,
    )
