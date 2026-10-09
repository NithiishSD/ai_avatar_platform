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

Concepts used here, explained once:

**Embedding.** A neural network that maps an input to a fixed-length vector
of numbers (here 1024) is an *embedding* network. SyncNet has two of them,
one for mouth frames and one for audio, trained so that a mouth and the sound
it is making land close together and mismatched pairs land far apart. The
Euclidean distance between the two vectors is therefore a sync score.

**MFCC** (mel-frequency cepstral coefficients). A compact summary of how a
short slice of audio sounds: the spectrum is warped onto the mel scale (which
spaces pitches the way hearing does) and compressed to 13 numbers per 10 ms
step. It is the audio representation SyncNet was trained on.

**Offset search.** Audio and video may be shifted against each other. The
distance is computed with the audio slid -15..+15 frames against the video;
the shift with the smallest average distance is where sync is best. A
well-synced clip shows a sharp dip at zero, which is what LSE-C measures.

**Per-second agreement.** One offset for the whole clip can hide a part that
drifts. Cutting the clip into one-second blocks and finding each block's own
best offset (``second_agreement``) shows how much of the clip is in sync.
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

# The frame rate, audio rate and crop size SyncNet was trained with; inputs must match them.
SYNCNET_FPS = 25
SYNCNET_SAMPLE_RATE = 16000
CROP_SIZE = 224
CROP_SCALE = 0.40
WINDOW_FRAMES = 5          # 0.2 s of video
MFCC_PER_FRAME = 4         # 100 Hz MFCC / 25 fps
MAX_SHIFT = 15             # offset search, in frames
REQUIRED_VRAM_MB = 900     # GPU memory freed before loading SyncNet
MIN_WINDOWS = 10           # fewer windows than this is too little evidence to score
# Carried in every result, so a score is never quoted without how it was measured (golden rule 2).
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
    # D-12: how many of the clip's 1-second blocks have their own best offset within +/-1 frame.
    seconds_within_one: int = 0
    seconds_scored: int = 0

    def to_dict(self) -> Dict[str, object]:
        """The camelCase JSON shape scripts and the API report, rounded for reading."""
        return {
            # percent is None, not 0, when no second could be scored: "unknown" is not "all bad".
            "secondsWithinOneFrame": {
                "within": self.seconds_within_one, "of": self.seconds_scored,
                "percent": round(100.0 * self.seconds_within_one / self.seconds_scored, 1) if self.seconds_scored else None,
            },
            "lseC": round(self.lse_c, 3),
            "lseD": round(self.lse_d, 3),
            "offsetFrames": self.offset_frames,
            # One frame at 25 fps is 40 ms.
            "offsetMs": int(round(self.offset_frames * 1000.0 / SYNCNET_FPS)),
            "windows": self.windows,
            "frames": self.frames,
            "method": self.method,
            "faceBox": list(self.face_box),
            "warnings": self.warnings,
        }


def _build_network():
    """SyncNet v2. Module names match the published state dict."""
    # A *state dict* is the saved weights keyed by layer name ("netcnnaud.0.weight", ...).
    # load_state_dict only fills a network whose layer names and shapes match exactly, which is
    # why these attribute names and layer sizes must not be changed.
    import torch.nn as nn

    # Defined inside the function so torch is imported only when the metric is actually used.

    class SyncNet(nn.Module):
        def __init__(self, embedding: int = 1024) -> None:
            super().__init__()
            # Audio branch: 2-D convolutions over the MFCC "image" (13 coefficients x 20 steps).
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
            # Fully connected heads that turn each branch's features into the 1024-number embedding.
            self.netfcaud = nn.Sequential(
                nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, embedding),
            )
            self.netfclip = nn.Sequential(
                nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, embedding),
            )
            # Video branch: 3-D convolutions, so the first layer sees the 5 frames together and can
            # respond to motion, not just one still mouth.
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
            """Audio embeddings for a batch of MFCC windows shaped (batch, 1, 13, 20)."""
            # flatten(1) keeps the batch axis and joins the rest into one feature vector.
            return self.netfcaud(self.netcnnaud(x).flatten(1))

        def forward_lip(self, x):
            """Lip embeddings for a batch of frame windows shaped (batch, 3, 5, 224, 224)."""
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
    # Work from the box centre and half of its longer side, so the crop is square.
    half = max(width, height) / 2.0
    cx = x + width / 2.0
    cy = y + height / 2.0
    x0 = int(round(cx - half * (1 + CROP_SCALE)))
    x1 = int(round(cx + half * (1 + CROP_SCALE)))
    # The top edge stays at the face box's top; all the extra height goes below, where the mouth is.
    y0 = int(round(cy - half))
    y1 = int(round(cy + half * (1 + 2 * CROP_SCALE)))
    return x0, y0, x1, y1


def crop_padded(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``box`` from ``frame``, padding with mid-grey where it runs off."""
    x0, y0, x1, y1 = box
    height, width = frame.shape[:2]
    # Start with a canvas of the full box size filled with grey, then copy in the part that
    # overlaps the frame. max(1, ...) keeps a degenerate box from making a zero-size array.
    out = np.full((max(1, y1 - y0), max(1, x1 - x0), 3), 110, dtype=np.uint8)
    # The source rectangle clipped to the frame's edges.
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if sx1 > sx0 and sy1 > sy0:
        # Subtracting x0 / y0 converts frame coordinates into canvas coordinates.
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return out


def shift_distances(lip: np.ndarray, audio: np.ndarray, max_shift: int = MAX_SHIFT) -> np.ndarray:
    """(windows, 2*max_shift+1): distance between each lip window and the audio window that many windows away."""
    windows = lip.shape[0]
    # Pad the audio with max_shift zero rows on each end, so every shift has a full set of rows.
    # Column ``shift`` then pairs lip window w with audio window w + shift - max_shift.
    padded = np.pad(audio, ((max_shift, max_shift), (0, 0)))
    distances = np.empty((windows, 2 * max_shift + 1), dtype=np.float64)
    for shift in range(2 * max_shift + 1):
        # norm along axis 1 is the Euclidean distance between each pair of embeddings.
        distances[:, shift] = np.linalg.norm(lip - padded[shift : shift + windows], axis=1)
    return distances


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
    # One average distance per shift, over the whole clip.
    mean = shift_distances(lip, audio, max_shift).mean(axis=0)
    # argmin gives the column index; max_shift - best turns it back into a signed offset in frames,
    # where 0 means audio and video are aligned.
    best = int(mean.argmin())
    return float(np.median(mean) - mean[best]), float(mean[best]), max_shift - best


def second_agreement(
    lip: np.ndarray, audio: np.ndarray, block: int = SYNCNET_FPS, tolerance: int = 1, max_shift: int = MAX_SHIFT
) -> Tuple[int, int]:
    """
    ``(blocks within tolerance, blocks scored)``: the D-12 "% of 1-second windows" figure.

    The clip is cut into consecutive blocks of ``block`` windows (one window per frame, so 25 is one
    second); each block gets its own best offset, and counts when that offset is within
    ``tolerance`` frames of zero. A trailing block shorter than half a block is dropped: too few
    windows to say anything. A block is noisier than the whole clip, so this figure is stricter
    than the clip-level offset.
    """
    # Computed once for the whole clip; each block below only averages its own rows.
    distances = shift_distances(lip, audio, max_shift)
    within = scored = 0
    for start in range(0, distances.shape[0], block):
        rows = distances[start : start + block]
        if rows.shape[0] < block // 2:
            continue
        scored += 1
        if abs(max_shift - int(rows.mean(axis=0).argmin())) <= tolerance:
            within += 1
    return within, scored


class SyncNetScorer:
    """Lazy-loading SyncNet with cached failure."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None, device: Optional[str] = None) -> None:
        self.model_path = Path(model_path) if model_path else SYNCNET_MODEL
        # None means "decide at load time": CUDA when PyTorch sees it, otherwise the CPU.
        self.device = device
        self._model = None
        # Cached failure: once loading fails, later calls raise the same message at once.
        self._load_error: Optional[str] = None
        # Lets gpu_utils unload SyncNet when another model needs the GPU (golden rule 5).
        gpu_utils.register_releaser("syncnet", self.release)

    @property
    def available(self) -> bool:
        """True when the weights file is on disk (not a guarantee that it loads)."""
        return self.model_path.is_file()

    def _load(self):
        """Return the loaded network, loading it on first use; raise ``SyncNetUnavailable`` with the fix."""
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
            # weights_only=True refuses pickled code in the file, so loading cannot run arbitrary
            # Python. map_location="cpu" loads to RAM first; .to(device) moves it afterwards.
            state = torch.load(self.model_path, map_location="cpu", weights_only=True)
            model = _build_network()
            model.load_state_dict(state)
            # eval() puts BatchNorm into inference mode, using its stored statistics.
            self._model = model.to(device).eval()
            self.device = device
            logger.info("Loaded SyncNet on %s", device)
            return self._model
        except gpu_utils.InsufficientVRAM:
            # A full GPU can clear later, so it is re-raised without being cached as a load failure.
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
        # Frames available on both sides, minus the window length, since each window needs
        # WINDOW_FRAMES frames starting at its index.
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
        # no_grad: no gradients are needed to score, which saves memory.
        with torch.no_grad():
            # Batches of windows bound the peak memory regardless of clip length.
            for start in range(0, windows, batch_size):
                stop = min(windows, start + batch_size)
                # (window, H, W, 3) -> (3, window, H, W)
                lips = torch.stack(
                    [video[i : i + WINDOW_FRAMES].permute(3, 0, 1, 2) for i in range(start, stop)]
                ).float()
                # The audio window for frame i is the 20 MFCC steps (4 per frame x 5 frames)
                # starting at step 4 * i; unsqueeze(1) adds the single channel axis.
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
        """Drop the network so its GPU memory can be reused; the next score reloads it."""
        if self._model is not None:
            self._model = None
            gpu_utils.empty_cache()


_shared_scorer: Optional[SyncNetScorer] = None


def shared_scorer() -> SyncNetScorer:
    """The process-wide scorer, created on first use on the configured device."""
    global _shared_scorer
    if _shared_scorer is None:
        _shared_scorer = SyncNetScorer(device=gpu_utils.preferred_device())
    return _shared_scorer


def mfcc_features(audio: np.ndarray) -> np.ndarray:
    """13 MFCCs at 100 Hz, shaped (13, steps), from 16 kHz float audio."""
    import python_speech_features

    # Convert float samples in [-1, 1] to the 16-bit integer range of a WAV file; clip first so
    # a sample at exactly 1.0 does not wrap around to -32768.
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    # The library returns (steps, 13); .T transposes to (13, steps), the layout embed() expects.
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

    # A scorer can be passed in (tests pass a fake); otherwise the shared one is used.
    scorer = scorer or shared_scorer()
    # Fail before decoding any video if the weights are missing.
    if not scorer.available:
        scorer._load()  # raises with the fetch command

    info = video_io.probe(video_path)
    if not (info.has_video and info.has_audio):
        raise LipSyncMetricError(f"{Path(video_path).name} needs both a video and an audio stream")

    crop_box: Optional[Tuple[int, int, int, int]] = None
    # Pre-allocate one array for all crops instead of growing a list; +2 frames of slack for rounding.
    expected = max(1, int(round(info.video_duration * SYNCNET_FPS)) + 2)
    crops = np.empty((expected, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    count = 0
    for frame in video_io.read_frames(video_path, fps=SYNCNET_FPS):
        # The crop box is decided once, on the first frame (the simplification in the module docstring).
        if crop_box is None:
            if face_box is None:
                from face_engine import FACE_ENGINE_LOCK, NoFaceDetected, shared_face_engine

                try:
                    # The face engine is shared across threads; its lock serialises analysis.
                    with FACE_ENGINE_LOCK:
                        box = shared_face_engine().analyze(frame).bounding_box
                except NoFaceDetected as err:
                    raise LipSyncMetricError(
                        f"no face in the first frame of {Path(video_path).name}"
                    ) from err
                face_box = (box.x, box.y, box.width, box.height)
            crop_box = syncnet_crop_box(*face_box)
        # INTER_AREA averages pixels when shrinking, which avoids the aliasing of plain sampling.
        crop = cv2.resize(crop_padded(frame, crop_box), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
        if count == len(crops):  # the container under-reported its length
            # Double the buffer, so repeated growth stays cheap.
            crops = np.concatenate([crops, np.empty_like(crops)])
        # video_io yields RGB; ::-1 on the last axis reverses the channel order to BGR.
        crops[count] = crop[:, :, ::-1]  # SyncNet was trained on BGR frames
        count += 1
    if count == 0:
        raise LipSyncMetricError(f"{Path(video_path).name} has no decodable frames")
    # Trim the unused tail of the pre-allocated buffer.
    crops = crops[:count]

    # The audio is read back from the rendered file itself, so the score covers the real muxed result.
    mfcc = mfcc_features(video_io.read_audio(video_path, SYNCNET_SAMPLE_RATE))
    lip, audio = scorer.embed(crops, mfcc)
    lse_c, lse_d, offset = sync_scores(lip, audio)
    within, scored = second_agreement(lip, audio)

    warnings: List[str] = []
    # More than 2 frames (80 ms) off is flagged; the score is still returned so it can be inspected.
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
        seconds_within_one=within,
        seconds_scored=scored,
    )
