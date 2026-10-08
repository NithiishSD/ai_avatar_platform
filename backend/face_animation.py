"""
Frame timeline: phoneme timestamps -> per-frame blendshape weights (task G2-03).

The contract hands the renderer a list of ``(viseme, startMs, endMs)`` and a
target frame rate. Sampling that list once per frame and snapping the mouth to
whichever viseme is active gives a face that jitters between poses: real lips
are never still and never teleport. This module turns the step function into
motion in four passes:

1. **Hold and rasterise.** Each phoneme is held until the next begins (CTC
   alignment spans are far shorter than the sound), then the visemes are
   drawn onto a fine 5 ms grid, one column per blendshape.
2. **Smooth** each column with a Gaussian, which is a cheap, symmetric model
   of coarticulation -- the mouth starts moving toward a sound before it
   arrives and leaves it late.
3. **Re-assert closures.** Smoothing a 60 ms ``p`` or ``b`` between two open
   vowels would leave the lips apart, and a bilabial that never closes is the
   single most visible lip-sync error. Closures are gated back in afterwards.
4. **Sample** at frame centres, then layer the emotion expression and blinks.

An optional speech-energy envelope scales the jaw with loudness and shuts the
mouth during real silence. That also bounds the damage when the aligner had to
fall back to its acoustic guess: the timing of *which* viseme may be off, but
the mouth still opens only while there is sound.

Terms, explained once:

* A **blendshape track** is a 2-D array: one row per video frame, one column
  per blendshape (``jawOpen``, ``mouthPucker``...), each cell a weight in
  [0, 1]. ``viseme_blendshapes.py`` says which weights each viseme sets.
* **Gaussian smoothing** replaces each value with a weighted average of its
  neighbours, the weights following a bell curve of width ``sigma``. Larger
  sigma means a softer, slower mouth; it never overshoots the input range.
* **Energy gating** multiplies the mouth shapes by a 0..1 loudness signal, so
  a shape can only show while the audio actually has sound in it.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from viseme_blendshapes import (
    CLOSURE_VISEMES,
    OPENING_SHAPES,
    VISEME_BLENDSHAPES,
    emotion_weights,
    resolve_viseme,
)

logger = logging.getLogger(__name__)

# Every curve is built on a 5 ms grid, finer than any frame rate in use (25 fps
# is 40 ms a frame), so a short phoneme still owns a few grid cells.
GRID_MS = 5.0
# Smoothing widths in milliseconds, turned into grid steps by dividing by
# GRID_MS. Closures get a narrower kernel so a short "p" is not blurred away.
DEFAULT_SMOOTHING_MS = 38.0
CLOSURE_SMOOTHING_MS = 14.0

# The aligner's CTC spans mark where a phoneme is *emitted*, typically 20-60 ms,
# with gaps before the next one. Read literally, the mouth would return to
# rest between every pair of sounds and barely open at all. A phoneme is held
# until the next one starts, up to this long; a longer gap is a real pause and
# the aligner reports those as explicit silence anyway.
MAX_HOLD_MS = 240.0

# ``eyeblinkRate`` in the contract has no unit. The audio side emits 1.0 for a
# neutral voice and up to 2.4 for anger, which as blinks per second would be a
# flutter. It is read here as a multiplier on the resting human rate of about
# 17 blinks a minute, which makes 1.0 look natural and 2.4 look agitated.
BASE_BLINKS_PER_SECOND = 17.0 / 60.0
# Blink shape: the lid closes in 80 ms and reopens in 150 ms, the asymmetry of a
# real blink. Two blinks are never closer together than MIN_BLINK_INTERVAL_S.
MIN_BLINK_INTERVAL_S = 0.45
BLINK_CLOSE_S = 0.08
BLINK_OPEN_S = 0.15

# Below this share of the clip's loud level the speaker is treated as silent.
SILENCE_LEVEL = 0.06

BLINK_SHAPES = ("eyeBlinkLeft", "eyeBlinkRight")

# Timestamps arrive either as contract objects (attributes) or as plain JSON
# dicts (keys); everything below accepts both.
TimestampLike = Union[Mapping[str, object], object]


def frame_count_for(duration_seconds: float, fps: int) -> int:
    """
    Frames needed to cover ``duration_seconds`` at ``fps`` (at least one).

    Rounded up, so the video is never shorter than the audio it carries; the
    epsilon keeps an exact product such as 2.0 s x 25 from becoming 51.
    """
    return max(1, int(math.ceil(float(duration_seconds) * int(fps) - 1e-6)))


def _field(item: TimestampLike, snake: str, camel: str) -> Any:
    """Read one field from a dict (camelCase wire name first) or an object (snake_case)."""
    if isinstance(item, Mapping):
        return item.get(camel, item.get(snake))
    return getattr(item, snake)


def normalise_timestamps(
    timestamps: Iterable[TimestampLike],
) -> List[Tuple[str, float, float]]:
    """``(viseme, start_ms, end_ms)`` from contract objects or plain dicts."""
    out: List[Tuple[str, float, float]] = []
    for item in timestamps:
        viseme = _field(item, "viseme", "viseme")
        start = float(_field(item, "start_ms", "startMs"))
        end = float(_field(item, "end_ms", "endMs"))
        # Zero- or negative-length spans carry no timing and are dropped.
        if end > start:
            out.append((str(viseme), start, end))
    return out


def hold_until_next(
    segments: Sequence[Tuple[str, float, float]],
    duration_ms: float,
    max_hold_ms: float = MAX_HOLD_MS,
) -> List[Tuple[str, float, float]]:
    """
    Extend each segment to the start of the next one (at most ``max_hold_ms``).

    Segments that already touch are unchanged, so a densely labelled payload
    passes through as-is.
    """
    held: List[Tuple[str, float, float]] = []
    for position, (viseme, start, end) in enumerate(segments):
        # The last segment is held toward the end of the clip instead of a next start.
        limit = segments[position + 1][1] if position + 1 < len(segments) else duration_ms
        if limit > end:
            end = min(limit, end + max_hold_ms)
        held.append((viseme, start, end))
    return held


@dataclass
class AnimationTrack:
    """Per-frame blendshape weights for one render."""

    fps: int
    names: Tuple[str, ...]
    weights: np.ndarray  # (frames, len(names)) float32, each in [0, 1]
    unknown_visemes: Dict[str, int] = field(default_factory=dict)
    blink_times: List[float] = field(default_factory=list)
    energy_gated: bool = False

    @property
    def frame_count(self) -> int:
        """Number of video frames in the track (rows of ``weights``)."""
        return int(self.weights.shape[0])

    def frame(self, index: int) -> Dict[str, float]:
        """Non-zero weights of one frame, keyed by blendshape name."""
        row = self.weights[index]
        return {
            name: float(value)
            for name, value in zip(self.names, row, strict=True)
            # Near-zero weights are dropped so callers and logs see only active shapes.
            if value > 1e-4
        }

    def column(self, name: str) -> np.ndarray:
        """One blendshape's weight across all frames (zeros if unused)."""
        if name not in self.names:
            return np.zeros(self.frame_count, dtype=np.float32)
        return self.weights[:, self.names.index(name)]


def speech_energy_envelope(
    audio_path: Union[str, Path], grid_ms: float = GRID_MS
) -> np.ndarray:
    """
    Loudness of a clip on the animation grid, normalised to [0, 1].

    RMS per grid step, divided by the clip's 95th percentile so one loud
    plosive does not flatten the rest of the envelope.
    """
    import soundfile as sf

    # always_2d gives (samples, channels) even for mono, so mean(axis=1) down-mixes
    # any file to one channel the same way.
    data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    return energy_envelope(data.mean(axis=1), sample_rate, grid_ms)[0]


def energy_envelope(
    mono: np.ndarray, sample_rate: int, grid_ms: float = GRID_MS, reference: Optional[float] = None
) -> Tuple[np.ndarray, Optional[float]]:
    """
    ``(envelope, reference)``: RMS per grid step divided by ``reference``.

    With no ``reference`` it is the 95th percentile of the voiced steps (a whole clip). A live
    stream passes the loudest level heard so far instead, so a quiet chunk stays quiet rather than
    being stretched to full loudness; the reference used is returned for the next chunk.
    """
    # hop: audio samples per grid step (5 ms at 24 kHz is 120 samples).
    hop = max(1, int(round(sample_rate * grid_ms / 1000.0)))
    steps = int(math.ceil(len(mono) / hop))
    if steps == 0:
        return np.zeros(0, dtype=np.float32), reference
    # Pad with silence to a whole number of steps so reshape(steps, hop) works.
    padded = np.zeros(steps * hop, dtype=np.float32)
    padded[: len(mono)] = mono
    # RMS (root mean square) is the standard loudness measure of a block of samples.
    rms = np.sqrt((padded.reshape(steps, hop) ** 2).mean(axis=1))
    # Percentile over the voiced steps only: in a clip that is mostly pauses
    # the plain 95th percentile is itself silence, which would normalise
    # every step to zero and shut the mouth for the whole clip.
    voiced = rms[rms > 1e-4]
    if reference is None:
        if voiced.size == 0:
            return np.zeros(0, dtype=np.float32), None  # truly silent: nothing to gate on
        reference = float(np.percentile(voiced, 95))
    return np.clip(rms / reference, 0.0, 1.0).astype(np.float32), reference


def _gaussian(array: np.ndarray, sigma_steps: float) -> np.ndarray:
    """
    Gaussian-smooth ``array`` along time (axis 0), column by column.

    ``mode="nearest"`` repeats the edge values past the ends, so the first and last
    frames are not pulled toward zero. Imported lazily to keep scipy off the import
    path of modules that never animate.
    """
    if sigma_steps <= 0 or array.shape[0] < 2:
        return array
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(array, sigma=sigma_steps, axis=0, mode="nearest")


def blink_schedule(
    duration_seconds: float, eyeblink_rate: float, seed: int
) -> List[float]:
    """
    Blink start times in seconds.

    Deterministic for a given seed so re-rendering a job reproduces it
    frame for frame, but jittered so the blinks do not tick like a metronome.
    """
    rate = max(0.0, float(eyeblink_rate)) * BASE_BLINKS_PER_SECOND
    if rate <= 0 or duration_seconds <= 0:
        return []
    interval = max(MIN_BLINK_INTERVAL_S, 1.0 / rate)
    # np.random.default_rng(seed) is a private generator: the same seed gives the
    # same sequence, and it does not disturb any global random state.
    rng = np.random.default_rng(seed)
    times: List[float] = []
    # The first blink lands 35-90 % of an interval in, not at time zero.
    cursor = interval * float(rng.uniform(0.35, 0.9))
    blink_length = BLINK_CLOSE_S + BLINK_OPEN_S
    while cursor + blink_length <= duration_seconds:
        times.append(round(cursor, 4))
        cursor += max(MIN_BLINK_INTERVAL_S, interval * float(rng.uniform(0.6, 1.4)))
    return times


def _blink_curve(frame_times: np.ndarray, starts: Sequence[float]) -> np.ndarray:
    """Eyelid closure in [0, 1] per frame: fast down, slower up."""
    curve = np.zeros_like(frame_times, dtype=np.float32)
    for start in starts:
        local = frame_times - start
        closing = (local >= 0) & (local < BLINK_CLOSE_S)
        opening = (local >= BLINK_CLOSE_S) & (local < BLINK_CLOSE_S + BLINK_OPEN_S)
        # sin^2 / cos^2 ramps rise and fall smoothly (zero slope at both ends), so the
        # lid accelerates and decelerates instead of snapping. np.maximum lets two
        # overlapping blinks combine without exceeding fully closed.
        curve[closing] = np.maximum(
            curve[closing],
            np.sin(0.5 * np.pi * local[closing] / BLINK_CLOSE_S) ** 2,
        )
        curve[opening] = np.maximum(
            curve[opening],
            np.cos(0.5 * np.pi * (local[opening] - BLINK_CLOSE_S) / BLINK_OPEN_S) ** 2,
        )
    return curve


def seed_from_job_id(job_id: str) -> int:
    """A stable 32-bit seed for a job (Python's ``hash`` is salted per run)."""
    # The first 4 bytes of a SHA-256 digest give the same number on every run and
    # every machine, which a salted ``hash()`` would not.
    return int.from_bytes(hashlib.sha256(job_id.encode("utf-8")).digest()[:4], "big")


def build_animation(
    phoneme_timestamps: Iterable[TimestampLike],
    duration_seconds: float,
    fps: int,
    emotion_vector: Optional[Mapping[str, Any]] = None,
    seed: int = 0,
    energy_envelope: Optional[np.ndarray] = None,
    smoothing_ms: float = DEFAULT_SMOOTHING_MS,
) -> AnimationTrack:
    """
    Build the per-frame blendshape track for a render job.

    ``frame_count == ceil(duration_seconds * fps)``; frame ``i`` is sampled at
    its centre, ``(i + 0.5) / fps``, so the first and last frames are not
    biased toward the clip edges.
    """
    fps = int(fps)
    if fps <= 0:
        raise ValueError("fps must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    segments = hold_until_next(
        normalise_timestamps(phoneme_timestamps), duration_seconds * 1000.0
    )
    emotion = emotion_weights(emotion_vector)

    # Column order is sorted names, so the same job always yields the same layout.
    names = sorted(
        {shape for shapes in VISEME_BLENDSHAPES.values() for shape in shapes}
        | set(emotion)
        | set(BLINK_SHAPES)
    )
    index = {name: i for i, name in enumerate(names)}

    # Pass 1: rasterise. ``closure`` is a separate 1-D gate track, 0 = no forced
    # closure, 1 = lips must be shut.
    steps = max(1, int(math.ceil(duration_seconds * 1000.0 / GRID_MS)))
    grid = np.zeros((steps, len(names)), dtype=np.float32)
    closure = np.zeros(steps, dtype=np.float32)
    unknown: Dict[str, int] = {}

    for viseme, start_ms, end_ms in segments:
        canonical, known = resolve_viseme(viseme)
        if not known:
            unknown[viseme] = unknown.get(viseme, 0) + 1
        # a..b is the segment's span in grid cells, clamped to the clip and at least one
        # cell wide so a very short phoneme still appears.
        a = int(min(steps, max(0, math.floor(start_ms / GRID_MS))))
        b = int(min(steps, max(a + 1, math.ceil(end_ms / GRID_MS))))
        grid[a:b, :] = 0.0
        # The later segment owns the overlap, including its closure gate.
        closure[a:b] = 0.0
        for shape, weight in VISEME_BLENDSHAPES[canonical].items():
            grid[a:b, index[shape]] = weight
        strength = CLOSURE_VISEMES.get(canonical)
        if strength:
            # Gate the middle of the segment: its edges belong to the
            # transition in and out of the closure.
            inset = max(0, (b - a) // 5)
            closure[a + inset : max(a + inset + 1, b - inset)] = strength

    if unknown:
        logger.warning(
            "Render payload uses viseme names outside the 15-viseme contract; "
            "they are rendered as the rest pose: %s",
            ", ".join(f"{name} x{count}" for name, count in sorted(unknown.items())),
        )

    # Pass 2: smooth. sigma is expressed in grid steps, hence the division.
    smoothed = _gaussian(grid, smoothing_ms / GRID_MS)
    # Pass 3: re-assert closures. Where the gate is 1, every opening shape is
    # multiplied by 0, shutting the lips even after smoothing blurred them open.
    closure = _gaussian(closure[:, None], CLOSURE_SMOOTHING_MS / GRID_MS)[:, 0]
    opening_columns = [index[name] for name in OPENING_SHAPES if name in index]
    smoothed[:, opening_columns] *= (1.0 - np.clip(closure, 0.0, 1.0))[:, None]

    energy_gated = False
    if energy_envelope is not None and len(energy_envelope) > 0:
        # Optional energy gating. The envelope is padded to the grid length and lightly
        # smoothed (30 ms) so the gate does not flicker on single loud samples.
        envelope = np.asarray(energy_envelope, dtype=np.float32)
        if len(envelope) < steps:
            envelope = np.pad(envelope, (0, steps - len(envelope)))
        envelope = _gaussian(envelope[:steps, None], 30.0 / GRID_MS)[:, 0]
        # Speaking: jaw follows loudness. Silent: every mouth shape eases out.
        # 0 below SILENCE_LEVEL, ramping to 1 at twice that level.
        voiced = np.clip((envelope - SILENCE_LEVEL) / SILENCE_LEVEL, 0.0, 1.0)
        mouth_columns = [
            i for name, i in index.items() if name.startswith(("jaw", "mouth", "tongue"))
        ]
        smoothed[:, mouth_columns] *= voiced[:, None]
        # The jaw keeps 60 % of its viseme opening at low volume and gains the last
        # 40 % with loudness, so quiet speech still moves the mouth.
        smoothed[:, index["jawOpen"]] *= 0.6 + 0.4 * np.clip(envelope, 0.0, 1.0)
        energy_gated = True

    # Pass 4: sample each frame at its centre time and look up the grid cell.
    frames = frame_count_for(duration_seconds, fps)
    frame_times = (np.arange(frames, dtype=np.float64) + 0.5) / fps
    sample = np.clip((frame_times * 1000.0 / GRID_MS).astype(np.int64), 0, steps - 1)
    weights = smoothed[sample].astype(np.float32)

    # Emotion is a constant offset added on top of the speech motion.
    for shape, weight in emotion.items():
        weights[:, index[shape]] += weight

    blink_rate = 1.0
    if emotion_vector:
        raw = emotion_vector.get("eyeblinkRate", emotion_vector.get("eyeblink_rate"))
        if raw is not None:
            try:
                blink_rate = float(raw)
            except (TypeError, ValueError):
                blink_rate = 1.0
    blinks = blink_schedule(duration_seconds, blink_rate, seed)
    if blinks:
        curve = _blink_curve(frame_times, blinks)
        for shape in BLINK_SHAPES:
            # Blinks take the larger of the existing eye weight and the blink curve.
            weights[:, index[shape]] = np.maximum(weights[:, index[shape]], curve)

    np.clip(weights, 0.0, 1.0, out=weights)
    return AnimationTrack(
        fps=fps,
        names=tuple(names),
        weights=weights,
        unknown_visemes=unknown,
        blink_times=blinks,
        energy_gated=energy_gated,
    )
