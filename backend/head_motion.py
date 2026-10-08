"""
Whole-portrait motion: the head and shoulders move a little while the avatar talks (I-01).

The owner's verdict on the first renders: only the lips move, so the photo looks pasted. A real
speaker is never still: the head drifts, tilts and nods on stressed syllables, the brows lift for
emphasis, the chest rises and falls. This module produces that motion as numbers per frame; the
portrait warp (``face_warp.PortraitAnimator.apply_pose``) turns them into pixels.

Per frame it returns five channels (``POSE_CHANNELS``):

    roll    head tilt in degrees (positive = clockwise on screen)
    dx, dy  head shift in fractions of the face height (positive = right / down)
    scale   head size change, a fraction (positive = a slight lean toward the camera)
    breath  0..1, the phase of a slow breath that lifts the shoulders

and a ``brow`` emphasis track (0..1) that the animation adds to the brow shapes.

How it is built:
- **Drift:** each channel is a sum of three slow sine waves with seeded frequencies (0.08-0.45 Hz)
  and phases, so the motion never repeats visibly and the same job always moves the same way.
- **Emphasis:** where the speech gets louder quickly (an onset in the loudness envelope), the head
  dips in a small nod and the brows lift, both shaped by a short smoothing kernel so they ease in
  and out instead of jerking.
- **Rest:** while silent the drift is halved, so a pause looks like listening, not freezing.
Amplitudes are small on purpose (about 2 degrees, 1-2 % of the face height): bigger moves expose
the 2-D warp's limits (the background behind the head has to stretch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

POSE_CHANNELS = ("roll", "dx", "dy", "scale", "breath")

ROLL_DEG = 1.8        # largest drift tilt
SHIFT_X = 0.012       # largest sideways drift, in face heights
SHIFT_Y = 0.008       # largest vertical drift, in face heights
NOD_DY = 0.018        # extra downward dip on an emphasis
NOD_ROLL = 0.6        # a little tilt with the nod, degrees
LEAN = 0.006          # loud speech leans in this much (scale)
BREATH_PERIOD_S = 4.2
IDLE_FACTOR = 0.5     # drift while silent, relative to speaking


@dataclass
class Motion:
    pose: np.ndarray   # (frames, 5) in POSE_CHANNELS order
    brow: np.ndarray   # (frames,) emphasis 0..1


def _drift(times: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A smooth, non-repeating wobble in [-1, 1]: three slow sines with random frequencies and phases."""
    total = np.zeros_like(times)
    for weight in (0.55, 0.3, 0.15):
        frequency = rng.uniform(0.08, 0.45)
        total += weight * np.sin(2 * np.pi * frequency * times + rng.uniform(0, 2 * np.pi))
    return total


def _smooth(signal: np.ndarray, fps: int, seconds: float) -> np.ndarray:
    """Gaussian smoothing over about ``seconds``, so impulses become gentle bumps."""
    sigma = max(1.0, seconds * fps)
    radius = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(signal, radius, mode="edge"), kernel, mode="valid")


def emphasis(loudness: np.ndarray, fps: int) -> np.ndarray:
    """
    0..1 bumps where the speech gets louder quickly (stressed syllables, the start of a phrase).

    The rise of the smoothed loudness is kept where it is in the top quarter of all rises; each
    such moment becomes a bump about a quarter of a second wide.
    """
    if loudness.size < 3 or float(loudness.max()) <= 0:
        return np.zeros_like(loudness)
    rise = np.clip(np.diff(_smooth(loudness, fps, 0.04), prepend=loudness[0]), 0, None)
    positive = rise[rise > 0]
    if positive.size == 0:
        return np.zeros_like(loudness)
    threshold = float(np.percentile(positive, 75))
    peaks = np.where(rise >= threshold, rise, 0.0)
    bumps = _smooth(peaks, fps, 0.12)
    return np.clip(bumps / (bumps.max() + 1e-9), 0.0, 1.0)


def motion_track(frame_count: int, fps: int, seed: int, loudness: Optional[np.ndarray] = None,
                 intensity: float = 1.0) -> Motion:
    """
    Head, shoulder and brow motion for ``frame_count`` frames at ``fps``.

    ``loudness`` (0..1 per frame) drives the nods, brow lifts and the speaking/idle balance; without it
    the motion is drift and breathing only. ``intensity`` scales everything (0 = a still photo).
    """
    times = (np.arange(frame_count) + 0.5) / fps
    rng = np.random.default_rng(seed)
    level = np.zeros(frame_count) if loudness is None else np.clip(np.asarray(loudness, dtype=np.float64)[:frame_count], 0, 1)
    if level.size < frame_count:
        level = np.pad(level, (0, frame_count - level.size))
    speaking = _smooth((level > 0.15).astype(np.float64), fps, 0.4)   # 0 silent .. 1 talking, eased
    activity = IDLE_FACTOR + (1 - IDLE_FACTOR) * speaking
    bumps = emphasis(level, fps)

    pose = np.zeros((frame_count, len(POSE_CHANNELS)))
    pose[:, 0] = activity * ROLL_DEG * _drift(times, rng) + NOD_ROLL * bumps * rng.choice([-1, 1])
    pose[:, 1] = activity * SHIFT_X * _drift(times, rng)
    pose[:, 2] = activity * SHIFT_Y * _drift(times, rng) + NOD_DY * bumps
    pose[:, 3] = LEAN * _smooth(level, fps, 0.3)
    pose[:, 4] = 0.5 * (1 - np.cos(2 * np.pi * times / BREATH_PERIOD_S + rng.uniform(0, 2 * np.pi)))
    pose[:, :4] *= intensity
    pose[:, 4] *= min(1.0, intensity)
    return Motion(pose=pose.astype(np.float32), brow=(0.45 * bumps * intensity).astype(np.float32))


def demo() -> None:
    """Self-check: bounded, deterministic per seed, still at intensity 0, nods follow loud onsets."""
    loud = np.zeros(250)
    loud[100:140] = 1.0  # one loud phrase starting at frame 100
    a = motion_track(250, 25, seed=7, loudness=loud)
    b = motion_track(250, 25, seed=7, loudness=loud)
    assert np.array_equal(a.pose, b.pose)
    assert np.abs(a.pose[:, 0]).max() <= ROLL_DEG + NOD_ROLL + 1e-6
    assert int(np.argmax(a.brow)) in range(95, 115)
    still = motion_track(250, 25, seed=7, loudness=loud, intensity=0.0)
    assert np.abs(still.pose[:, :4]).max() == 0 and still.brow.max() == 0


if __name__ == "__main__":
    demo()
    print("head_motion self-check passed")
