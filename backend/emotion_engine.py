"""
Emotion prosody engine (Phase 3).

Roadmap Phase 3 asks Developer 1 for "emotion prosody vectors (joy, anger,
sorrow, authority)". None of the four TTS backends in this project exposes an
emotion conditioning input, so emotion is applied as a deterministic prosody
transform on the synthesized waveform:

    pitch shift  - mean F0 offset in semitones
    rate         - speaking rate (time stretch, pitch preserving)
    energy       - broadband gain
    contour      - slope of the F0 track across the utterance
    tilt         - spectral tilt, i.e. how bright or dark the voice sounds
    jitter       - micro-variation in the F0 track

A request carries a *vector*, not a single label, so emotions blend: the
parameters of each preset are averaged weighted by that preset's activation.
The same vector is also projected onto the frozen ``EmotionVector`` contract
so Developer 2's renderer receives matching facial-expression weights.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProsodyParams:
    """Prosody transform parameters. All are multiplicative or additive offsets."""

    pitch_semitones: float = 0.0   # additive, semitones
    rate: float = 1.0              # multiplicative, >1 is faster
    energy: float = 1.0            # multiplicative gain
    contour: float = 0.0           # semitones drift start -> end
    tilt: float = 0.0              # >0 brighter, <0 darker
    jitter: float = 0.0            # semitones of micro-variation

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EmotionPreset:
    """One named emotion and how it maps onto prosody and a face."""

    name: str
    label: str
    description: str
    prosody: ProsodyParams
    # Projection onto the frozen AvatarRenderJob EmotionVector contract.
    happy_weight: float
    eyeblink_rate: float


NEUTRAL = EmotionPreset(
    name="neutral",
    label="Neutral",
    description="Flat, informational delivery. The identity transform.",
    prosody=ProsodyParams(),
    happy_weight=0.0,
    eyeblink_rate=1.0,
)

# Prosody directions follow the classic activation/valence findings for
# emotional speech: high-arousal states raise F0, rate and energy; sorrow
# lowers all three and adds a falling contour; authority lowers F0 while
# keeping energy high and the contour flat.
PRESETS: Dict[str, EmotionPreset] = {
    "neutral": NEUTRAL,
    "joy": EmotionPreset(
        name="joy",
        label="Joy",
        description="Bright, energetic, rising contour.",
        prosody=ProsodyParams(
            pitch_semitones=2.2, rate=1.10, energy=1.15,
            contour=1.2, tilt=0.35, jitter=0.25,
        ),
        happy_weight=1.0,
        eyeblink_rate=1.6,
    ),
    "anger": EmotionPreset(
        name="anger",
        label="Anger",
        description="Loud, fast, harsh and tense.",
        prosody=ProsodyParams(
            pitch_semitones=1.6, rate=1.14, energy=1.35,
            contour=-0.6, tilt=0.55, jitter=0.55,
        ),
        happy_weight=0.0,
        eyeblink_rate=2.4,
    ),
    "sorrow": EmotionPreset(
        name="sorrow",
        label="Sorrow",
        description="Low, slow, dark, with a falling contour.",
        prosody=ProsodyParams(
            pitch_semitones=-2.0, rate=0.86, energy=0.82,
            contour=-1.6, tilt=-0.45, jitter=0.15,
        ),
        happy_weight=0.0,
        eyeblink_rate=0.6,
    ),
    "authority": EmotionPreset(
        name="authority",
        label="Authority",
        description="Deliberate, deep, level and controlled.",
        prosody=ProsodyParams(
            pitch_semitones=-1.4, rate=0.93, energy=1.12,
            contour=-0.3, tilt=-0.15, jitter=0.05,
        ),
        happy_weight=0.1,
        eyeblink_rate=0.8,
    ),
    "calm": EmotionPreset(
        name="calm",
        label="Calm",
        description="Soft, unhurried, gently falling.",
        prosody=ProsodyParams(
            pitch_semitones=-0.6, rate=0.92, energy=0.92,
            contour=-0.5, tilt=-0.25, jitter=0.05,
        ),
        happy_weight=0.35,
        eyeblink_rate=0.9,
    ),
    "excitement": EmotionPreset(
        name="excitement",
        label="Excitement",
        description="Fast and loud with a strongly rising contour.",
        prosody=ProsodyParams(
            pitch_semitones=3.0, rate=1.18, energy=1.28,
            contour=1.8, tilt=0.5, jitter=0.4,
        ),
        happy_weight=0.9,
        eyeblink_rate=2.0,
    ),
}

EMOTION_NAMES = tuple(PRESETS)

# Clamps keep a blended vector inside what librosa can transform without
# audible artefacts.
_PITCH_LIMIT = 4.0
_RATE_LIMITS = (0.7, 1.35)
_ENERGY_LIMITS = (0.6, 1.6)


@dataclass(frozen=True)
class EmotionApplication:
    """What the engine actually did to a waveform."""

    vector: Dict[str, float]
    dominant: str
    intensity: float
    prosody: ProsodyParams
    applied: bool
    duration_seconds: float
    sample_rate: int
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "vector": self.vector,
            "dominant": self.dominant,
            "intensity": round(self.intensity, 4),
            "prosody": self.prosody.to_dict(),
            "applied": self.applied,
            "durationSeconds": round(self.duration_seconds, 4),
            "sampleRate": self.sample_rate,
            "note": self.note,
        }


def normalize_vector(
    vector: Optional[Mapping[str, float]] = None,
    emotion: Optional[str] = None,
    intensity: float = 1.0,
) -> Dict[str, float]:
    """
    Build a clean emotion vector from either a named emotion or a raw mapping.

    Unknown keys are dropped. Negative weights are clamped to zero. The result
    is normalized so the weights sum to at most 1.0, with any remainder held by
    ``neutral`` — so a half-strength joy is literally half joy, half neutral.
    """
    raw: Dict[str, float] = {}
    if vector:
        for key, value in vector.items():
            name = str(key).strip().lower()
            if name in PRESETS:
                raw[name] = max(0.0, float(value))
    elif emotion:
        name = str(emotion).strip().lower()
        if name in PRESETS and name != "neutral":
            raw[name] = max(0.0, min(1.0, float(intensity)))

    raw.pop("neutral", None)
    total = sum(raw.values())
    if total <= 0:
        return {"neutral": 1.0}
    if total > 1.0:
        raw = {k: v / total for k, v in raw.items()}
        total = 1.0
    raw["neutral"] = round(max(0.0, 1.0 - total), 6)
    return {k: round(v, 6) for k, v in raw.items() if v > 0}


def blend_prosody(vector: Mapping[str, float]) -> ProsodyParams:
    """Weighted blend of preset prosody parameters, clamped to safe ranges."""
    pitch = rate = energy = contour = tilt = jitter = 0.0
    weight_sum = 0.0
    for name, weight in vector.items():
        preset = PRESETS.get(name)
        if preset is None or weight <= 0:
            continue
        p = preset.prosody
        pitch += p.pitch_semitones * weight
        rate += p.rate * weight
        energy += p.energy * weight
        contour += p.contour * weight
        tilt += p.tilt * weight
        jitter += p.jitter * weight
        weight_sum += weight

    if weight_sum <= 0:
        return ProsodyParams()

    # rate and energy are multiplicative, so an incomplete weight sum must fall
    # back to 1.0 (identity) rather than 0.0.
    rate += 1.0 * (1.0 - weight_sum)
    energy += 1.0 * (1.0 - weight_sum)

    return ProsodyParams(
        pitch_semitones=round(max(-_PITCH_LIMIT, min(_PITCH_LIMIT, pitch)), 4),
        rate=round(max(_RATE_LIMITS[0], min(_RATE_LIMITS[1], rate)), 4),
        energy=round(max(_ENERGY_LIMITS[0], min(_ENERGY_LIMITS[1], energy)), 4),
        contour=round(max(-3.0, min(3.0, contour)), 4),
        tilt=round(max(-1.0, min(1.0, tilt)), 4),
        jitter=round(max(0.0, min(1.0, jitter)), 4),
    )


def dominant_emotion(vector: Mapping[str, float]) -> tuple[str, float]:
    """Strongest non-neutral emotion and its weight, or ('neutral', 0.0)."""
    best_name, best_weight = "neutral", 0.0
    for name, weight in vector.items():
        if name == "neutral":
            continue
        if weight > best_weight:
            best_name, best_weight = name, float(weight)
    return best_name, best_weight


def to_render_emotion_vector(vector: Mapping[str, float]) -> Dict[str, float]:
    """
    Project an emotion vector onto the frozen AvatarRenderJob ``EmotionVector``.

    Developer 2's renderer reads ``happy`` / ``neutral`` / ``eyeblinkRate``, so
    those three are always present and always valid; the per-emotion weights
    ride along in the optional fields added in Phase 3.
    """
    happy = 0.0
    blink = 0.0
    weight_sum = 0.0
    for name, weight in vector.items():
        preset = PRESETS.get(name)
        if preset is None or weight <= 0:
            continue
        happy += preset.happy_weight * weight
        blink += preset.eyeblink_rate * weight
        weight_sum += weight
    if weight_sum <= 0:
        happy, blink = 0.0, NEUTRAL.eyeblink_rate
    else:
        blink += NEUTRAL.eyeblink_rate * max(0.0, 1.0 - weight_sum)

    happy = max(0.0, min(1.0, happy))
    payload: Dict[str, float] = {
        "happy": round(happy, 4),
        "neutral": round(max(0.0, min(1.0, 1.0 - happy)), 4),
        "eyeblinkRate": round(max(0.0, min(10.0, blink)), 4),
    }
    for name in ("joy", "anger", "sorrow", "authority", "calm", "excitement"):
        weight = float(vector.get(name, 0.0))
        if weight > 0:
            payload[name] = round(weight, 4)
    return payload


class EmotionProsodyEngine:
    """Applies a blended emotion vector to a synthesized waveform, in place."""

    def __init__(self, sample_rate_hint: int = 24000):
        self.sample_rate_hint = sample_rate_hint

    def apply_to_file(
        self,
        audio_path: Path | str,
        vector: Optional[Mapping[str, float]] = None,
        emotion: Optional[str] = None,
        intensity: float = 1.0,
        extra_rate: float = 1.0,
        extra_semitones: float = 0.0,
    ) -> EmotionApplication:
        """
        Rewrite ``audio_path`` with the emotion transform applied.

        ``extra_rate`` and ``extra_semitones`` fold the caller's explicit speed
        and pitch controls into the same transform. Time-stretching and
        pitch-shifting twice would compound artefacts, so emotion and manual
        prosody are always blended first and applied once.

        A neutral vector with no manual prosody is a no-op: the file is left
        untouched and ``applied=False`` is returned.
        """
        path = Path(audio_path)
        normalized = normalize_vector(vector=vector, emotion=emotion, intensity=intensity)
        params = blend_prosody(normalized)
        if abs(extra_rate - 1.0) > 0.001 or abs(extra_semitones) > 0.001:
            params = ProsodyParams(
                pitch_semitones=params.pitch_semitones + float(extra_semitones),
                rate=params.rate * float(extra_rate),
                energy=params.energy,
                contour=params.contour,
                tilt=params.tilt,
                jitter=params.jitter,
            )
        name, weight = dominant_emotion(normalized)

        info = sf.info(str(path))
        sample_rate = int(info.samplerate)
        duration = float(info.duration)

        if self._is_identity(params):
            return EmotionApplication(
                vector=normalized,
                dominant=name,
                intensity=weight,
                prosody=params,
                applied=False,
                duration_seconds=duration,
                sample_rate=sample_rate,
                note="neutral vector - waveform unchanged",
            )

        try:
            audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            transformed = self.apply_to_array(audio, sample_rate, params)
            sf.write(str(path), transformed, sample_rate)
            duration = len(transformed) / sample_rate if sample_rate else duration
            note = ""
        except Exception as exc:  # noqa: BLE001 - degrade to unmodified audio
            logger.warning("Emotion prosody transform failed: %s", exc)
            return EmotionApplication(
                vector=normalized,
                dominant=name,
                intensity=weight,
                prosody=params,
                applied=False,
                duration_seconds=duration,
                sample_rate=sample_rate,
                note=f"transform failed, original audio kept: {exc}",
            )

        return EmotionApplication(
            vector=normalized,
            dominant=name,
            intensity=weight,
            prosody=params,
            applied=True,
            duration_seconds=duration,
            sample_rate=sample_rate,
            note=note,
        )

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    @staticmethod
    def _is_identity(params: ProsodyParams) -> bool:
        return (
            abs(params.pitch_semitones) < 0.05
            and abs(params.rate - 1.0) < 0.01
            and abs(params.energy - 1.0) < 0.01
            and abs(params.contour) < 0.05
            and abs(params.tilt) < 0.02
            and params.jitter < 0.02
        )

    def apply_to_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        params: ProsodyParams,
    ) -> np.ndarray:
        """Apply a prosody transform to a mono float32 array."""
        import librosa

        signal = np.asarray(audio, dtype=np.float32)
        if signal.size == 0:
            return signal

        source_rms = float(np.sqrt(np.mean(signal**2)))

        if abs(params.rate - 1.0) >= 0.01:
            signal = librosa.effects.time_stretch(signal, rate=float(params.rate))

        if abs(params.pitch_semitones) >= 0.05:
            signal = librosa.effects.pitch_shift(
                signal, sr=sample_rate, n_steps=float(params.pitch_semitones)
            )

        if abs(params.contour) >= 0.05 or params.jitter >= 0.02:
            signal = self._apply_contour(signal, sample_rate, params)

        if abs(params.tilt) >= 0.02:
            signal = self._apply_tilt(signal, float(params.tilt))

        # Phase-vocoder stretching, pitch shifting and tilt filtering all change
        # the level as a side effect. Restoring the source RMS first makes
        # ``energy`` the only thing that decides loudness, so "anger is louder
        # than neutral" holds no matter which other transforms ran.
        result_rms = float(np.sqrt(np.mean(signal**2))) if signal.size else 0.0
        if source_rms > 1e-6 and result_rms > 1e-6:
            signal = signal * (source_rms / result_rms)
        if abs(params.energy - 1.0) >= 0.01:
            signal = signal * float(params.energy)

        peak = float(np.max(np.abs(signal))) if signal.size else 0.0
        if peak > 0.99:
            signal = signal * (0.99 / peak)

        return signal.astype(np.float32)

    @staticmethod
    def _apply_contour(
        signal: np.ndarray,
        sample_rate: int,
        params: ProsodyParams,
    ) -> np.ndarray:
        """
        Impose an F0 trajectory by pitch-shifting successive blocks by a
        different amount, then cross-fading the blocks back together.

        A rising contour makes the utterance end higher than it began; jitter
        adds a small deterministic wobble on top so the line is not a ruler.
        """
        import librosa

        block = max(int(0.25 * sample_rate), 2048)
        fade = max(int(0.03 * sample_rate), 128)
        if signal.size <= block + fade:
            return signal

        # Blocks are pitch-shifted independently, so their phases no longer
        # line up. Overlap-adding them at 50% cancels energy and drops the
        # level by several dB, so instead the blocks abut and only a short
        # equal-power crossfade overlaps - cancellation stays inside the seam.
        hop = block - fade
        starts = list(range(0, signal.size - block + 1, hop))
        if not starts:
            return signal
        if starts[-1] + block < signal.size:
            starts.append(signal.size - block)

        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        fade_in = np.sin(ramp * np.pi / 2.0).astype(np.float32)
        fade_out = np.cos(ramp * np.pi / 2.0).astype(np.float32)

        out = np.zeros(signal.size, dtype=np.float32)
        last = max(len(starts) - 1, 1)

        for index, start in enumerate(starts):
            chunk = signal[start : start + block]
            position = index / last
            # Contour is centred so the mean pitch stays put.
            steps = params.contour * (position - 0.5)
            # Deterministic wobble: the same input always gives the same output.
            steps += params.jitter * float(math.sin(2.0 * math.pi * 1.7 * position))
            if abs(steps) >= 0.02:
                chunk = librosa.effects.pitch_shift(
                    chunk, sr=sample_rate, n_steps=float(steps)
                )
            chunk = np.asarray(chunk[:block], dtype=np.float32)

            if index == 0:
                out[start : start + block] = chunk
                continue
            out[start : start + fade] = (
                out[start : start + fade] * fade_out + chunk[:fade] * fade_in
            )
            out[start + fade : start + block] = chunk[fade:]

        return out.astype(np.float32)

    @staticmethod
    def _apply_tilt(signal: np.ndarray, tilt: float) -> np.ndarray:
        """
        First-order spectral tilt: blend in a high-passed copy to brighten
        (``tilt > 0``) or a low-passed copy to darken (``tilt < 0``).
        """
        # y[n] - y[n-1] is a one-pole differentiator, i.e. a +6 dB/octave tilt.
        high = np.empty_like(signal)
        high[0] = signal[0]
        high[1:] = signal[1:] - signal[:-1]

        if tilt > 0:
            return (signal + tilt * high).astype(np.float32)

        from scipy.signal import lfilter

        alpha = 0.6
        low = lfilter([1.0 - alpha], [1.0, -alpha], signal).astype(np.float32)
        mix = min(1.0, abs(tilt))
        return ((1.0 - mix) * signal + mix * low).astype(np.float32)


def preset_catalogue() -> list[dict]:
    """Serializable description of every preset, for the UI and API."""
    return [
        {
            "name": preset.name,
            "label": preset.label,
            "description": preset.description,
            "prosody": preset.prosody.to_dict(),
            "renderHint": {
                "happy": preset.happy_weight,
                "eyeblinkRate": preset.eyeblink_rate,
            },
        }
        for preset in PRESETS.values()
    ]
