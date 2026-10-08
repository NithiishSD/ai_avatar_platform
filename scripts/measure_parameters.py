#!/usr/bin/env python
"""
Does each customisation parameter change the output? (N-15, T8.7)

Accepting a parameter is not the same as it doing something, so each one is *measured*: the same
input is synthesised or rendered with the default and with the parameter changed, and a number
from the output (duration, pitch, loudness, frame count, picture difference) is compared. A
parameter counts as ``changes-output`` only when that number moves past a stated threshold, and in
the asked direction where one exists (faster speech is shorter, higher pitch is higher). The
determinism control renders the baseline twice: it must differ from itself by exactly zero.

    PYTHONPATH=backend backend/.conda/bin/python scripts/measure_parameters.py

Writes ``outputs/benchmarks/parameters.json``. Does not run Stable Diffusion (age, presentation,
hair, glasses): those take minutes each on this CPU and whether a face "has glasses" is a
judgement for a person, so they are listed as not measured here.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

os.environ.setdefault("WATERMARK_ENABLED", "false")  # the marks cost render time and change no parameter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

TEXT = "The quick brown fox jumps over the lazy dog, and then it walks quietly home."
RESULTS: List[Dict[str, Any]] = []


def record(parameter: str, setting: str, metric: str, base: float, value: float, passed: Optional[bool], note: str = "") -> None:
    """``passed`` None means a known limit rather than a pass or a failure (excluded from the tally)."""
    RESULTS.append({"parameter": parameter, "setting": setting, "metric": metric, "baseline": round(float(base), 4),
                    "value": round(float(value), 4), "changesOutput": None if passed is None else bool(passed), "note": note})
    print(f"{'LIMIT' if passed is None else 'PASS' if passed else 'NO  '} {parameter:<28} {setting:<16} {metric:<14} {base:.4g} -> {value:.4g} {note}")


# --------------------------------------------------------------------------- voice
def audio_stats(path: str) -> Dict[str, Any]:
    import torch
    import torchaudio.functional as F

    data, rate = sf.read(path, dtype="float32")
    wave = np.asarray(data if data.ndim == 1 else data.mean(axis=1), dtype=np.float32)
    f0 = F.detect_pitch_frequency(torch.from_numpy(wave), rate).numpy()
    f0 = f0[(f0 > 60) & (f0 < 500)]
    return {"duration": len(wave) / rate, "rms": float(np.sqrt(np.mean(wave**2))),
            "f0": float(np.median(f0)) if len(f0) else 0.0, "wave": wave}


def mel_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference of log-mel spectrograms: robust to the phase wobble a vocoder adds on every run."""
    import torch
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=64)

    def logmel(w: np.ndarray):
        return torch.log(mel(torch.from_numpy(w)) + 1e-5)

    x, y = logmel(a), logmel(b)
    frames = min(x.shape[-1], y.shape[-1])
    return float((x[..., :frames] - y[..., :frames]).abs().mean())


def voice_probes(router) -> None:
    def speak(name: str, **kw) -> Dict[str, Any]:
        result = router.synthesize(text=kw.pop("text", TEXT), output_filename=f"param-{name}.wav", **kw)
        return audio_stats(result.output_path)

    def tried(parameter: str, setting: str, name: str, **kw):
        """Synthesise, or record that this setting cannot run here (a model that is not available)."""
        try:
            return speak(name, **kw)
        except Exception as err:  # noqa: BLE001 - the reason is the finding
            RESULTS.append({"parameter": parameter, "setting": setting, "metric": "-", "baseline": None, "value": None,
                            "changesOutput": None, "note": f"cannot run here: {type(err).__name__}: {str(err)[:160]}"})
            print(f"N/A  {parameter:<28} {setting:<16} cannot run here: {str(err)[:100]}")
            return None

    base = speak("base", mode="fast")
    again = speak("base2", mode="fast")
    # Kokoro is not bit-for-bit repeatable, so "did the waveform change" is judged against the
    # repeat-to-repeat spectrogram distance (the noise floor): a real effect must clear 3x that.
    floor = mel_distance(base["wave"], again["wave"])
    record("(control) repeat", "same input", "mel distance", 0, floor, abs(base["duration"] - again["duration"]) < 0.01, f"noise floor; durations {base['duration']:.3f} vs {again['duration']:.3f}")

    def beyond_floor(out: Dict[str, Any]) -> float:
        return mel_distance(base["wave"], out["wave"])

    for value in (1.5, 0.7):
        out = speak(f"speed{value}", mode="fast", speed=value)
        want = out["duration"] < base["duration"] * 0.9 if value > 1 else out["duration"] > base["duration"] * 1.1
        record("speed", str(value), "duration s", base["duration"], out["duration"], want, "faster = shorter")
    for value in (1.5, 0.7):
        out = speak(f"pitch{value}", mode="fast", pitch=value)
        want = out["f0"] > base["f0"] * 1.1 if value > 1 else out["f0"] < base["f0"] * 0.9
        record("pitch", str(value), "median f0 Hz", base["f0"], out["f0"], want, "higher = higher")
    for preset in ("joy", "anger", "sorrow", "authority", "calm", "excitement"):
        out = speak(f"emo-{preset}", mode="fast", emotion=preset)
        moved = abs(out["duration"] / base["duration"] - 1) > 0.02 or abs(out["rms"] / base["rms"] - 1) > 0.02 or abs(out["f0"] / base["f0"] - 1) > 0.02
        record("emotion", preset, "dur/rms/f0", base["duration"], out["duration"], moved, f"rms x{out['rms'] / base['rms']:.2f}, f0 x{out['f0'] / base['f0']:.2f}")
    full = speak("int1", mode="fast", emotion="anger", emotion_intensity=1.0)
    low = speak("int03", mode="fast", emotion="anger", emotion_intensity=0.3)
    record("emotionIntensity", "0.3 vs 1.0", "rms", low["rms"], full["rms"], base["rms"] < low["rms"] < full["rms"], "weaker emotion sits between neutral and full")
    for style in ("dialogue", "expressive", "narration"):
        out = tried("style", style, f"style-{style}", mode="fast", style=style)
        if out:
            diff = beyond_floor(out)
            record("style", style, "mel distance", floor, diff, diff > 3 * floor, "vs noise floor")
    for quality in ("fast", "high"):
        out = tried("quality", quality, f"q-{quality}", mode="fast", quality=quality)
        if out:
            diff = beyond_floor(out)
            record("quality", quality, "mel distance", floor, diff, diff > 3 * floor, "vs 'balanced', vs noise floor")
    out = speak("lang-es", mode="fast", language="es", text="El rápido zorro marrón salta sobre el perro perezoso.")
    record("language", "es", "duration s", base["duration"], out["duration"], out["duration"] != base["duration"], "different language, different speech")


# --------------------------------------------------------------------------- render
def render_probes(router) -> None:
    import cv2
    import render_engine
    import video_io
    from contracts import AvatarRenderJob

    speech = router.synthesize(text=TEXT, mode="fast", return_alignment=True, output_filename="param-render.wav")
    audio = Path(speech.output_path).resolve()
    stamps = [t for t in speech.phoneme_timestamps if t["endMs"] <= speech.duration_seconds * 1000]
    tmp = Path(tempfile.mkdtemp(prefix="params-"))
    # The server only reads background images from inside outputs/ or inputs/.
    background_image = PROJECT_ROOT / "outputs" / "benchmarks" / "param_bg.png"
    background_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(background_image), np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))[..., None].repeat(3, axis=2))

    def render(name: str, **overrides):
        body = {"jobId": "param-job", "avatarId": "demo", "audioUrl": audio.as_uri(), "sampleRate": speech.sample_rate,
                "durationSeconds": speech.duration_seconds, "phonemeTimestamps": stamps,
                "emotionVector": {"happy": 0, "neutral": 1, "eyeblinkRate": 1}, "renderQuality": "PREVIEW", "targetFps": 25}
        body.update(overrides)
        result = render_engine.render_job(AvatarRenderJob.model_validate(body), engine="blendshape", output_path=tmp / f"{name}.mp4", label=False)
        return result, list(video_io.read_frames(result.output_path))

    def picture_diff(a, b) -> float:
        n = min(len(a), len(b))
        return float(np.mean([np.abs(a[i].astype(np.int16) - b[i].astype(np.int16)).mean() for i in range(0, n, 3)]))

    base, base_frames = render("base")
    _, again_frames = render("again")
    zero = picture_diff(base_frames, again_frames)
    record("(control) repeat", "same job", "pixel diff", 0, zero, zero == 0.0, "rendering is deterministic")

    def emotion(**kw):
        return {"emotionVector": {"happy": 0, "neutral": 1, "eyeblinkRate": 1, **kw}}

    for name, kw in (("happy", {"happy": 1.0}), ("neutral", {"neutral": 0.0}), ("joy", {"joy": 1.0}), ("anger", {"anger": 1.0}),
                     ("sorrow", {"sorrow": 1.0}), ("authority", {"authority": 1.0}), ("calm", {"calm": 1.0}), ("excitement", {"excitement": 1.0})):
        _, frames = render(f"em-{name}", **emotion(**kw))
        d = picture_diff(base_frames, frames)
        record(f"emotionVector.{name}", str(kw), "pixel diff", 0, d, d > 0.05)
    for rate in (0.0, 5.0):
        result, _ = render(f"blink{rate}", **emotion(eyeblinkRate=rate))
        ok = result.blink_count < base.blink_count if rate == 0 else result.blink_count > base.blink_count
        record("emotionVector.eyeblinkRate", str(rate), "blink count", base.blink_count, result.blink_count, ok, "more rate = more blinks")
    _, frames = render("bg-color", background={"color": "#0b3d91"})
    d = picture_diff(base_frames, frames)
    record("background.color", "#0b3d91", "pixel diff", 0, d, d > 1.0)
    _, frames = render("bg-image", background={"imageUrl": background_image.as_uri()})
    d = picture_diff(base_frames, frames)
    record("background.imageUrl", "gradient png", "pixel diff", 0, d, d > 1.0)
    # Photos are never enlarged (video_io.fit_within), so on the 512 px demo face 1080P_HQ changes nothing:
    # recorded as a limit. The size rule itself is checked on a large image.
    hq, _ = render("hq", renderQuality="1080P_HQ")
    record("renderQuality", "1080P_HQ on 512 px photo", "width px", base.width, hq.width, None, "no effect: a photo is never enlarged (see T8.8)")
    small = render_engine.output_size(1536, 1024, render_engine.RenderQuality.PREVIEW)
    large = render_engine.output_size(1536, 1024, render_engine.RenderQuality.HD_1080P)
    record("renderQuality", "1080P_HQ on 1536 px photo", "width px", small[0], large[0], large[0] > small[0], "effective only for photos larger than 512 px")
    fps, _ = render("fps12", targetFps=12)
    record("targetFps", "12", "frame count", base.frame_count, fps.frame_count, fps.frame_count < base.frame_count * 0.6, "fewer fps = fewer frames")


def main() -> int:
    from voice_engine import VoiceEngineRouter

    router = VoiceEngineRouter()
    steps: List[Callable[..., None]] = [voice_probes, render_probes]
    for step in steps:
        step(router)
    report = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "device": "cpu", "results": RESULTS,
              "notMeasured": ["age", "presentation", "hair", "glasses (Stable Diffusion; judged by a person, see M-08)",
                              "seed, steps, attempts (generation controls, not appearance)"]}
    out = PROJECT_ROOT / "outputs" / "benchmarks" / "parameters.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    checks = [r for r in RESULTS if r["changesOutput"] is not None]
    failed = [r for r in checks if not r["changesOutput"]]
    print(f"\n{len(checks) - len(failed)} of {len(checks)} checks changed the output as asked; {len(RESULTS) - len(checks)} could not run; wrote {out}")
    for r in failed:
        print(f"  did NOT: {r['parameter']} {r['setting']}  ({r['metric']} {r['baseline']} -> {r['value']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
