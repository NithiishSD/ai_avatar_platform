#!/usr/bin/env python
"""
Lip sync across many languages (N-05, N-18, T8.4): SyncNet on a rendered talking avatar per language.

For each language a short sentence is spoken (XTTS-v2 cloning the consented LJSpeech reference, or
MMS-TTS where only that has the language), aligned, rendered with each engine, and scored with
SyncNet. Per clip it reports the best offset, LSE-C / LSE-D and the D-12 figure: the share of
1-second blocks whose own best offset is within +/-1 frame. The summary averages that figure over
languages. Watermarks are off here (``WATERMARK_ENABLED=false``) because they change render time,
not the mouth.

    PYTHONPATH=backend backend/.conda/bin/python scripts/measure_lipsync_languages.py [--engines blendshape wav2lip]

Writes ``outputs/benchmarks/lipsync-languages-<date>.json``. One failing language is recorded with
its reason and the run goes on.

Terms used in the report:

* **SyncNet** is a network that embeds a short window of mouth video and the
  matching audio, and measures how close the two embeddings are. Sliding the
  audio against the video finds the **offset** (in frames) where they agree
  best; 0 means audio and mouth are in step.
* **LSE-D** (lip-sync error, distance) is the embedding distance at that best
  offset: lower is better. **LSE-C** (confidence) is how much better the best
  offset is than the others: higher is better.
* ``--only hi tam`` limits the run to some language codes; ``--engines``
  picks the render engines; ``--face`` the registered avatar.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Watermarking is switched off for this process; the backend reads the
# variable each time it marks a file. setdefault keeps a value the caller
# already set, so WATERMARK_ENABLED=true on the command line still wins.
os.environ.setdefault("WATERMARK_ENABLED", "false")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Lets "import render_engine" and friends find backend/ without PYTHONPATH.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# The consented voice every XTTS case clones (golden rule 3: only a voice with
# a provenance record may be used).
REFERENCE = PROJECT_ROOT / "inputs" / "ljspeech_reference.wav"
# (language code, how it is spoken, sentence). "xtts" = clone the reference in that language; "mms" = MMS-TTS.
# MMS cases use three-letter ISO 639-3 codes (hin, swh, tam), which is how
# MMS-TTS names its languages; XTTS uses two-letter codes. Hindi appears in
# both forms, once per speech path.
CASES = [
    ("en", "xtts", "Hello, I am a synthetic avatar and I speak with a cloned voice."),
    ("es", "xtts", "Hola, soy un avatar sintético y hablo con una voz clonada."),
    ("fr", "xtts", "Bonjour, je suis un avatar synthétique et je parle avec une voix clonée."),
    ("de", "xtts", "Hallo, ich bin ein synthetischer Avatar und spreche mit einer geklonten Stimme."),
    ("it", "xtts", "Ciao, sono un avatar sintetico e parlo con una voce clonata."),
    ("pt", "xtts", "Olá, eu sou um avatar sintético e falo com uma voz clonada."),
    ("pl", "xtts", "Cześć, jestem syntetycznym awatarem i mówię sklonowanym głosem."),
    ("ru", "xtts", "Привет, я синтетический аватар и говорю клонированным голосом."),
    ("tr", "xtts", "Merhaba, ben yapay bir avatarım ve klonlanmış bir sesle konuşuyorum."),
    ("nl", "xtts", "Hallo, ik ben een synthetische avatar en ik spreek met een gekloneerde stem."),
    ("hi", "xtts", "नमस्ते, मैं एक कृत्रिम अवतार हूँ और मैं क्लोन की गई आवाज़ में बोलता हूँ।"),
    ("hin", "mms", "नमस्ते, मैं एक कृत्रिम अवतार हूँ और मैं बोल रहा हूँ।"),
    ("swh", "mms", "Habari, mimi ni avatar bandia na ninazungumza kwa sauti ya kompyuta."),
    ("tam", "mms", "வணக்கம், நான் ஒரு செயற்கை அவதாரம், நான் பேசுகிறேன்."),
]


def run_case(router, render_engine, lipsync, language: str, how: str, text: str, engine: str, face: str) -> Dict[str, Any]:
    """
    Speak, render and score one (language, engine) pair; return its report row.

    The backend modules are passed in rather than imported here, so ``main``
    imports them once for the whole run.
    """
    from contracts import AvatarRenderJob
    from emotion_engine import to_render_emotion_vector

    row: Dict[str, Any] = {"language": language, "speech": how, "engine": engine}
    job_id = f"lang-{language}-{engine}"
    started = time.perf_counter()
    kwargs: Dict[str, Any] = {"text": text, "language": language, "return_alignment": True, "output_filename": f"{job_id}.wav"}
    kwargs.update({"mode": "clone", "speaker_wav": str(REFERENCE), "clone_engine": "xtts-v2"} if how == "xtts" else {"mode": "fast"})
    speech = router.synthesize(**kwargs)
    # Record what actually spoke and how it was aligned (golden rule 2: every
    # number records its method).
    row.update({"model": speech.model, "alignment": speech.alignment_method, "audioSeconds": round(speech.duration_seconds, 2)})
    # Drop any timestamp that ends past the audio; the render contract
    # cannot animate a phoneme after the sound has stopped.
    stamps = [t for t in (speech.phoneme_timestamps or []) if t["endMs"] <= speech.duration_seconds * 1000]
    if not stamps:
        raise RuntimeError("alignment produced no timestamps")
    row["speechSeconds"] = round(time.perf_counter() - started, 1)
    # model_validate runs the contract's checks on the dict, exactly as the
    # API would, so a bad job fails here with a clear validation error.
    # as_uri() turns the local path into a file:// URL.
    job = AvatarRenderJob.model_validate({
        "jobId": job_id, "avatarId": face, "audioUrl": Path(speech.output_path).resolve().as_uri(), "sampleRate": speech.sample_rate,
        "durationSeconds": speech.duration_seconds, "phonemeTimestamps": stamps,
        "emotionVector": to_render_emotion_vector((speech.emotion or {}).get("vector") or {}), "renderQuality": "PREVIEW", "targetFps": 25})
    # label=False omits the visible disclosure label from the frames.
    result = render_engine.render_job(job, engine=engine, label=False)
    score = lipsync.score_video(result.output_path).to_dict()
    row.update({"renderSeconds": round(result.render_seconds, 1), "offsetFrames": score["offsetFrames"], "lseC": score["lseC"], "lseD": score["lseD"],
                "secondsWithinOneFrame": score["secondsWithinOneFrame"]})
    return row


def main(argv: Optional[List[str]] = None) -> int:
    """Run every selected case, print each row as it finishes, and write the report."""
    # argv=None makes argparse read sys.argv; a test can pass its own list.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # nargs="+" takes one or more values after the flag; "*" allows zero.
    parser.add_argument("--engines", nargs="+", default=["blendshape", "wav2lip"])
    parser.add_argument("--face", default="demo")
    parser.add_argument("--only", nargs="*", help="language codes to run (default: all)")
    args = parser.parse_args(argv)

    # Imported after argument parsing so --help is instant and does not pull
    # in torch and the model code.
    import lipsync_metric
    import render_engine
    from voice_engine import VoiceEngineRouter

    # One router for the whole run, shared by every case.
    router = VoiceEngineRouter()
    rows: List[Dict[str, Any]] = []
    for language, how, text in CASES:
        if args.only and language not in args.only:
            continue
        for engine in args.engines:
            try:
                row = run_case(router, render_engine, lipsync_metric, language, how, text, engine, args.face)
            except Exception as err:  # noqa: BLE001 - the reason is the finding
                # Truncated to 200 characters so one long traceback message
                # does not swamp the report.
                row = {"language": language, "speech": how, "engine": engine, "error": f"{type(err).__name__}: {str(err)[:200]}"}
            rows.append(row)
            # Printed as each case finishes (flush=True), so a long run shows
            # progress. ensure_ascii=False keeps non-Latin text readable.
            print(json.dumps(row, ensure_ascii=False), flush=True)

    summary: Dict[str, Any] = {}
    for engine in args.engines:
        ok = [r for r in rows if r["engine"] == engine and "error" not in r]
        pct = [r["secondsWithinOneFrame"]["percent"] for r in ok if r["secondsWithinOneFrame"]["percent"] is not None]
        blocks = [r["secondsWithinOneFrame"] for r in ok]
        # Two averages of the D-12 figure: "mean" weights every clip equally,
        # "pooled" adds up all 1-second blocks first, so longer clips count
        # for more. Reporting both shows whether short clips skew the mean.
        summary[engine] = {
            "languagesScored": len({r["language"] for r in ok}), "clips": len(ok), "failed": len([r for r in rows if r["engine"] == engine and "error" in r]),
            "offsetZeroClips": sum(1 for r in ok if r["offsetFrames"] == 0), "offsetWithinOne": sum(1 for r in ok if abs(r["offsetFrames"]) <= 1),
            "meanPercentSecondsWithinOne": round(sum(pct) / len(pct), 1) if pct else None,
            "pooledPercentSecondsWithinOne": round(100.0 * sum(b["within"] for b in blocks) / max(1, sum(b["of"] for b in blocks)), 1),
            "meanLseC": round(sum(r["lseC"] for r in ok) / len(ok), 2) if ok else None}
    # The device is a fixed string here, not detected from torch.
    report = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "device": "cpu", "face": args.face, "rows": rows, "summary": summary}
    out = PROJECT_ROOT / "outputs" / "benchmarks" / f"lipsync-languages-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n" + json.dumps(summary, indent=2))
    print(f"wrote {out}")
    # Always 0: failed cases are findings recorded in the report, not a
    # failure of the measurement script itself.
    return 0


if __name__ == "__main__":
    sys.exit(main())
