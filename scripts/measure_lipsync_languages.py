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

os.environ.setdefault("WATERMARK_ENABLED", "false")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

REFERENCE = PROJECT_ROOT / "inputs" / "ljspeech_reference.wav"
# (language code, how it is spoken, sentence). "xtts" = clone the reference in that language; "mms" = MMS-TTS.
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
    from contracts import AvatarRenderJob
    from emotion_engine import to_render_emotion_vector

    row: Dict[str, Any] = {"language": language, "speech": how, "engine": engine}
    job_id = f"lang-{language}-{engine}"
    started = time.perf_counter()
    kwargs: Dict[str, Any] = {"text": text, "language": language, "return_alignment": True, "output_filename": f"{job_id}.wav"}
    kwargs.update({"mode": "clone", "speaker_wav": str(REFERENCE), "clone_engine": "xtts-v2"} if how == "xtts" else {"mode": "fast"})
    speech = router.synthesize(**kwargs)
    row.update({"model": speech.model, "alignment": speech.alignment_method, "audioSeconds": round(speech.duration_seconds, 2)})
    stamps = [t for t in (speech.phoneme_timestamps or []) if t["endMs"] <= speech.duration_seconds * 1000]
    if not stamps:
        raise RuntimeError("alignment produced no timestamps")
    row["speechSeconds"] = round(time.perf_counter() - started, 1)
    job = AvatarRenderJob.model_validate({
        "jobId": job_id, "avatarId": face, "audioUrl": Path(speech.output_path).resolve().as_uri(), "sampleRate": speech.sample_rate,
        "durationSeconds": speech.duration_seconds, "phonemeTimestamps": stamps,
        "emotionVector": to_render_emotion_vector((speech.emotion or {}).get("vector") or {}), "renderQuality": "PREVIEW", "targetFps": 25})
    result = render_engine.render_job(job, engine=engine, label=False)
    score = lipsync.score_video(result.output_path).to_dict()
    row.update({"renderSeconds": round(result.render_seconds, 1), "offsetFrames": score["offsetFrames"], "lseC": score["lseC"], "lseD": score["lseD"],
                "secondsWithinOneFrame": score["secondsWithinOneFrame"]})
    return row


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", nargs="+", default=["blendshape", "wav2lip"])
    parser.add_argument("--face", default="demo")
    parser.add_argument("--only", nargs="*", help="language codes to run (default: all)")
    args = parser.parse_args(argv)

    import lipsync_metric
    import render_engine
    from voice_engine import VoiceEngineRouter

    router = VoiceEngineRouter()
    rows: List[Dict[str, Any]] = []
    for language, how, text in CASES:
        if args.only and language not in args.only:
            continue
        for engine in args.engines:
            try:
                row = run_case(router, render_engine, lipsync_metric, language, how, text, engine, args.face)
            except Exception as err:  # noqa: BLE001 - the reason is the finding
                row = {"language": language, "speech": how, "engine": engine, "error": f"{type(err).__name__}: {str(err)[:200]}"}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    summary: Dict[str, Any] = {}
    for engine in args.engines:
        ok = [r for r in rows if r["engine"] == engine and "error" not in r]
        pct = [r["secondsWithinOneFrame"]["percent"] for r in ok if r["secondsWithinOneFrame"]["percent"] is not None]
        blocks = [r["secondsWithinOneFrame"] for r in ok]
        summary[engine] = {
            "languagesScored": len({r["language"] for r in ok}), "clips": len(ok), "failed": len([r for r in rows if r["engine"] == engine and "error" in r]),
            "offsetZeroClips": sum(1 for r in ok if r["offsetFrames"] == 0), "offsetWithinOne": sum(1 for r in ok if abs(r["offsetFrames"]) <= 1),
            "meanPercentSecondsWithinOne": round(sum(pct) / len(pct), 1) if pct else None,
            "pooledPercentSecondsWithinOne": round(100.0 * sum(b["within"] for b in blocks) / max(1, sum(b["of"] for b in blocks)), 1),
            "meanLseC": round(sum(r["lseC"] for r in ok) / len(ok), 2) if ok else None}
    report = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "device": "cpu", "face": args.face, "rows": rows, "summary": summary}
    out = PROJECT_ROOT / "outputs" / "benchmarks" / f"lipsync-languages-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n" + json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
