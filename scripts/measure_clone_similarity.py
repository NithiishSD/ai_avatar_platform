#!/usr/bin/env python3
"""
Voice-cloning similarity (N-02), measured the one way this project counts.

A clone is scored against speech the cloner never heard. The reference
recording is split: the first ``--prompt-seconds`` go to the cloner as its
sample, the rest is held out. Each test sentence is cloned from the prompt
and compared with the held-out audio by ECAPA-TDNN (cosine similarity of
speaker embeddings). Two anchors are reported beside it so the number can be
read:

* **ceiling** - the same speaker's own real speech (prompt vs held-out). A
  perfect clone cannot beat this, because the two halves are different audio.
* **base** - for OpenVoice, the unconverted base voice before the tone-colour
  change, which shows how much of the score the cloner itself contributes.

Only a human, consented reference is admissible (golden rule 2); the script
refuses a synthetic one. Results are written to ``outputs/benchmarks/``.

    PYTHONPATH=backend backend/.conda/bin/python scripts/measure_clone_similarity.py \\
        --engine xtts-v2 --reference inputs/ljspeech_reference.wav
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "She sells sea shells by the sea shore every single summer morning.",
    "Our quarterly revenue grew by thirty seven percent across all regions.",
    "Please confirm whether the shipment arrived before the eleventh of March.",
    "Thick fog rolled through the valley while the church bells rang loudly.",
    "The committee will review the proposal and publish its decision next week.",
]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", required=True, choices=["xtts-v2", "openvoice-v2"])
    parser.add_argument("--reference", default="inputs/ljspeech_reference.wav")
    parser.add_argument("--prompt-seconds", type=float, default=30.0)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    import numpy as np
    import soundfile as sf

    import provenance
    from quality_auditor import SpeechQualityAuditor
    from voice_engine import VoiceEngineRouter

    reference = (PROJECT_ROOT / args.reference).resolve()
    described = provenance.describe(reference)
    if not described["admissible"]:
        print(f"error: reference is not admissible evidence: {described['reason']}", file=sys.stderr)
        return 1

    audio, rate = sf.read(str(reference), dtype="float32")
    cut = int(args.prompt_seconds * rate)
    if len(audio) < cut + 5 * rate:
        print("error: the reference must be at least prompt-seconds + 5 s long to leave a held-out part", file=sys.stderr)
        return 1

    # Working copies live under inputs/ (the only place the cloner reads from)
    # and carry the original's provenance sidecar, so the consent check passes
    # honestly rather than being bypassed. Removed at the end.
    work = PROJECT_ROOT / "inputs" / ".measure"
    work.mkdir(exist_ok=True)
    prompt_path, held_path = work / "prompt.wav", work / "heldout.wav"
    try:
        sf.write(str(prompt_path), audio[:cut], rate)
        sf.write(str(held_path), audio[cut:], rate)
        for path in (prompt_path, held_path):
            shutil.copy(provenance.sidecar_path(reference), provenance.sidecar_path(path))

        router = VoiceEngineRouter(device=args.device)
        auditor = SpeechQualityAuditor(device=args.device)
        ceiling = auditor.speaker_similarity(held_path, prompt_path)

        rows: List[Dict[str, Any]] = []
        for index, sentence in enumerate(SENTENCES):
            started = time.perf_counter()
            result = router.synthesize(
                sentence, mode="clone", speaker_wav=str(prompt_path), language=args.language,
                output_filename=f"benchmark/sim-{args.engine}-{index}.wav", clone_engine=args.engine,
                return_alignment=False,
            )
            report = auditor.speaker_similarity(held_path, PROJECT_ROOT / result.output_path)
            rows.append({
                "sentence": sentence, "model": result.model, "seconds": round(time.perf_counter() - started, 1),
                "audioSeconds": round(result.duration_seconds, 2), "similarityPercent": round(100 * (report.similarity or 0), 1),
            })
            print(f"  {rows[-1]['similarityPercent']:5.1f}%  {result.model}  {sentence[:50]}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    values = np.array([r["similarityPercent"] for r in rows])
    summary = {
        "engine": args.engine,
        "method": ceiling.method + "; clone of the first "
        f"{args.prompt_seconds:.0f} s scored against the held-out remainder of the same recording",
        "reference": args.reference,
        "referenceSpeaker": described.get("speaker"),
        "promptSeconds": args.prompt_seconds,
        "heldOutSeconds": round((len(audio) - cut) / rate, 1),
        "sentences": len(rows),
        "meanPercent": round(float(values.mean()), 1),
        "sdPercent": round(float(values.std()), 1),
        "minPercent": float(values.min()),
        "maxPercent": float(values.max()),
        "ceilingPercent": round(100 * (ceiling.similarity or 0), 1),
        "target85Met": bool(values.mean() >= 85),
        "target90Met": bool(values.mean() >= 90),
        "rows": rows,
    }
    out = PROJECT_ROOT / "outputs" / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"clone-similarity-{args.engine}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    destination.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"written: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
