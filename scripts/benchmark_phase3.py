#!/usr/bin/env python
"""
Benchmark harness for the assignment's acceptance matrix.

Produces real, dated evidence for the thresholds that were previously marked
NOT MEASURED, and writes both JSON and a Markdown summary under
``docs/benchmarks/``.

    PYTHONPATH=backend backend/.conda/bin/python scripts/benchmark_phase3.py

Sections, each skippable:
  --skip-synthesis   warm/cold synthesis latency per model
  --skip-quality     SQUIM MOS / PESQ / STOI over a sentence set
  --skip-api         API initiation latency and sustained throughput
  --skip-similarity  ECAPA speaker similarity for cloned voices

Honesty rules this harness follows:
  * Every number records the method that produced it.
  * A threshold is only marked PASS when the measurement that backs it is the
    method the assignment names (e.g. ECAPA-TDNN for cloning similarity).
  * API initiation latency is meaningless while the queue runs in eager mode,
    because the POST then performs the synthesis inline. The harness detects
    that and says so instead of reporting a misleading number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

# A type variable: lets timed() return whatever the timed function returns,
# instead of erasing it to `object` and making every caller cast.
T = TypeVar("T")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

REPORT_DIR = PROJECT_ROOT / "docs" / "benchmarks"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmark"

# Phonetically varied sentences, so MOS is not measured on one lucky utterance.
SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "She sells sea shells by the sea shore every single summer morning.",
    "Our quarterly revenue grew by thirty seven percent across all regions.",
    "Please confirm whether the shipment arrived before the eleventh of March.",
    "Thick fog rolled through the valley while the church bells rang loudly.",
]

MULTILINGUAL_CASES = [
    ("hin", "नमस्ते, मैं आपका कृत्रिम बुद्धिमत्ता अवतार हूँ।"),
    ("tam", "வணக்கம், நான் உங்கள் செயற்கை நுண்ணறிவு அவதாரம்."),
    ("swh", "Habari, mimi ni avatar yako ya akili bandia."),
    ("spa", "Hola, soy tu avatar de inteligencia artificial."),
]

# Thresholds from the assignment.
TARGETS = {
    "mos": ("MOS", 3.5, "greater"),
    "similarity": ("Voice cloning similarity", 0.85, "greater"),
    "api_initiation_ms": ("API initiation response", 500.0, "less"),
    "throughput_rpm": ("API capacity", 100.0, "greater"),
    "warm_synthesis_ms": ("Warm synthesis latency", 2000.0, "less"),
}


def timed(fn: Callable[[], T]) -> tuple[float, T]:
    start = time.perf_counter()
    value = fn()
    return (time.perf_counter() - start) * 1000.0, value


def summarize(samples: list[float]) -> dict:
    if not samples:
        return {}
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "meanMs": round(statistics.fmean(ordered), 2),
        "medianMs": round(statistics.median(ordered), 2),
        "p95Ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
        "minMs": round(ordered[0], 2),
        "maxMs": round(ordered[-1], 2),
    }


def verdict(key: str, value: Optional[float]) -> str:
    if value is None:
        return "NOT MEASURED"
    _, threshold, direction = TARGETS[key]
    ok = value > threshold if direction == "greater" else value < threshold
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def benchmark_synthesis(router, repeats: int = 3) -> dict:
    """Cold vs warm latency, and the real-time factor, for each backend."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"english": {}, "multilingual": {}}

    cold_ms, result = timed(
        lambda: router.synthesize(
            SENTENCES[0], mode="fast",
            output_filename="benchmark/kokoro_cold.wav",
        )
    )
    results["english"]["kokoro"] = {
        "coldMs": round(cold_ms, 2),
        "model": result.model,
    }

    warm_samples = []
    audio_seconds = 0.0
    for index in range(repeats):
        for sentence in SENTENCES:
            elapsed, result = timed(
                lambda s=sentence, i=index: router.synthesize(
                    s, mode="fast",
                    output_filename=f"benchmark/kokoro_warm_{i}.wav",
                )
            )
            warm_samples.append(elapsed)
            audio_seconds += result.duration_seconds

    stats = summarize(warm_samples)
    total_ms = sum(warm_samples)
    stats["realTimeFactor"] = round((total_ms / 1000.0) / max(audio_seconds, 1e-6), 4)
    stats["audioSecondsGenerated"] = round(audio_seconds, 2)
    results["english"]["kokoro"].update({"warm": stats})

    for iso3, text in MULTILINGUAL_CASES:
        try:
            cold_ms, result = timed(
                lambda t=text, c=iso3: router.synthesize(
                    t, mode="multilingual", language=c,
                    output_filename=f"benchmark/mms_{c}.wav",
                )
            )
            warm_ms, _ = timed(
                lambda t=text, c=iso3: router.synthesize(
                    t, mode="multilingual", language=c,
                    output_filename=f"benchmark/mms_{c}.wav",
                )
            )
            results["multilingual"][iso3] = {
                "model": result.model,
                "languageName": (result.language or {}).get("name"),
                "coldMs": round(cold_ms, 2),
                "warmMs": round(warm_ms, 2),
                "durationSeconds": round(result.duration_seconds, 2),
                "sampleRate": result.sample_rate,
            }
        except Exception as exc:  # noqa: BLE001 - a failed language is data too
            results["multilingual"][iso3] = {"error": f"{type(exc).__name__}: {exc}"}

    return results


def benchmark_emotions(router) -> dict:
    """Confirm each emotion changes duration and loudness in its stated direction."""
    import numpy as np
    import soundfile as sf

    from emotion_engine import PRESETS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = router.synthesize(
        SENTENCES[0], mode="fast", output_filename="benchmark/emotion_neutral.wav"
    )
    reference, _ = sf.read(str(PROJECT_ROOT / baseline.output_path), dtype="float32")
    base_rms = float(np.sqrt(np.mean(reference**2)))
    base_duration = baseline.duration_seconds

    rows = {}
    for name in PRESETS:
        if name == "neutral":
            continue
        result = router.synthesize(
            SENTENCES[0], mode="fast", emotion=name,
            output_filename=f"benchmark/emotion_{name}.wav",
        )
        audio, _ = sf.read(str(PROJECT_ROOT / result.output_path), dtype="float32")
        rms = float(np.sqrt(np.mean(audio**2)))
        rows[name] = {
            "durationRatio": round(result.duration_seconds / base_duration, 3),
            "expectedRateInverse": round(1.0 / PRESETS[name].prosody.rate, 3),
            "loudnessRatio": round(rms / base_rms, 3),
            "expectedEnergy": PRESETS[name].prosody.energy,
            "latencyMs": round(result.latency_ms, 1),
        }
    return {"baselineSeconds": round(base_duration, 3), "emotions": rows}


def benchmark_quality(router, auditor) -> dict:
    """SQUIM MOS / PESQ / STOI across the sentence set."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # A non-matching reference removes the self-reference bias in SQUIM's
    # subjective head.
    reference = router.synthesize(
        "This recording is the non matching reference clip for scoring.",
        mode="fast", output_filename="benchmark/quality_reference.wav",
    )
    reference_path = PROJECT_ROOT / reference.output_path

    rows: list[dict[str, Any]] = []
    for index, sentence in enumerate(SENTENCES):
        result = router.synthesize(
            sentence, mode="fast", output_filename=f"benchmark/quality_{index}.wav"
        )
        report = auditor.audit(
            PROJECT_ROOT / result.output_path, reference_path=reference_path
        )
        rows.append(
            {
                "sentence": sentence,
                "model": result.model,
                **{
                    key: report.to_dict()[key]
                    for key in ("mos", "pesq", "stoi", "siSdr", "method")
                },
            }
        )

    scored = [row["mos"] for row in rows if row["mos"] is not None]
    methods = {row["method"] for row in rows}
    return {
        "perSentence": rows,
        "meanMos": round(statistics.fmean(scored), 3) if scored else None,
        "minMos": round(min(scored), 3) if scored else None,
        "meanPesq": round(
            statistics.fmean([r["pesq"] for r in rows if r["pesq"] is not None]), 3
        ) if any(r["pesq"] is not None for r in rows) else None,
        "methods": sorted(methods),
        # Only a real model prediction counts as evidence.
        "isModelPrediction": methods == {"torchaudio-squim"},
    }


def benchmark_similarity(router, auditor) -> dict:
    """ECAPA speaker similarity between a reference recording and its clone."""
    from audio_utils import list_voice_samples

    samples = list_voice_samples(PROJECT_ROOT / "inputs")
    if not samples:
        return {
            "status": "SKIPPED",
            "reason": (
                "No reference recording in inputs/. Add 30-60s of consented "
                "speech to measure cloning similarity."
            ),
        }

    import provenance

    # Prefer a reference that may actually back the published threshold, so a
    # synthetic smoke-test clip sitting in inputs/ never shadows a real one.
    described = [(s, provenance.describe(s.path)) for s in samples]
    admissible = [pair for pair in described if pair[1]["admissible"]]
    reference, prov = admissible[0] if admissible else described[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        cloned = router.synthesize(
            SENTENCES[0], mode="clone", speaker_wav=reference.path,
            output_filename="benchmark/clone.wav",
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}

    report = auditor.speaker_similarity(
        reference.path, PROJECT_ROOT / cloned.output_path
    )
    measured = report.to_dict()

    # A number was produced either way; whether it counts is a separate
    # question, and build_matrix only admits an admissible reference.
    status = "MEASURED" if prov["admissible"] else "PIPELINE_TEST"
    return {
        "status": status,
        "reason": prov["reason"],
        "referenceFile": reference.filename,
        "referenceSeconds": round(reference.duration_seconds, 2),
        "cloneLatencyMs": round(cloned.latency_ms, 1),
        "provenance": prov,
        "admissible": prov["admissible"],
        **measured,
        "warnings": list(measured.get("warnings", []))
        + provenance.warnings_for(reference.path),
    }


def benchmark_api(requests_count: int = 120) -> dict:
    """
    API initiation latency and throughput, measured in-process.

    Two different numbers matter here and they are reported separately:

    * ``configuredLimitRpm`` - what the rate limiter actually allows. This is
      the capacity a client sees, and it is what the acceptance threshold is
      judged against.
    * ``rawThroughputRpm``  - how fast the application can serve requests with
      the limiter bypassed. This is the headroom above the policy, not a
      deployed-server figure.
    """
    import os

    from fastapi.testclient import TestClient

    import app as app_module
    from security import SecurityConfig, SecurityGate

    eager = os.getenv("QUEUE_BACKEND", "in_memory").lower() != "celery"
    configured = app_module.security_gate.config
    # Any values: the report mixes flags, counts, nested summaries and floats.
    result: dict[str, Any] = {
        "queueMode": "eager (in_memory)" if eager else "celery",
        "configuredLimitRpm": (
            configured.requests_per_minute if configured.rate_limit_enabled else None
        ),
        "rateLimitEnabled": configured.rate_limit_enabled,
        "authEnabled": configured.auth_enabled,
    }

    # Measure the application itself with the limiter out of the way, then put
    # the real gate back. Benchmarking through the limiter would only measure
    # the limiter.
    original_gate = app_module.security_gate
    app_module.security_gate = SecurityGate(
        SecurityConfig(
            auth_enabled=False,
            api_keys=frozenset(),
            rate_limit_enabled=False,
            requests_per_minute=configured.requests_per_minute,
            burst=configured.burst,
        )
    )
    try:
        client = TestClient(app_module.app)

        light = []
        for _ in range(requests_count):
            elapsed, response = timed(
                lambda: client.get("/api/v1/audio/languages?limit=5")
            )
            if response.status_code == 200:
                light.append(elapsed)

        start = time.perf_counter()
        completed = 0
        for _ in range(requests_count):
            if client.get("/api/v1/audio/languages?limit=20").status_code == 200:
                completed += 1
        window = time.perf_counter() - start

        result["readEndpoint"] = summarize(light)
        result["rawThroughputRpm"] = (
            round(completed / window * 60.0, 1) if window > 0 else 0.0
        )
        result["throughputSampleSize"] = completed
        result["throughputCaveat"] = (
            "Sequential in-process ASGI calls with no network, TLS or "
            "concurrency. An upper bound on the application layer; a load test "
            "against a running uvicorn with concurrent clients is still "
            "outstanding."
        )

        if eager:
            result["initiation"] = {
                "status": "NOT COMPARABLE",
                "reason": (
                    "QUEUE_BACKEND=in_memory runs Celery eagerly, so POST "
                    "/api/v1/audio/synthesize performs the synthesis inline and "
                    "its response time is synthesis time, not initiation time. "
                    "Re-run with QUEUE_BACKEND=celery and a live worker to "
                    "measure the <500 ms initiation threshold."
                ),
            }
        else:
            samples = []
            for index in range(10):
                elapsed, response = timed(
                    lambda i=index: client.post(
                        "/api/v1/audio/synthesize",
                        json={
                            "text": SENTENCES[i % len(SENTENCES)],
                            "mode": "fast",
                            "outputFilename": f"benchmark/api_{i}.wav",
                        },
                    )
                )
                if response.status_code in (200, 202):
                    samples.append(elapsed)
            result["initiation"] = summarize(samples)
    finally:
        app_module.security_gate = original_gate

    # The capacity a client actually gets is the lower of policy and headroom.
    served = result.get("rawThroughputRpm") or 0.0
    limit = result.get("configuredLimitRpm")
    result["effectiveCapacityRpm"] = min(served, limit) if limit else served
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _inadmissible_method(similarity: dict) -> str:
    """Explain why a similarity score is not counted, quoting it if one exists."""
    if not similarity:
        return "not run"
    reason = similarity.get("reason") or "no admissible reference"
    score = similarity.get("similarityPercent")
    if score is not None and not similarity.get("admissible", True):
        return f"not counted ({reason}); indicative score was {score}%"
    return reason


def build_matrix(report: dict) -> list[dict]:
    """Map raw measurements onto the assignment's acceptance thresholds."""
    quality = report.get("quality") or {}
    similarity = report.get("similarity") or {}
    api = report.get("api") or {}
    synthesis = report.get("synthesis") or {}

    mos = quality.get("meanMos") if quality.get("isModelPrediction") else None
    warm = (
        synthesis.get("english", {}).get("kokoro", {}).get("warm", {}).get("medianMs")
    )
    initiation = api.get("initiation", {})
    initiation_ms = initiation.get("medianMs") if isinstance(initiation, dict) else None
    # Admissible evidence for the cloning threshold needs both the method the
    # assignment names (ECAPA-TDNN) and a reference whose provenance allows it
    # to be published. Synthetic references are measured but never counted.
    sim = (
        similarity.get("similarity")
        if similarity.get("isEcapa") and similarity.get("admissible")
        else None
    )

    rows = [
        {
            "requirement": "TTS quality MOS > 3.5",
            "measured": mos,
            "verdict": verdict("mos", mos),
            "method": ", ".join(quality.get("methods", [])) or "not run",
        },
        {
            "requirement": "Voice cloning similarity > 85%",
            "measured": sim,
            "verdict": verdict("similarity", sim),
            # When a score exists but is inadmissible, say so and quote it,
            # rather than presenting the method as though it counted.
            "method": (
                similarity.get("method", "not run")
                if sim is not None
                else _inadmissible_method(similarity)
            ),
        },
        {
            "requirement": "API initiation response < 500 ms",
            "measured": initiation_ms,
            "verdict": verdict("api_initiation_ms", initiation_ms),
            "method": initiation.get("reason", f"queue={api.get('queueMode', 'n/a')}")
            if isinstance(initiation, dict) else "not run",
        },
        {
            "requirement": "API capacity 100+ requests/minute",
            "measured": api.get("effectiveCapacityRpm"),
            "verdict": verdict("throughput_rpm", api.get("effectiveCapacityRpm")),
            "method": (
                f"min(rate-limit policy {api.get('configuredLimitRpm')} rpm, "
                f"raw app throughput {api.get('rawThroughputRpm')} rpm over "
                f"{api.get('throughputSampleSize', 0)} sequential in-process "
                "requests); not a deployed-server load test"
            ),
        },
        {
            "requirement": "Warm synthesis latency < 2 s",
            "measured": warm,
            "verdict": verdict("warm_synthesis_ms", warm),
            "method": "median of repeated Kokoro runs, model resident",
        },
    ]
    return rows


def render_markdown(report: dict) -> str:
    lines = [
        "# Phase 3 Benchmark Report",
        "",
        f"Generated: {report['generatedAt']}",
        f"Device: {report['device']}  |  Queue: {report.get('api', {}).get('queueMode', 'n/a')}",
        "",
        "## Acceptance matrix",
        "",
        "| Requirement | Measured | Verdict | Method |",
        "|---|---|---|---|",
    ]
    for row in report["matrix"]:
        measured = row["measured"]
        shown = "—" if measured is None else f"{measured:g}"
        lines.append(
            f"| {row['requirement']} | {shown} | **{row['verdict']}** | {row['method']} |"
        )

    synthesis = report.get("synthesis")
    if synthesis:
        warm = synthesis.get("english", {}).get("kokoro", {})
        lines += [
            "",
            "## Synthesis latency",
            "",
            f"- Kokoro cold start: {warm.get('coldMs', '—')} ms",
        ]
        stats = warm.get("warm", {})
        if stats:
            lines += [
                f"- Kokoro warm median: {stats.get('medianMs')} ms "
                f"(p95 {stats.get('p95Ms')} ms, n={stats.get('count')})",
                f"- Real-time factor: {stats.get('realTimeFactor')} "
                f"({stats.get('audioSecondsGenerated')} s of audio generated)",
            ]
        multilingual = synthesis.get("multilingual", {})
        if multilingual:
            lines += ["", "| Language | Model | Cold ms | Warm ms | Duration s |", "|---|---|---|---|---|"]
            for iso3, row in multilingual.items():
                if "error" in row:
                    lines.append(f"| {iso3} | — | — | — | {row['error']} |")
                else:
                    lines.append(
                        f"| {iso3} ({row.get('languageName')}) | {row.get('model')} | "
                        f"{row.get('coldMs')} | {row.get('warmMs')} | {row.get('durationSeconds')} |"
                    )

    emotions = report.get("emotions")
    if emotions:
        lines += [
            "",
            "## Emotion prosody",
            "",
            "| Emotion | Duration ratio | Expected | Loudness ratio | Expected |",
            "|---|---|---|---|---|",
        ]
        for name, row in emotions["emotions"].items():
            lines.append(
                f"| {name} | {row['durationRatio']} | {row['expectedRateInverse']} | "
                f"{row['loudnessRatio']} | {row['expectedEnergy']} |"
            )

    quality = report.get("quality")
    if quality:
        lines += [
            "",
            "## Speech quality (torchaudio SQUIM)",
            "",
            f"- Mean MOS: **{quality.get('meanMos')}** (min {quality.get('minMos')}) "
            f"over {len(quality.get('perSentence', []))} sentences",
            f"- Mean PESQ: {quality.get('meanPesq')}",
            f"- Method: {', '.join(quality.get('methods', []))}",
        ]

    similarity = report.get("similarity")
    if similarity:
        lines += ["", "## Voice cloning similarity", ""]
        if similarity.get("status") in ("SKIPPED", "FAILED"):
            lines.append(f"- {similarity.get('status')}: {similarity.get('reason')}")
        else:
            admissible = similarity.get("admissible")
            headline = "Similarity" if admissible else "Similarity (NOT evidence)"
            lines += [
                f"- {headline}: **{similarity.get('similarityPercent')}%**",
                f"- Method: {similarity.get('method')}",
                f"- Reference: {similarity.get('referenceFile')} "
                f"({similarity.get('referenceSeconds')} s)",
                f"- Provenance: {similarity.get('reason')}",
            ]
            if not admissible:
                lines.append(
                    "- This run exercised the cloning pipeline; the score above "
                    "does **not** satisfy the >85% requirement. Supply a human "
                    "reference with a recorded consent basis to measure it."
                )
            for warning in similarity.get("warnings", []):
                lines.append(f"- ⚠️ {warning}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-synthesis", action="store_true")
    parser.add_argument("--skip-emotions", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-similarity", action="store_true")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--api-requests", type=int, default=120)
    args = parser.parse_args()

    from voice_engine import VoiceEngineRouter

    router = VoiceEngineRouter()
    report: dict = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": router.device,
    }

    if not args.skip_synthesis:
        print("[1/5] synthesis latency ...", flush=True)
        report["synthesis"] = benchmark_synthesis(router, repeats=args.repeats)
    if not args.skip_emotions:
        print("[2/5] emotion prosody ...", flush=True)
        report["emotions"] = benchmark_emotions(router)
    if not args.skip_quality:
        print("[3/5] speech quality (SQUIM) ...", flush=True)
        report["quality"] = benchmark_quality(router, router.auditor)
    if not args.skip_similarity:
        print("[4/5] voice cloning similarity ...", flush=True)
        report["similarity"] = benchmark_similarity(router, router.auditor)
    if not args.skip_api:
        print("[5/5] API latency and throughput ...", flush=True)
        report["api"] = benchmark_api(requests_count=args.api_requests)

    report["matrix"] = build_matrix(report)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    json_path = REPORT_DIR / f"phase3_benchmark_{stamp}.json"
    md_path = REPORT_DIR / "phase3_benchmark.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print("\n" + render_markdown(report))
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
