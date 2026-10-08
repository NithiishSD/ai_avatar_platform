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
  --skip-emotions    whether each emotion preset changes duration and loudness as intended

Honesty rules this harness follows:
  * Every number records the method that produced it.
  * A threshold is only marked PASS when the measurement that backs it is the
    method the assignment names (e.g. ECAPA-TDNN for cloning similarity).
  * API initiation latency is meaningless while the queue runs in eager mode,
    because the POST then performs the synthesis inline. The harness detects
    that and says so instead of reporting a misleading number.

What it writes:
  * ``docs/benchmarks/phase3_benchmark_<YYYYMMDD>.json`` - every raw number, one file per day.
  * ``docs/benchmarks/phase3_benchmark.md`` - the readable summary, overwritten each run.
  * ``outputs/benchmark/*.wav`` - the audio it synthesised, kept so a person can listen.

Concepts used here, explained once:

**Cold vs warm latency.** The first call to a model pays for loading the weights (cold). Later
calls reuse the loaded model (warm). Users mostly see warm latency, so that is what the
threshold is judged on, but the cold number is kept because it is what a restart costs.

**Real-time factor (RTF)** is seconds spent generating divided by seconds of audio produced.
Below 1.0 the engine is faster than real time; 0.1 means one second of speech takes 0.1 s.

**Percentiles.** The median (p50) is the middle sample; p95 is the value that 95% of samples
fall at or below. A mean hides a few slow outliers; p95 shows them, and the slow tail is what
users notice.

**MOS / PESQ / STOI / SI-SDR.** MOS (mean opinion score, 1 to 5) is how good listeners rate a
clip. PESQ and STOI are standard quality and intelligibility scores; SI-SDR measures signal
against distortion. Here all four are *predicted* by torchaudio's SQUIM models rather than
collected from people (see ``backend/quality_auditor.py``).

**ECAPA-TDNN** is a speaker-recognition network that maps a recording to a voice vector.
Similarity of two vectors says how alike two voices are; it is the method the assignment names.

**Eager mode.** With ``QUEUE_BACKEND=in_memory`` the task queue runs each task immediately
inside the request instead of handing it to a worker, so "time to accept the job" and "time to
do the job" become the same number.

How to read the report: the acceptance matrix at the top is the answer. Each row says PASS,
FAIL or NOT MEASURED and names the method behind it. NOT MEASURED is an honest result, not a
bug: it means no admissible measurement exists yet, and the method column says what is missing.
The sections below the matrix hold the raw numbers the verdicts were taken from.

Runs on CPU or GPU, whichever the voice router selects; the device is recorded in the report
because a CPU latency must not be compared with a GPU threshold.
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
# Only the standard library is imported at module level. The heavy imports (numpy, torch via
# the engines, FastAPI) happen inside the functions that need them, so a skipped section never
# pays for loading them.

# A type variable: lets timed() return whatever the timed function returns,
# instead of erasing it to `object` and making every caller cast.
T = TypeVar("T")

# The repository root (this file is scripts/benchmark_phase3.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Put backend/ first on the import path so its modules import by name, as they do in the server.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Reports go under docs/ because they are evidence that is read and cited; the audio goes under
# outputs/, which holds generated files.
REPORT_DIR = PROJECT_ROOT / "docs" / "benchmarks"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmark"

# Phonetically varied sentences, so MOS is not measured on one lucky utterance.
# They mix tongue-twister sibilants ("sea shells"), numbers spoken as words and dates, which are
# common places for a TTS engine to stumble. SENTENCES[0] is reused as the fixed sentence for
# the cold-start, emotion and cloning measurements, so those compare like with like.
SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "She sells sea shells by the sea shore every single summer morning.",
    "Our quarterly revenue grew by thirty seven percent across all regions.",
    "Please confirm whether the shipment arrived before the eleventh of March.",
    "Thick fog rolled through the valley while the church bells rang loudly.",
]

# (ISO 639-3 code, sentence) pairs: Hindi, Tamil, Swahili, Spanish. Three scripts and two
# lower-resource languages, so the multilingual path is not only tested on Latin text.
MULTILINGUAL_CASES = [
    ("hin", "नमस्ते, मैं आपका कृत्रिम बुद्धिमत्ता अवतार हूँ।"),
    ("tam", "வணக்கம், நான் உங்கள் செயற்கை நுண்ணறிவு அவதாரம்."),
    ("swh", "Habari, mimi ni avatar yako ya akili bandia."),
    ("spa", "Hola, soy tu avatar de inteligencia artificial."),
]

# Thresholds from the assignment. Each value is (label, threshold, direction): "greater" means
# the measurement must be above the threshold to pass, "less" means below. Latencies are in ms.
TARGETS = {
    "mos": ("MOS", 3.5, "greater"),
    "similarity": ("Voice cloning similarity", 0.85, "greater"),
    "api_initiation_ms": ("API initiation response", 500.0, "less"),
    "throughput_rpm": ("API capacity", 100.0, "greater"),
    "warm_synthesis_ms": ("Warm synthesis latency", 2000.0, "less"),
}


def timed(fn: Callable[[], T]) -> tuple[float, T]:
    """
    Call ``fn`` once and return ``(elapsed_ms, its_return_value)``.

    ``fn`` takes no arguments, so callers wrap the real call in a ``lambda``. Returning the
    value as well as the time means a measured call does not have to be made twice.
    """
    # perf_counter() is the highest-resolution clock and never jumps backwards, which makes it
    # the right clock for durations (time.time() can be adjusted mid-measurement).
    start = time.perf_counter()
    value = fn()
    return (time.perf_counter() - start) * 1000.0, value


def summarize(samples: list[float]) -> dict:
    """
    Reduce latency samples (ms) to count, mean, median, p95, min and max.

    Returns an empty dict for no samples, so a section that measured nothing reports nothing
    rather than a misleading zero.
    """
    if not samples:
        return {}
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "meanMs": round(statistics.fmean(ordered), 2),
        "medianMs": round(statistics.median(ordered), 2),
        # Nearest-rank p95: the sample 95% of the way up the sorted list. int() truncates, so the
        # index is already below len(); min() is a guard that keeps it in range regardless.
        "p95Ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
        "minMs": round(ordered[0], 2),
        "maxMs": round(ordered[-1], 2),
    }


def verdict(key: str, value: Optional[float]) -> str:
    """
    PASS, FAIL or NOT MEASURED for one threshold in ``TARGETS``.

    ``None`` means "no admissible measurement", which is reported as NOT MEASURED and never
    as a pass (golden rule 8: say what is not done).
    """
    if value is None:
        return "NOT MEASURED"
    # Strict comparisons: the assignment says "> 3.5" and "< 500 ms", so equality fails.
    _, threshold, direction = TARGETS[key]
    ok = value > threshold if direction == "greater" else value < threshold
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def benchmark_synthesis(router, repeats: int = 3) -> dict:
    """
    Cold vs warm latency, and the real-time factor, for each backend.

    English goes through ``mode="fast"`` (Kokoro); the non-English cases go through
    ``mode="multilingual"``, where the router picks the engine per language (MMS-TTS first).
    ``result.model`` records which engine actually answered, so a fallback is visible.
    """
    # parents=True creates missing parent folders; exist_ok=True makes a second run a no-op.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"english": {}, "multilingual": {}}

    # Cold: the first synthesis also loads Kokoro (the "fast" mode engine), so it is timed alone.
    cold_ms, result = timed(
        lambda: router.synthesize(
            SENTENCES[0], mode="fast",
            output_filename="benchmark/kokoro_cold.wav",
        )
    )
    # `model` comes from the result, not from an assumption, so a silent fallback would show up.
    results["english"]["kokoro"] = {
        "coldMs": round(cold_ms, 2),
        "model": result.model,
    }

    # Warm: every sentence, `repeats` times, with the model already resident.
    warm_samples = []
    audio_seconds = 0.0
    for index in range(repeats):
        for sentence in SENTENCES:
            # `s=sentence, i=index` binds the current loop values as default arguments. A lambda
            # that read `sentence` directly would see the variable, not its value at this
            # iteration; that only bites if the call is deferred, but ruff's B023 rule flags it
            # either way, and this form is correct in both cases.
            elapsed, result = timed(
                lambda s=sentence, i=index: router.synthesize(
                    s, mode="fast",
                    output_filename=f"benchmark/kokoro_warm_{i}.wav",
                )
            )
            warm_samples.append(elapsed)
            # Summed so the real-time factor below covers the whole run, not one sentence.
            audio_seconds += result.duration_seconds

    # summarize() gives the latency distribution; the RTF fields are added to the same dict.
    stats = summarize(warm_samples)
    total_ms = sum(warm_samples)
    # RTF = generation seconds / audio seconds. max(..., 1e-6) avoids dividing by zero if every
    # synthesis somehow returned empty audio.
    stats["realTimeFactor"] = round((total_ms / 1000.0) / max(audio_seconds, 1e-6), 4)
    stats["audioSecondsGenerated"] = round(audio_seconds, 2)
    # update() merges the warm stats in next to the cold number already stored.
    results["english"]["kokoro"].update({"warm": stats})

    # Each language is timed twice: the first call may load a per-language checkpoint (cold),
    # the second reuses it (warm). Writing both to the same file is fine; only timings are kept.
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
            # The cold call's result describes the run; the warm call's is discarded (`_`).
            results["multilingual"][iso3] = {
                "model": result.model,
                # `(x or {})` lets .get() run even when the engine returned no language info.
                "languageName": (result.language or {}).get("name"),
                "coldMs": round(cold_ms, 2),
                "warmMs": round(warm_ms, 2),
                "durationSeconds": round(result.duration_seconds, 2),
                "sampleRate": result.sample_rate,
            }
        # One failing language must not abort the whole benchmark; the error is recorded in its
        # row instead, where the report shows it.
        except Exception as exc:  # noqa: BLE001 - a failed language is data too
            results["multilingual"][iso3] = {"error": f"{type(exc).__name__}: {exc}"}

    return results


def benchmark_emotions(router) -> dict:
    """
    Confirm each emotion changes duration and loudness in its stated direction.

    Each preset declares a speaking ``rate`` and an ``energy`` gain (``backend/emotion_engine.py``).
    A rate of 1.1 should make the same sentence about 1/1.1 as long as neutral, and an energy
    of 1.15 should make it about 1.15 times as loud. The report puts measured and expected ratios
    side by side; it does not pass or fail them.
    """
    # Imported here, not at the top, so --help and skipped sections stay fast.
    import numpy as np
    import soundfile as sf

    # PRESETS maps an emotion name to its prosody (pitch, rate, energy) and face settings.
    from emotion_engine import PRESETS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # The neutral rendering of the fixed sentence is the yardstick every emotion is divided by.
    baseline = router.synthesize(
        SENTENCES[0], mode="fast", output_filename="benchmark/emotion_neutral.wav"
    )
    # output_path is relative to the project root. sf.read returns (samples, sample_rate); only
    # the samples are needed. float32 gives samples in [-1, 1] whatever the file's bit depth.
    reference, _ = sf.read(str(PROJECT_ROOT / baseline.output_path), dtype="float32")
    # RMS (root mean square) of the samples is a simple loudness measure: square, average, root.
    base_rms = float(np.sqrt(np.mean(reference**2)))
    base_duration = baseline.duration_seconds

    rows = {}
    for name in PRESETS:
        # Neutral is the baseline itself; its ratios would be 1.0 by definition.
        if name == "neutral":
            continue
        result = router.synthesize(
            SENTENCES[0], mode="fast", emotion=name,
            output_filename=f"benchmark/emotion_{name}.wav",
        )
        # Same loudness measure as the baseline, so the ratio compares like with like.
        audio, _ = sf.read(str(PROJECT_ROOT / result.output_path), dtype="float32")
        rms = float(np.sqrt(np.mean(audio**2)))
        # Ratios against neutral, so the numbers are independent of the sentence and voice.
        rows[name] = {
            "durationRatio": round(result.duration_seconds / base_duration, 3),
            # Duration scales with the inverse of rate: faster speech is shorter.
            "expectedRateInverse": round(1.0 / PRESETS[name].prosody.rate, 3),
            "loudnessRatio": round(rms / base_rms, 3),
            "expectedEnergy": PRESETS[name].prosody.energy,
            "latencyMs": round(result.latency_ms, 1),
        }
    # The baseline length is kept so the ratios can be turned back into seconds if needed.
    return {"baselineSeconds": round(base_duration, 3), "emotions": rows}


def benchmark_quality(router, auditor) -> dict:
    """
    SQUIM MOS / PESQ / STOI across the sentence set.

    ``isModelPrediction`` is True only when every clip was scored by the real SQUIM model.
    ``build_matrix`` admits the mean MOS only in that case, so a fallback estimate can never
    pass the MOS threshold.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # A non-matching reference removes the self-reference bias in SQUIM's
    # subjective head.
    reference = router.synthesize(
        "This recording is the non matching reference clip for scoring.",
        mode="fast", output_filename="benchmark/quality_reference.wav",
    )
    # One reference for all five clips, so every clip is scored against the same yardstick.
    reference_path = PROJECT_ROOT / reference.output_path

    rows: list[dict[str, Any]] = []
    # enumerate() yields (index, item) pairs; the index gives each clip its own file name.
    for index, sentence in enumerate(SENTENCES):
        result = router.synthesize(
            sentence, mode="fast", output_filename=f"benchmark/quality_{index}.wav"
        )
        # audit() runs SQUIM on the clip and returns a report whose `method` names the scorer.
        report = auditor.audit(
            PROJECT_ROOT / result.output_path, reference_path=reference_path
        )
        rows.append(
            {
                "sentence": sentence,
                "model": result.model,
                # `**{...}` unpacks a dict comprehension into this dict: copy just these five
                # keys from the audit report.
                **{
                    key: report.to_dict()[key]
                    for key in ("mos", "pesq", "stoi", "siSdr", "method")
                },
            }
        )

    # A clip may have no MOS (for example if SQUIM failed to load); those are left out of the
    # mean rather than counted as zero.
    scored = [row["mos"] for row in rows if row["mos"] is not None]
    # A set: if every row used the same method it collapses to one entry, which is checked below.
    methods = {row["method"] for row in rows}
    return {
        "perSentence": rows,
        "meanMos": round(statistics.fmean(scored), 3) if scored else None,
        "minMos": round(min(scored), 3) if scored else None,
        # The conditional expression guards fmean, which raises on an empty list.
        "meanPesq": round(
            statistics.fmean([r["pesq"] for r in rows if r["pesq"] is not None]), 3
        ) if any(r["pesq"] is not None for r in rows) else None,
        "methods": sorted(methods),
        # Only a real model prediction counts as evidence.
        "isModelPrediction": methods == {"torchaudio-squim"},
    }


def benchmark_similarity(router, auditor) -> dict:
    """
    ECAPA speaker similarity between a reference recording and its clone.

    Status values: SKIPPED (no reference recording), FAILED (cloning raised), MEASURED (an
    admissible reference) or PIPELINE_TEST (a number exists but the reference may not back
    the published threshold, for example a synthetic clip).
    """
    from audio_utils import list_voice_samples

    # Recordings found in inputs/, each with its path, filename and duration. Unreadable files
    # are skipped with a warning, and a missing folder gives an empty list.
    samples = list_voice_samples(PROJECT_ROOT / "inputs")
    # No reference is a normal state on a fresh checkout, not an error: report SKIPPED and say
    # what to add (golden rule 7).
    if not samples:
        return {
            "status": "SKIPPED",
            "reason": (
                "No reference recording in inputs/. Add 30-60s of consented "
                "speech to measure cloning similarity."
            ),
        }

    # provenance reads and validates the consent sidecar stored next to each recording.
    import provenance

    # Prefer a reference that may actually back the published threshold, so a
    # synthetic smoke-test clip sitting in inputs/ never shadows a real one.
    # provenance.describe reads each recording's sidecar and says whether it is admissible.
    described = [(s, provenance.describe(s.path)) for s in samples]
    admissible = [pair for pair in described if pair[1]["admissible"]]
    # Fall back to the first recording if none is admissible: the pipeline still gets
    # exercised, and the status below marks the score as a pipeline test, not evidence.
    reference, prov = admissible[0] if admissible else described[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # mode="clone" synthesises the fixed sentence in the reference speaker's voice.
    try:
        cloned = router.synthesize(
            SENTENCES[0], mode="clone", speaker_wav=reference.path,
            output_filename="benchmark/clone.wav",
        )
    # Cloning engines fail in many ways (missing weights, licence not accepted); any of them is
    # reported as FAILED with its type and message rather than crashing the other sections.
    except Exception as exc:  # noqa: BLE001
        return {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}

    # Compare the clone with the very recording it was cloned from.
    report = auditor.speaker_similarity(
        reference.path, PROJECT_ROOT / cloned.output_path
    )
    # to_dict() gives the camelCase fields (similarity, similarityPercent, method, isEcapa...)
    # that build_matrix and render_markdown read.
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
        # Listed last so it replaces the "warnings" key copied in by **measured, adding the
        # reference's provenance warnings to the auditor's own.
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

    ``TestClient`` (from FastAPI, built on httpx) calls the ASGI app directly in this process:
    no socket, no network. That makes it a clean measure of the application code and an upper
    bound on what a real server over a network would do.
    """
    import os

    from fastapi.testclient import TestClient

    # Imported as a module object, not `from app import ...`, so its `security_gate` attribute
    # can be replaced below and the app sees the replacement.
    import app as app_module
    from security import SecurityConfig, SecurityGate

    # Anything other than "celery" runs tasks inline (eager mode, see the module docstring).
    eager = os.getenv("QUEUE_BACKEND", "in_memory").lower() != "celery"
    configured = app_module.security_gate.config
    # Any values: the report mixes flags, counts, nested summaries and floats.
    result: dict[str, Any] = {
        "queueMode": "eager (in_memory)" if eager else "celery",
        "configuredLimitRpm": (
            # None when the limiter is off: there is then no policy ceiling to report.
            configured.requests_per_minute if configured.rate_limit_enabled else None
        ),
        "rateLimitEnabled": configured.rate_limit_enabled,
        "authEnabled": configured.auth_enabled,
    }

    # Measure the application itself with the limiter out of the way, then put
    # the real gate back. Benchmarking through the limiter would only measure
    # the limiter.
    # The app reads the module attribute `security_gate` on each request, so replacing the
    # attribute swaps the gate for the running app. try/finally below guarantees the real gate
    # is put back even if a request raises.
    original_gate = app_module.security_gate
    app_module.security_gate = SecurityGate(
        SecurityConfig(
            auth_enabled=False,
            api_keys=frozenset(),
            rate_limit_enabled=False,
            # Copied from the real config so the only difference is that both checks are off.
            requests_per_minute=configured.requests_per_minute,
            burst=configured.burst,
        )
    )
    try:
        # A TestClient behaves like an HTTP client (client.get, client.post, response.status_code)
        # but runs the request through the app in-process.
        client = TestClient(app_module.app)

        # Pass 1, per-request latency on a cheap read endpoint. `_` is the convention for a loop
        # variable that is not used. Only 200s are kept so errors do not pose as fast responses.
        light = []
        for _ in range(requests_count):
            # The languages listing does no synthesis, so this times the framework, routing and
            # middleware rather than a model.
            elapsed, response = timed(
                lambda: client.get("/api/v1/audio/languages?limit=5")
            )
            if response.status_code == 200:
                light.append(elapsed)

        # Pass 2, throughput: one clock around the whole batch, counting successful responses.
        # Measured separately because per-call timing adds its own overhead.
        start = time.perf_counter()
        completed = 0
        for _ in range(requests_count):
            if client.get("/api/v1/audio/languages?limit=20").status_code == 200:
                completed += 1
        # Total wall time for the batch, in seconds.
        window = time.perf_counter() - start

        result["readEndpoint"] = summarize(light)
        # Requests per second times 60. The guard covers a zero-length window.
        result["rawThroughputRpm"] = (
            round(completed / window * 60.0, 1) if window > 0 else 0.0
        )
        result["throughputSampleSize"] = completed
        # The caveat travels with the number into the JSON, so it is never quoted without it.
        result["throughputCaveat"] = (
            "Sequential in-process ASGI calls with no network, TLS or "
            "concurrency. An upper bound on the application layer; a load test "
            "against a running uvicorn with concurrent clients is still "
            "outstanding."
        )

        # Initiation latency is only meaningful when the POST hands off to a worker.
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
            # Ten submissions, cycling through the sentence set. 202 Accepted is the normal
            # answer for a queued job; 200 is accepted too.
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
            # Time from POST to "job accepted"; the synthesis itself happens later on the worker.
            result["initiation"] = summarize(samples)
    # Runs on success and on exception alike: the process must not be left with security off.
    finally:
        app_module.security_gate = original_gate

    # The capacity a client actually gets is the lower of policy and headroom.
    # `or 0.0` turns a missing value into 0 so min() below never compares None.
    served = result.get("rawThroughputRpm") or 0.0
    limit = result.get("configuredLimitRpm")
    # With no rate limit configured, the application's own speed is the only ceiling.
    result["effectiveCapacityRpm"] = min(served, limit) if limit else served
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _inadmissible_method(similarity: dict) -> str:
    """
    Explain why a similarity score is not counted, quoting it if one exists.

    Used for the matrix's method column when the similarity row has no admissible number.
    Quoting the indicative score keeps the information without letting it count as a pass.
    """
    # An empty dict means the similarity section was skipped.
    if not similarity:
        return "not run"
    # `or` also replaces an empty-string reason, not just a missing key.
    reason = similarity.get("reason") or "no admissible reference"
    score = similarity.get("similarityPercent")
    # admissible defaults to True so a report without that key is not called inadmissible.
    if score is not None and not similarity.get("admissible", True):
        return f"not counted ({reason}); indicative score was {score}%"
    return reason


def build_matrix(report: dict) -> list[dict]:
    """
    Map raw measurements onto the assignment's acceptance thresholds.

    Returns one row per threshold. A measurement that is not admissible evidence is passed to
    ``verdict`` as None, so it shows as NOT MEASURED even when a number was produced.
    """
    # `or {}` covers sections that were skipped (absent) as well as ones that returned None.
    quality = report.get("quality") or {}
    similarity = report.get("similarity") or {}
    api = report.get("api") or {}
    synthesis = report.get("synthesis") or {}

    # MOS counts only when every clip was scored by the real SQUIM model (golden rule 2).
    mos = quality.get("meanMos") if quality.get("isModelPrediction") else None
    # Median, not mean: one slow outlier should not decide the warm-latency verdict.
    # Chained .get() calls with {} defaults walk the nested dict without a KeyError when any
    # level is missing (for example when --skip-synthesis was used).
    warm = (
        synthesis.get("english", {}).get("kokoro", {}).get("warm", {}).get("medianMs")
    )
    initiation = api.get("initiation", {})
    # In eager mode "initiation" holds a status and reason with no medianMs, so this is None.
    # The isinstance check is defensive: only a dict has .get().
    initiation_ms = initiation.get("medianMs") if isinstance(initiation, dict) else None
    # Admissible evidence for the cloning threshold needs both the method the
    # assignment names (ECAPA-TDNN) and a reference whose provenance allows it
    # to be published. Synthetic references are measured but never counted.
    sim = (
        similarity.get("similarity")
        if similarity.get("isEcapa") and similarity.get("admissible")
        else None
    )

    # One dict per threshold, in the order the assignment lists them. Each row carries its
    # method, so a reader can judge every verdict without opening the JSON.
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
            # In eager mode the reason explains why there is no number; otherwise name the queue.
            "method": initiation.get("reason", f"queue={api.get('queueMode', 'n/a')}")
            if isinstance(initiation, dict) else "not run",
        },
        {
            # Judged on effective capacity: the lower of policy and measured application speed.
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
            # "model resident" is what makes this warm: the load cost was paid by the cold run.
            "method": "median of repeated Kokoro runs, model resident",
        },
    ]
    return rows


def render_markdown(report: dict) -> str:
    """
    Turn the report dict into the Markdown summary: the matrix first, then one section per
    benchmark that ran. Sections that were skipped are left out.
    """
    # Built as a list of lines and joined once at the end, which is simpler and cheaper than
    # repeated string concatenation.
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
    # A Markdown table is pipe-separated cells; the "|---|" line above marks the header row.
    for row in report["matrix"]:
        measured = row["measured"]
        # A dash for "nothing admissible". The `g` format drops trailing zeros (3.50 -> 3.5).
        shown = "—" if measured is None else f"{measured:g}"
        lines.append(
            f"| {row['requirement']} | {shown} | **{row['verdict']}** | {row['method']} |"
        )

    # Each optional section below follows the same pattern: read it with .get(), and append
    # lines only if that section ran.
    synthesis = report.get("synthesis")
    if synthesis:
        warm = synthesis.get("english", {}).get("kokoro", {})
        lines += [
            "",
            "## Synthesis latency",
            "",
            f"- Kokoro cold start: {warm.get('coldMs', '—')} ms",
        ]
        # {} when the report has no warm block, so a partial report still renders.
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
                # A failed language still gets a row, with its error in the last column, so a
                # gap in coverage is visible rather than silently missing.
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
        # Measured and expected columns sit side by side so drift is easy to spot by eye.
        for name, row in emotions["emotions"].items():
            lines.append(
                f"| {name} | {row['durationRatio']} | {row['expectedRateInverse']} | "
                f"{row['loudnessRatio']} | {row['expectedEnergy']} |"
            )

    quality = report.get("quality")
    if quality:
        # Bold (**...**) on the headline number; the method line says which model produced it.
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
        # An inadmissible score is still shown, but labelled so it cannot be read as evidence.
        lines += ["", "## Voice cloning similarity", ""]
        # SKIPPED and FAILED have a reason but no score, so they get one line.
        if similarity.get("status") in ("SKIPPED", "FAILED"):
            lines.append(f"- {similarity.get('status')}: {similarity.get('reason')}")
        else:
            admissible = similarity.get("admissible")
            # The label changes, not the number: the score is shown either way.
            headline = "Similarity" if admissible else "Similarity (NOT evidence)"
            lines += [
                f"- {headline}: **{similarity.get('similarityPercent')}%**",
                f"- Method: {similarity.get('method')}",
                f"- Reference: {similarity.get('referenceFile')} "
                f"({similarity.get('referenceSeconds')} s)",
                f"- Provenance: {similarity.get('reason')}",
            ]
            # Spell out what the number does not prove and how to get one that does.
            if not admissible:
                lines.append(
                    "- This run exercised the cloning pipeline; the score above "
                    "does **not** satisfy the >85% requirement. Supply a human "
                    "reference with a recorded consent basis to measure it."
                )
            for warning in similarity.get("warnings", []):
                lines.append(f"- ⚠️ {warning}")

    # A trailing newline, as POSIX text files and Markdown linters expect.
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the selected sections, build the acceptance matrix, write JSON and Markdown."""
    # argparse turns "--skip-synthesis" into the attribute `args.skip_synthesis`; store_true
    # flags default to False and become True when given.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-synthesis", action="store_true")
    parser.add_argument("--skip-emotions", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-similarity", action="store_true")
    parser.add_argument("--skip-api", action="store_true")
    # type=int makes argparse convert and validate the value; "abc" exits with a usage error.
    parser.add_argument("--repeats", type=int, default=3)
    # Requests per API pass (used for both the latency and the throughput pass); the default
    # matches benchmark_api's own.
    parser.add_argument("--api-requests", type=int, default=120)
    args = parser.parse_args()

    # Imported after parsing so `--help` answers without loading the voice stack.
    from voice_engine import VoiceEngineRouter

    # One router for every section: models it loads stay resident, so later sections are warm.
    router = VoiceEngineRouter()
    # The report is one dict that each section adds a key to. The UTC timestamp and device are
    # recorded first so every number can be traced to when and where it was measured.
    report: dict = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # "cuda" when torch sees a GPU, else "cpu"; the voice router chooses it at start-up.
        "device": router.device,
    }

    if not args.skip_synthesis:
        # flush=True prints the progress line immediately; sections take minutes, and buffered
        # output would otherwise appear only at the end.
        print("[1/5] synthesis latency ...", flush=True)
        report["synthesis"] = benchmark_synthesis(router, repeats=args.repeats)
    if not args.skip_emotions:
        print("[2/5] emotion prosody ...", flush=True)
        report["emotions"] = benchmark_emotions(router)
    if not args.skip_quality:
        print("[3/5] speech quality (SQUIM) ...", flush=True)
        # The router's own auditor, so SQUIM and ECAPA are loaded once and shared.
        report["quality"] = benchmark_quality(router, router.auditor)
    if not args.skip_similarity:
        print("[4/5] voice cloning similarity ...", flush=True)
        report["similarity"] = benchmark_similarity(router, router.auditor)
    # The API section runs last: it builds its own app client and does not need the router.
    if not args.skip_api:
        print("[5/5] API latency and throughput ...", flush=True)
        report["api"] = benchmark_api(requests_count=args.api_requests)

    # Built after every section has run; it reads whatever subset of sections exists.
    report["matrix"] = build_matrix(report)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    # The JSON is dated so earlier evidence is kept; the Markdown always shows the latest run.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    # Two runs on the same UTC day overwrite that day's JSON; the last run of the day is kept.
    json_path = REPORT_DIR / f"phase3_benchmark_{stamp}.json"
    md_path = REPORT_DIR / "phase3_benchmark.md"
    # ensure_ascii=False keeps the Hindi and Tamil text readable instead of \u escapes, which is
    # why the encoding is set explicitly.
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    # Print the summary as well, so a terminal run shows the result without opening a file.
    print("\n" + render_markdown(report))
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


# The guard means importing this module (for example from a test) does not start a benchmark.
# raise SystemExit(main()) passes main()'s return value out as the process exit code.
if __name__ == "__main__":
    raise SystemExit(main())
