# 08 — Testing

## Layers

| Layer | Tool | Where | Rule |
|---|---|---|---|
| Unit | `unittest` | `tests/test_*.py` | models mocked; no network, no downloads |
| API | FastAPI `TestClient` | `tests/test_app*.py` | real routing and validation, mocked engines |
| Live | real weights, real data | commands recorded in `12-PROGRESS.md` | once per feature |
| E2E | Playwright (Chromium) | `frontend/e2e/*.spec.js` | real backend + real dev server |
| Measurement | scripts | `scripts/benchmark_*.py` → `outputs/benchmarks/` | number + method + date |

Run: `scripts/check.sh` (unit), `scripts/check.sh all` (everything).
Every new rule gets a test that **rejects** a violating input. A concurrency
test is proven able to fail once by removing its lock.

## Requirement → test traceability

Status: **pass** (test exists and passes) · **measured** (number recorded) ·
**gap** (nothing yet) · **not met** (measured, below target).

| ID | Test / measurement | Status |
|---|---|---|
| R-01 | `test_voice_engine.py`, `test_mms_engine.py`; engine count in T2.6 | pass (3 engines with weights; target 5 — gap) |
| R-02 | `test_voice_engine.py` routing tests | pass |
| R-03 | T2.1 live clone | gap |
| R-04 | T3.3 | gap |
| R-05 | `test_alignment_engine.py`, `test_alignment_method.py`, `test_alignment_accuracy.py` | pass |
| R-06 | `test_mms_engine.py`, `test_language_registry.py` | pass |
| R-07 | `test_emotion_engine.py` | pass |
| R-08 | `test_quality_auditor.py` | pass |
| R-09 | `test_voice_engine.py` prosody tests | pass |
| R-10 | `test_face_engine.py`; T1.3 pose signs | pass (signs unverified — gap) |
| R-11 | `test_face_quality.py` | pass |
| R-12 | `test_render_engine.py`, `test_app_vision.py` | pass |
| R-13 | `test_render_engine.py` engine selection | pass |
| R-14 | `test_viseme_blendshapes.py`, `test_face_animation.py` | pass |
| R-15 | `test_avatar_generator.py`; API in T3.2 | partial (CLI only) |
| R-16 | T3.1 | gap |
| R-17 | `test_lipsync_metric.py` | pass |
| R-18 | T4.1 | gap |
| R-19 | T4.2 | gap |
| R-20 | `test_job_queue.py`, `test_app.py` | pass |
| R-21 | T6.1 | gap |
| R-22 | `test_security.py` | pass |
| R-23 | `test_model_registry.py`, `test_app.py` health | pass |
| R-24 | E2E in T0.6, T1.1, T1.2, T2.5, T3.4, T4.3 | gap |
| R-25 | T7.1 | gap |
| R-26 | T7.2 | gap |
| R-27 | T6.2 | gap |
| R-30 | `test_provenance.py`, `test_avatar_store.py` | pass |
| R-31 | T2.2 | gap |
| R-32 | T5.1 | gap |
| R-33 | T5.2 | gap |
| R-34 | `test_render_engine.py` label | pass |
| R-35 | T5.4 | gap |
| R-36 | T5.3 | gap |
| N-01 | benchmark 17 Sep 2026: MOS 4.33 (SQUIM, Kokoro) | measured — met (re-run in T6.6) |
| N-02 | T2.1 | gap |
| N-03 | in-process 13 ms (17 Sep); real server in T6.3 | partial |
| N-04 | in-process only; real server in T6.3 | gap |
| N-05 | blendshape LSE-C EN 2.8 / HI 4.7 / TA 6.8, offset 0 (30 Sep) | measured — mapping in D-12 |
| N-06 | T6.6 | gap |
| N-07 | T4.4 | gap |
| N-08 | T6.4 | gap |
| N-09 | T3.5 | gap |
| N-10 | not measurable here (no CUDA in sandbox) | gap |
| N-11 | needs a deployment (Q-07) | gap |
