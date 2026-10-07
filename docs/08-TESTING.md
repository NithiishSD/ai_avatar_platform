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
| R-01 | `test_voice_engine.py`, `test_mms_engine.py`, `test_openvoice_engine.py`, `test_bark_engine.py`, `UnrunnableEngineTests` | partial (**4** usable: Kokoro, MMS-TTS, OpenVoice V2, Bark; XTTS-v2 = 5th, waiting on M-03; Higgs and Dia cannot run on this stack, D-32) |
| R-02 | `test_voice_engine.py` routing + `MissingWeightsTests`; `test_app.py` 503/400 before queueing | pass |
| R-03 | OpenVoice V2 live clone through the API (T2.6b); XTTS-v2 **blocked** (M-03) | pass (OpenVoice); XTTS gap |
| R-04 | OpenVoice V2 clone in Hindi over MMS-TTS, live (T2.6b); XTTS cross-lingual in T3.3 | partial |
| R-05 | `test_alignment_engine.py`, `test_alignment_method.py`, `test_alignment_accuracy.py`; E2E `synthesis.spec.js` (measured, not estimated) | pass |
| R-06 | `test_mms_engine.py`, `test_language_registry.py`, `test_romanizer.py` | pass |
| R-07 | `test_emotion_engine.py` | pass |
| R-08 | `test_quality_auditor.py` (incl. partial-SQUIM fallback) | pass |
| R-09 | `test_voice_engine.py` prosody tests | pass |
| R-10 | `test_face_engine.py`; E2E `avatar.spec.js` (mesh drawn inside the face box, measured from canvas pixels); pose signs measured on real MediaPipe (T1.3) + `PoseFallbackTests` | pass |
| R-11 | `test_face_quality.py` | pass |
| R-12 | `test_render_engine.py`, `test_app_vision.py`; E2E `clone.spec.js`; live: cloned voice → Wav2Lip video through the API (`clone_to_video.py`, T2.4) | pass |
| R-13 | `test_render_engine.py` engine selection; live: blendshape and Wav2Lip on the same 3 jobs (T2.3) | pass |
| R-14 | `test_viseme_blendshapes.py`, `test_face_animation.py` | pass |
| R-15 | `test_avatar_generator.py` (prompt choices, registration, id clash), `test_app_vision.py` `AvatarGenerateRouteTests` (202/409/422/503/404, mocked SD); E2E `generate.spec.js`; live SD 1.5 through the API (T3.2) | pass |
| R-16 | `test_render_engine.py` (`BackgroundSpecTests`, `BackgroundRenderTests`), `test_app_vision.py` (400 without segmenter, 422 malformed); E2E `background.spec.js` (decoded video corners are the chosen colour); live on the real segmenter (T3.1) | pass |
| R-17 | `test_lipsync_metric.py` | pass |
| R-18 | T4.1 | gap |
| R-19 | T4.2 | gap |
| R-20 | `test_job_queue.py`, `test_app.py` | pass |
| R-21 | T6.1 | gap |
| R-22 | `test_security.py` | pass |
| R-23 | `test_model_registry.py`, `test_app.py` health; live: /health lists higgs/dia unavailable (T1.4) | pass |
| R-24 | E2E `smoke.spec.js` (T0.6), `synthesis.spec.js` (T1.1), `avatar.spec.js` (T1.2), `clone.spec.js` (T2.5: clone → render → video decodes). Still to come: T3.4, T4.3 | partial |
| R-25 | `test_sdk.py` (10: polling, eager answer, failure and missing-timing errors, job validated against the real `AvatarRenderJob`, error detail + request id, 429 retry and give-up, headers); live against a real server (T7.1) | pass |
| R-26 | T7.2 | gap |
| R-27 | `test_request_context.py` (id on every response, safe caller id kept, unsafe one replaced, log lines stamped, no leak past the request, worker thread inherits the id, 500 names the id); live on a real server (T6.2) | pass |
| R-30 | `test_provenance.py`, `test_avatar_store.py`, `VoiceConsentTests` (voices now checked too) | pass |
| R-31 | `VoiceConsentTests` (8), `test_app.py` 403/400/202; live on a real server (T2.2) | pass |
| R-32 | T5.1 | gap |
| R-33 | T5.2 | gap |
| R-34 | `test_render_engine.py` label | pass |
| R-35 | T5.4 | gap |
| R-36 | T5.3 | gap |
| N-01 | 17 Sep: MOS 4.33 (SQUIM, Kokoro). 8 Oct: Bark dialogue 4.12 (SQUIM, self-referenced, biased up). Re-run with a non-matching reference in T6.6 | measured — met |
| N-02 | 8 Oct, ECAPA-TDNN vs held-out LJSpeech (admissible), 6 sentences: **OpenVoice V2 34.2%** (base 26.9%, real speech 91.7%). XTTS-v2 not measurable (M-03) | **not met** (OpenVoice) |
| N-03 | in-process 13 ms (17 Sep); real server in T6.3 | partial |
| N-04 | in-process only; real server in T6.3 | gap |
| N-05 | 8 Oct, SyncNet v2, 25 fps: **Wav2Lip** offset 0 on 3/3 clips, LSE-C 9.84 / 10.51 / 11.18, LSE-D 6.61 / 5.97 / 5.48; blendshape offset 0, LSE-C 3.98 / 4.23 / 5.55 (T2.3). The D-12 window percentage not yet computed (T6.6) | measured — % figure pending |
| N-06 | T6.6 | gap |
| N-07 | T4.4 | gap |
| N-08 | T6.4 | gap |
| N-09 | 8 Oct, `jitter_metric.py` (static anchor landmarks per frame, % of inter-ocular distance): `demo` blendshape render mean **0.278%**, p95 0.758%, max 0.963% (target < 2%); negative control through the real detector: still photo 0.0%, 5 px shake 3.59%. Unit: `test_jitter_metric.py` | measured — met (upper bound, see method) |
| N-10 | not measurable here (no CUDA in sandbox); host measurement requested (M-04) | gap |
| N-11 | needs a deployment (Q-07) | gap |
