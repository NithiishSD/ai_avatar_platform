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
| R-01 | `test_voice_engine.py`, `test_mms_engine.py`, `test_openvoice_engine.py`, `test_bark_engine.py`, `UnrunnableEngineTests`; live XTTS-v2 clone (T2.1) | pass (**5** usable, each synthesised live: Kokoro, MMS-TTS, OpenVoice V2, Bark, XTTS-v2; Higgs and Dia cannot run on this stack, D-32) |
| R-02 | `test_voice_engine.py` routing + `MissingWeightsTests`; `test_app.py` 503/400 before queueing | pass |
| R-03 | OpenVoice V2 live clone through the API (T2.6b); XTTS-v2 live clone, 6 sentences, `measure_clone_similarity.py` (T2.1) | pass |
| R-04 | XTTS-v2 cross-lingual, live: Spanish, Hindi, French (T3.3); OpenVoice V2 in Hindi over MMS-TTS (T2.6b) | pass (speaks the language, voice carries over only partly: see N-02 note) |
| R-05 | `test_alignment_engine.py`, `test_alignment_method.py`, `test_alignment_accuracy.py`; E2E `synthesis.spec.js` (measured, not estimated) | pass |
| R-06 | `test_mms_engine.py`, `test_language_registry.py`, `test_romanizer.py` | pass |
| R-07 | `test_emotion_engine.py` | pass |
| R-08 | `test_quality_auditor.py` (incl. partial-SQUIM fallback) | pass |
| R-09 | `test_voice_engine.py` prosody tests | pass |
| R-10 | `test_face_engine.py`; E2E `avatar.spec.js` (mesh drawn inside the face box, measured from canvas pixels); pose signs measured on real MediaPipe (T1.3) + `PoseFallbackTests` | pass |
| R-11 | `test_face_quality.py` | pass |
| R-12 | `test_render_engine.py`, `test_app_vision.py`; E2E `clone.spec.js`; live: cloned voice → Wav2Lip video through the API (`clone_to_video.py`, T2.4) | pass |
| R-13 | `test_render_engine.py` engine selection; live: blendshape and Wav2Lip on the same 3 jobs (T2.3) | pass |
| R-14 | `test_viseme_blendshapes.py`, `test_face_animation.py`; E2E `gate3.spec.js` (custom avatar + cloned multilingual voice + emotion → preview render) | pass |
| R-15 | `test_avatar_generator.py` (prompt choices, registration, id clash), `test_app_vision.py` `AvatarGenerateRouteTests` (202/409/422/503/404, mocked SD); E2E `generate.spec.js`; live SD 1.5 through the API (T3.2) | pass |
| R-16 | `test_render_engine.py` (`BackgroundSpecTests`, `BackgroundRenderTests`), `test_app_vision.py` (400 without segmenter, 422 malformed); E2E `background.spec.js` (decoded video corners are the chosen colour); live on the real segmenter (T3.1) | pass |
| R-17 | `test_lipsync_metric.py` | pass |
| R-18 | `test_live.py` (26: sentence splitting, framing, event order and timeline, pipelining, early close, protocol, auth, session cap, consent, interrupt, malformed input, cleanup; **4 plants each caught**); live: `live_client.py` on a real server, audio chunks per sentence from real Kokoro (T4.1) | pass |
| R-19 | same file; live: 182 JPEG frames decoded and size-checked, face moving (123 of 181 consecutive frame pairs differ, mean change 0.58, max 2.88 grey levels), presentation times monotonic (T4.2) | pass |
| R-20 | `test_job_queue.py`, `test_app.py` | pass |
| R-21 | `test_job_store.py` (store round trip; finished job and result survive; mid-render job becomes FAILED with the restart reason; never-started job re-runs; ids stay unique; generation jobs likewise); live two-process restart (T6.1) | pass |
| R-22 | `test_security.py` | pass |
| R-23 | `test_model_registry.py`, `test_app.py` health; live: /health lists higgs/dia unavailable (T1.4) | pass |
| R-24 | E2E `smoke.spec.js` (T0.6), `synthesis.spec.js` (T1.1), `avatar.spec.js` (T1.2), `clone.spec.js` (T2.5), `background.spec.js` (T3.1), `generate.spec.js` (T3.2), `gate3.spec.js` (T3.4: generated avatar + XTTS-v2 Spanish clone + joy + new background → video whose corners are the new colour), `live.spec.js` (T4.3: a live session over a real WebSocket: 2 audio chunks, 60+ frames, the canvas changes while speaking, a second utterance works, interrupt stops the stream, stop closes) | pass |
| R-25 | `test_sdk.py` (10: polling, eager answer, failure and missing-timing errors, job validated against the real `AvatarRenderJob`, error detail + request id, 429 retry and give-up, headers); live against a real server (T7.1) | pass |
| R-26 | T7.2 | gap |
| R-27 | `test_request_context.py` (id on every response, safe caller id kept, unsafe one replaced, log lines stamped, no leak past the request, worker thread inherits the id, 500 names the id); live on a real server (T6.2) | pass |
| R-30 | `test_provenance.py`, `test_avatar_store.py`, `VoiceConsentTests` (voices now checked too) | pass |
| R-31 | `VoiceConsentTests` (8), `test_app.py` 403/400/202; live on a real server (T2.2) | pass |
| R-32 | `test_watermark.py` (26; real AudioSeal round trip when the weights are on disk; **4 plants caught**); `scripts/measure_watermark.py`, 8 Oct, CPU, 16 clips (8 human LJSpeech segments, 8 Kokoro sentences), method `torchaudio-squim` for quality: **mark is 26.6 dB below the speech (min 22.8)**; SQUIM MOS change **+0.06 mean (human +0.008, TTS +0.111; worst −0.003)**, PESQ estimate −0.067 (worst −0.277), STOI −0.001. **Detected after:** float32, 16-bit WAV, 24→8→24 kHz resample, AAC 128 kbps *and* 64 kbps, MP3 128, Opus 32, 30 dB noise, trim 0.7 s, volume ×0.3: **16/16 each**; 20 dB noise 13/16. **Lost after:** 10 dB noise 0/16, 4 kHz IIR low-pass 0/16, speed ×1.10 0/16. **False positives: 0/55** (8 human, 8 TTS, 8 noise, 6 tones, silence, and our 16 marked clips checked with another key); the bit rule alone admits a random message with probability 137/65536 = 0.2%; 0/55 only bounds the real rate under ~5% (95%). Not verified by a person: that it is inaudible (M-06) | measured; **inaudibility pending M-06** |
| R-33 | `test_video_watermark.py`, `test_manifest.py`, `test_render_engine.py` (`RenderWatermarkAndManifestTests`); **8 plants caught**; live: 128/128 tag bits read from the encoded MP4, manifest verifies, an edited field breaks the signature, a different video breaks the hash; strength sweep chosen from measured H.264 results (T5.2) | pass (look at the frames: M-07) |
| R-34 | `test_render_engine.py` label | pass |
| R-35 | `test_audit_log.py` (23: hash-chain tamper detection per field and for a deleted row, concurrent writers, filters, nearest-id lookup, the voice / face / live / registration hooks, the API) and `test_render_engine.py` (a render writes `face_use` and `manifest_issued`); **4 plants caught** | pass |
| R-36 | `test_authenticity.py` (22: every verdict, precedence, audit-record trace from a watermark id read with bit errors, sidecar manifests, upload limits and temp-file cleanup, no server path in errors; **4 plants caught**); live through a real server (T5.3): exact file + manifest -> `authentic_original`; a CRF-27 re-encode -> `ours_modified`, traced to its job with 0 bit errors; on files: CRF 26 re-encode, a 1 s trim, a 320 px downscale (127/127/124 of 128 tag bits), the audio track alone as AAC, an unmarked old render -> `no_evidence`, someone else's video with our manifest -> `manifest_for_another_file`, an edited manifest -> `tampered_manifest`, our speech and a human recording | pass |
| N-01 | 17 Sep: MOS 4.33 (SQUIM, Kokoro). 8 Oct: Bark dialogue 4.12 (SQUIM, self-referenced, biased up). **Re-run 8 Oct (T6.6), non-matching reference clip, SQUIM, Kokoro, 5 sentences, CPU: mean MOS 4.894 (min 4.804), PESQ 3.77** (`docs/benchmarks/phase3_benchmark_20261008.json`; a different sentence set from 17 Sep's 4.33, so the two are not a trend) | measured — met |
| N-02 | 8 Oct, ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb), clone of the first 30 s of the LJSpeech reference scored against the held-out last 20 s (admissible: human, open licence), 6 sentences, CPU: **XTTS-v2 mean 62.0%** (sd 4.6, min 55.1, max 67.6); OpenVoice V2 34.2% (base 26.9%); ceiling = the same speaker's real speech 91.7%. **Cross-lingual XTTS-v2 (T3.3), same method:** Spanish 43.9%, Hindi 48.8%, French 42.8% (6 sentences each). `scripts/measure_clone_similarity.py`, JSON in `outputs/benchmarks/` | **not met** (62.0% vs 85% and 90%) |
| N-03 | 8 Oct, real uvicorn, keep-alive client (`scripts/load_test.py`, queue `in_memory`): **render-job POST → 202 mean 4.5 ms, p95 6.6, max 20.8 ms (20 jobs)** → met. Synthesis POST in this mode runs the task inside the request: first call 7,211 ms (cold models), warm 287–296 ms for a one-sentence clip, so it is **not** an initiation time and grows with the text; only `QUEUE_BACKEND=celery` answers 202 at once (not run here: no Redis) | **met for render jobs; synthesis not met in in_memory mode** |
| N-04 | 8 Oct, real uvicorn, 8 concurrent clients, 30 s, `GET /audio/languages?limit=5`: **default limiter (120 rpm + burst 30) admitted 138 requests/min** and refused 79,785 with fast 429s (p50 10 ms for accepted). Limiter lifted: **56,033 /min (1 client), 48,422 (8), 46,542 (32)**, p50/p95/p99 at 32 clients 36/105/116 ms: one process is GIL-bound, concurrency adds latency not throughput. These are a cheap read; renders are one-at-a-time by design | **met by policy (138 ≥ 100)**; the policy is only just above the target |
| N-05 | 8 Oct, SyncNet v2, 25 fps: **Wav2Lip** offset 0 on 3/3 clips, LSE-C 9.84 / 10.51 / 11.18, LSE-D 6.61 / 5.97 / 5.48; blendshape offset 0, LSE-C 3.98 / 4.23 / 5.55 (T2.3). **Blendshape offset re-checked (T6.6): the -4 seen once through the SDK did not reproduce** — 10 renders (2 lengths x with/without a `#0b3d91` background, plus 6 other sentences) gave offset 0 nine times and +1 once, LSE-C 3.4–6.7. The D-12 window percentage over 10+ languages is T8.4 | measured — % figure pending (T8.4) |
| N-06 | 8 Oct (T6.6), CPU only, `scripts/render_avatar.py`, `demo` face, blendshape, 512 px, 25 fps, 176 words -> **53.05 s of video**, whole command (model loads + synthesis + alignment + render + encode): **43.8 s wall with the marks off (render stage alone 5.7 s), 133.8 s wall with both watermarks on (render stage 81.4 s)**; scaled to 60 s that is ~50 s and ~151 s. The watermark is what costs the time: VideoSeal runs frame by frame on the CPU. Wav2Lip not timed on 60 s. No GPU here | **not met** on CPU (43.8 s unmarked / 133.8 s marked vs 30 s); render stage alone 5.7 s would meet it; GPU not measured |
| N-07 | 8 Oct, real server, real Kokoro + aligner + animator, CPU, 384 px, 25 fps, 3 sentences / 7.2 s of speech (`scripts/live_client.py`): **first audio → first frame 1.4 ms (run 1), 1.8 ms (run 2)**, the roadmap's definition ("first frame after the first chunk", < 200 ms) → **met**. The viewer-facing numbers are reported beside it, not hidden behind it: **text → first audio 425 / 461 ms warm** (7,252 ms cold: models loading), text → first frame 426 ms. Real-time check (client starts playing at first audio): 2 of 182 frames late (max 13 ms), 0 late audio chunks; frames render in 2–4 ms each | **met as defined; end-to-end warm first response ~0.43 s on CPU** |
| N-08 | 8 Oct: `test_queue_concurrency.py` (64 threads, one id: accepted exactly once; 60 distinct ids each run once; 80-submission mixed burst; **proved able to fail: with the lock removed the same-id test fails**) and live `scripts/concurrency_test.py` on a real server: **60 distinct render jobs + 12 repeats released together: 60×202, 12×409, 60 COMPLETED, 0 lost / failed, 60 distinct video files**; burst accepted in 0.22 s (accept latency mean 136 ms, p95 206), drained in 18.8 s = 191 jobs/min for ~1.2 s clips on CPU, one worker | **met** (60 ≥ 50, in_memory queue; Celery/Redis not run) |
| N-09 | 8 Oct, `jitter_metric.py` (static anchor landmarks per frame, % of inter-ocular distance): `demo` blendshape render mean **0.278%**, p95 0.758%, max 0.963% (target < 2%); negative control through the real detector: still photo 0.0%, 5 px shake 3.59%. Unit: `test_jitter_metric.py` | measured — met (upper bound, see method); also Wav2Lip 8 Oct: mean 0.354%, p95 0.834%, max 1.093% |
| N-10 | not measurable here (no CUDA in sandbox); host measurement requested (M-04) | gap |
| N-11 | needs a deployment (Q-07) | gap |

### Added from the real problem statement (8 Oct 2026)

| ID | Test / measurement | Status |
|---|---|---|
| R-40 | T8.2 | gap |
| R-41 | T8.5 | gap |
| R-42 | `test_app_vision.py` (4 batch tests: a mixed batch where an unknown avatar, a malformed job and a repeated id are refused by index with their reasons while the good ones complete; 50 jobs all accepted, each run once, none lost; over the cap of 50 and empty batches refused whole; engine passed through, unknown batch 404; **plant caught**: stopping at the first bad item fails the mixed test). Live (T8.1), real server, CPU: `scripts/batch_test.py --jobs 10`: POST -> 202 in 306 ms, 10 accepted, 3 refused (404 unknown avatar, 422 missing `phonemeTimestamps`, 409 repeated id), all 10 COMPLETED in 48.6 s, 10 distinct videos on disk | pass |
| R-43 | T8.6 | gap |
| R-44 | T8.8 | gap |
| R-45 | not planned (CPU-infeasible), D-50 | not built |
| R-46 | consent, rate limit, watermark, manifest exist; T8.9 adds the rest | partial |
| R-47 | emotion, speed, pitch only | partial |
| R-48 | WebSocket used instead, D-46 | not built |
| R-51 | `test_metrics.py` (6: empty case reports `None` not 0, nearest-rank percentiles are observed values, counts/timings/watermark/score aggregation, a result missing fields does not break the report) and `test_app_vision.py` (the endpoint, and that a lip-sync score is kept on the job); **2 plants caught** (percentile rank off by one, score not stored). Live (T8.10), real server: `GET /api/v1/metrics` over 123 finished renders: render time mean 1.63 s / p95 6.84 s, real-time factor mean 0.73 / p95 3.0, 22 watermarked and 101 not (older renders), no peak VRAM (CPU); after `POST .../lipsync-score` on one render: `lipSyncScored` 1, LSE-C 4.928, offset 0 | pass |
| R-52 | docs 01-15 exist; README and contribution guide T7.6 | partial |
| R-53 | cannot be done from here | not possible |
| N-12 | LPIPS < 0.1: T8.3 | gap |
| N-13 | identity > 90%: T8.3 | gap |
| N-14 | photo to avatar < 20 s / < 10 s: registration ~2 s, generation 141 s (CPU) | partly measured |
| N-15 | 50+ parameters: 4 face attributes + background colour | **not met** |
| N-16 | 1080p+: renders are capped at the photo's size | **not met** |
| N-17 | TTS < 2 s per 30 s of audio: T8.4 | gap |
| N-18 | 20+ languages: speech 1,077; lip sync scored on 3 (T8.4) | partial |
| N-19 | streaming < 100 ms: text to first audio 425-461 ms warm (CPU) | **not met** |
| N-20 | < 5 s for a 30 s avatar video: T8.4 | gap; **not met on CPU with the marks on** |
| N-21 | temporal consistency > 95%: jitter 0.28-0.35% (N-09), no 95% definition | measured, undefined |

