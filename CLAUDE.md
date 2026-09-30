# AI Avatar Platform — Context for AI Assistants

Read this first. It is the single entry point for any AI assistant (or new
developer) working on this repo. It says what we are building, the rules that
are not negotiable, how to run and verify things, and where progress lives.

| Need | Go to |
|---|---|
| What to build next, acceptance criteria, status | `docs/MILESTONES.md` |
| Something is broken | `docs/DEBUGGING.md`, then run `scripts/doctor.py` |
| Why a decision was made, full explanation | `docs/PROJECT_DOCUMENTATION.md` (PDF alongside) |
| Dated history of every work session | `docs/context.md` (append-only log) |
| The team roadmap (source of phases and gates) | `docs/AI_Avatar_Platform_2_Developer_Roadmap.pdf` |

---

## 1. Mission

Build an **open-source AI avatar platform**: script + optional voice sample +
one photo → a talking, lip-synced avatar video. Every model must be open
source and run locally on the hardware below.

The one outcome that matters most right now: **Integration Gate 2 — a cloned
voice driving a lip-synced face.** The lip-synced face exists; the cloned voice
does not (XTTS-v2 weights). Everything else is secondary until it does.

## 2. Hard constraints

| Constraint | Value | Consequence |
|---|---|---|
| GPU | NVIDIA RTX 4050 Laptop, **6141 MiB VRAM** | One large model resident at a time; pipeline runs sequentially |
| Disk | 43.5 GB free on 29 Sep 2026 (check with `doctor.py`) | Full model set ≈ 15 GB; `fetch_models.py` refuses to go below 5 GB free |
| Python | **3.10** in `backend/.conda` | Coqui TTS requires 3.10; base conda is 3.14 and cannot run this project |
| numpy | `<2.0.0` | Coqui TTS / numba break on numpy 2; never add a package that forces numpy 2 |
| transformers | `<4.48` | Dia and Higgs are loaded through transformers, not their own packages |
| MediaPipe | 1.x Tasks API | `mp.solutions` no longer exists; use `vision.FaceLandmarker` with `.task` bundles |
| Licences | Open source only; XTTS-v2 is Coqui CPML (non-commercial) | A human must accept the CPML before XTTS-v2 is fetched |

Requirement thresholds (from the problem statement; the PDF copy in the repo
is corrupted, so re-verify against a fresh copy when available):

| Requirement | Target |
|---|---|
| TTS models | 5+ (Kokoro, XTTS-v2, OpenVoice V2 named) |
| Cloning reference | 30–60 s |
| Speech quality | MOS > 3.5 |
| Cloning similarity | > 85% (ECAPA-TDNN) |
| API job initiation | < 500 ms |
| API capacity | 100+ requests/min |
| Security | Auth + rate limiting |
| Roadmap final gate | 50+ concurrent tasks, < 30 s per 60 s video, < 200 ms real-time, > 95% lip sync |

## 3. Golden rules (do not break these)

1. **No silent fallbacks.** A missing model or failed load must be visible:
   logged loudly, reported in `/health`, and reflected in `model_used`. A
   fallback that looks like success hid three missing models for three phases.
2. **Every number records its method.** Only admissible evidence may be
   reported as meeting a threshold: MOS from SQUIM, similarity from ECAPA-TDNN
   against a *human, consented* reference. See `scripts/benchmark_phase3.py`.
3. **Consent before use.** No voice or face is used without a provenance
   sidecar (`<file>.provenance.json`, see `backend/provenance.py`). Never
   download or commit a photo or recording of an identifiable person whose
   rights we have not confirmed. Check EXIF/licence text before using any image.
4. **The contract is frozen.** `AvatarRenderJob` in `backend/contracts.py`
   is the only interface between audio and vision. Changing it means: update
   both sides, update tests, and record the change in `docs/context.md`.
5. **Respect 6 GB.** Load lazily, keep one heavy model resident, release it
   (`del model; torch.cuda.empty_cache()`) before loading another heavy one.
6. **Tests never download models.** Mock loaders in unit tests. But every
   feature also needs one *live* verification on real weights/data, recorded
   in `docs/context.md` — mocked tests alone are how the missing models hid.
7. **Errors say how to fix themselves.** An exception for a missing model or
   file names the command that fixes it (e.g. `scripts/fetch_models.py`).
8. **Say what is not done.** Status in `docs/MILESTONES.md` must match
   reality: *built*, *wired*, *verified live*, and *passed gate* are different.

## 4. Environment and commands

Always use the project interpreter. Do not `conda activate base`.

```bash
PY=./backend/.conda/bin/python          # Python 3.10.21 with all deps
# or: conda activate ./backend/.conda   (from the repo root)
```

| Task | Command (from repo root) |
|---|---|
| Health check everything | `PYTHONPATH=backend $PY scripts/doctor.py` |
| Run all tests | `PYTHONPATH=backend:tests $PY -m unittest discover -s tests -p 'test_*.py'` |
| Run one test file | `PYTHONPATH=backend:tests $PY -m unittest discover -s tests -p 'test_face_engine.py'` |
| Start backend | `cd backend && PYTHONPATH=. ./.conda/bin/python -m uvicorn app:app --port 8000 --reload` |
| Start frontend | `cd frontend && npm run dev -- --port 5173` |
| Model weights status | `PYTHONPATH=backend $PY scripts/fetch_models.py --dry-run` |
| Fetch missing weights | `PYTHONPATH=backend $PY scripts/fetch_models.py [--only xtts-v2] [--all]` |
| Benchmark vs thresholds | `PYTHONPATH=backend $PY scripts/benchmark_phase3.py` → `docs/benchmarks/` |
| Smoke voice reference | `PYTHONPATH=backend $PY scripts/make_reference.py --smoke` |
| Make a synthetic avatar face | `PYTHONPATH=backend $PY scripts/make_avatar.py --synthetic --avatar-id demo --seed 7` |
| Register a real face | `PYTHONPATH=backend $PY scripts/make_avatar.py --human FILE --avatar-id ID --subject NAME --consent subject-provided` |
| Text + face → talking MP4 | `PYTHONPATH=backend $PY scripts/render_avatar.py --text "..." --face demo [--metric]` → `outputs/renders/` |
| Vision weights status / fetch | `PYTHONPATH=backend $PY scripts/fetch_vision_models.py [--dry-run] [--only KEY]` |
| Redraw the flowcharts | `$PY scripts/make_diagrams.py` → `docs/images/` |
| Register human reference | `PYTHONPATH=backend $PY scripts/make_reference.py --human FILE --speaker NAME --licence L --consent open-licence` |
| Redis + Postgres (optional) | `./start-docker.sh` (only needed when `QUEUE_BACKEND=celery`) |

`pytest` is not installed; the suite uses `unittest`. Run tests through
`discover` as shown — `python -m unittest tests.test_x` fails on imports.

Config lives in `.env` (template: `.env.example`). Key switches:
`QUEUE_BACKEND=in_memory` (Celery eager, no Redis needed) or `celery`;
`AUTH_ENABLED` + `API_KEYS`; `RATE_LIMIT_RPM`; `CORS_ORIGINS`;
`RENDER_ENGINE=blendshape` (default) or `wav2lip`.

## 5. Architecture

```
Script + voice ─► Speech router ─► Forced aligner ─┐
                  (5 TTS models)   (15 visemes)    ▼
                                            AvatarRenderJob  (frozen contract)
                                                   │
Photo ─────────► Face analysis ──────────► Lip sync ─► Composite ─► MP4
                 (478 pts, 52 blendshapes)
FastAPI accepts every step as a job; Celery runs it (eager in development).
```

### Module map

| File | Owns | State |
|---|---|---|
| `backend/contracts.py` | Pydantic schemas incl. `AvatarRenderJob`, `PhonemeTimestamp`, `EmotionVector` | Frozen contract |
| `backend/app.py` | FastAPI routes, security middleware, startup weight audit, `/outputs` static mount | Live |
| `backend/celery_app.py`, `job_queue.py` | Task dispatch (eager when `in_memory`) | Live |
| `backend/voice_engine.py` | `VoiceEngineRouter`: Kokoro, XTTS-v2, Higgs 3B, Dia 1.6B, MMS-TTS | Live; only Kokoro + MMS have weights |
| `backend/mms_engine.py`, `language_registry.py` | MMS-TTS per-language VITS, LRU cache, ISO codes | Live |
| `backend/romanizer.py` | Shared lazy `uroman` (MMS input + aligner) | Live |
| `backend/alignment_engine.py` | MMS_FA forced alignment → phonemes → 15 visemes; acoustic fallback | Live |
| `backend/emotion_engine.py` | Emotion vectors → prosody; `to_render_emotion_vector` | Live |
| `backend/quality_auditor.py` | SQUIM MOS/PESQ/STOI, ECAPA similarity | Live |
| `backend/audio_utils.py` | Probe, validate, convert to 24 kHz mono for cloning | Live |
| `backend/security.py` | API keys, token-bucket rate limit | Live |
| `backend/provenance.py` | Consent/provenance sidecars and admissibility | Live |
| `backend/model_registry.py` | On-disk weight audit (no downloads, no torch) | Live |
| `backend/face_engine.py` | MediaPipe landmarks, bbox, head pose, blendshapes, quality gate, segmenters, shared locked engine | Live |
| `backend/avatar_store.py` | Registry of faces in `inputs/faces`; enforces consent and the quality gate | Live |
| `backend/avatar_generator.py` | Synthetic faces from SD 1.5, seed walk until the gate passes | Live (CLI only) |
| `backend/viseme_blendshapes.py` | 15 visemes and 6 emotions → ARKit blendshape weights | Live |
| `backend/face_animation.py` | Timestamps → smoothed per-frame weights, closures, energy gate, blinks | Live |
| `backend/face_warp.py` | `PortraitAnimator`: CPU mesh warp, procedural mouth interior and eyelids | Live |
| `backend/video_io.py` | ffmpeg encode/mux, ffprobe, frame and audio readers | Live |
| `backend/render_engine.py` | `AvatarRenderJob` → MP4; preflight; engines `blendshape` / `wav2lip`; AI label | Live |
| `backend/wav2lip_engine.py` | Wav2Lip network + inference | Built; checkpoint not fetched (licence) |
| `backend/lipsync_metric.py` | SyncNet LSE-C / LSE-D / offset | Live |
| `backend/gpu_utils.py` | VRAM guard and model release registry | Live for vision models; TTS router not registered |
| `frontend/src/App.jsx` | Creator studio: synthesis, clone picker, emotions, viseme display | Live |
| `frontend/src/AvatarPanel.jsx` | Avatar picker, landmark canvas, photo registration, render, video, sync score | Built; not yet checked in a browser |
| `frontend/src/App.new.jsx` | Earlier sample UI, unused | Candidate for removal |

Flowcharts of all of this: `docs/images/` (see `docs/PROJECT_DOCUMENTATION.md`, "Engineering flowcharts").

### Key data

- **15 visemes**: `viseme_sil, PP, FF, TH, DD, kk, CH, SS, nn, RR, aa, E, I, O, U`.
- **Audio**: generated WAV is 24 kHz mono (MMS-TTS natively 16 kHz).
- **Paths**: generated media → `outputs/` (served at `/outputs`); voice
  references → `inputs/`; face photos → `inputs/faces/` (never committed);
  rendered videos → `outputs/renders/`; vision weights → `.models/{mediapipe,syncnet,sface,wav2lip}/`; ECAPA → `.models/ecapa/`; HF models → `~/.cache/huggingface/hub`.
- **Coqui download dir** depends on `XDG_DATA_HOME`, which the VS Code snap
  rewrites; `model_registry.py` checks both locations.

## 6. How to add a feature (checklist)

1. Find its task ID in `docs/MILESTONES.md`; mark it `In progress`.
2. Follow existing patterns: lazy load + cached failure + actionable error;
   dataclass results with `to_dict()`; camelCase JSON aliases in `contracts.py`.
3. New model weights → add to `model_registry.audit_model_weights()` and
   `scripts/fetch_models.py`, so the audit and fetcher know about it.
4. New endpoint → schema in `contracts.py`, route in `app.py`, test in
   `tests/test_app*.py`. Heavy work goes through the queue, not the request.
5. Write unit tests with mocked models (`tests/test_<module>.py`).
6. Run the full suite; run `scripts/doctor.py`.
7. Verify live on real weights and real (consented) data; save evidence
   (numbers, output path) — not just "it works".
8. Update the task status in `docs/MILESTONES.md` and append a dated entry to
   `docs/context.md`: what changed, how it was verified, what is still open.

## 7. Definition of done

A task is **done** only when all are true:

- [ ] Code follows the patterns above; no silent fallback introduced
- [ ] Unit tests added and the whole suite passes
- [ ] Verified live on real weights/data, with the evidence recorded
- [ ] `scripts/doctor.py` shows no new FAIL
- [ ] `docs/MILESTONES.md` status updated, `docs/context.md` entry appended
- [ ] Any metric it produces records its method and admissibility

## 8. Current state (update when it changes)

- Last verified suite: **446 tests passing** (30 Sep 2026). `doctor.py`: 32 pass, 6 warn, 0 fail.
- Gates passed: **Gate 0.** Gate 1's criterion is met through the API and CLI but the
  UI overlay has not been looked at. Gate 2 is not passed: there is no cloned voice.
- What works end to end: text → Kokoro or MMS-TTS speech → alignment → render job →
  lip-synced MP4 of a registered face, by CLI and by API, with a SyncNet score.
- Measured lip sync (blendshape engine, SyncNet LSE-C, real video ≈ 6–8): English 2.8,
  Hindi 4.7, Tamil 6.8; offset 0 frames in every clip. Details in `docs/context.md`.
- Weights present: Kokoro, MMS-TTS (hin, tam, swh, spa), ECAPA, MediaPipe landmarker +
  both segmenters, SyncNet, SFace, SD 1.5. Missing: XTTS-v2, Higgs, Dia, Wav2Lip.
- `inputs/` holds a synthetic smoke voice reference and one synthetic face (`demo`):
  both usable, neither admissible as evidence. No consented human voice or face yet.
- Next task: see the top of `docs/MILESTONES.md`.
