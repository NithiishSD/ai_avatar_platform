# 03 — Architecture

## Shape

One FastAPI process accepts every request, validates it against
`backend/contracts.py`, and hands heavy work to a queue. The queue is
`in_memory` in development (Celery eager plus a one-thread executor for
renders; no Redis needed) or `celery` with Redis in a deployment. Models load
lazily on first use and one heavy model is resident at a time.

```
                ┌──────────── FastAPI (app.py) ─────────────┐
 browser ──HTTP─┤ security middleware: API key, rate limit   │
 SDK     ──WS───┤ contracts.py validation at the edge        │
                └───────┬───────────────────────────┬────────┘
                        │ enqueue                   │ live session
                 job_queue.py / celery_app.py   live_engine (M4)
                        │
        ┌───────────────┼────────────────────────────┐
   voice_engine     alignment_engine            render_engine
   (5 TTS engines)  (MMS_FA → visemes)          (blendshape | wav2lip)
        │                                            │
   quality_auditor                       face_engine, face_animation,
   (SQUIM, ECAPA)                        face_warp, video_io, lipsync_metric
        │                                            │
        └──────────── provenance / watermark (M5) ───┘
```

## Module map

| File | Owns |
|---|---|
| `backend/contracts.py` | All request/response schemas; `AvatarRenderJob` is frozen |
| `backend/app.py` | Routes, security middleware, startup weight audit, `/outputs` mount |
| `backend/celery_app.py` | Celery app and tasks; clears `CELERY_*` env in in_memory mode |
| `backend/job_queue.py` | `InMemoryJobQueue` / `CeleryJobQueue`, same four methods |
| `backend/voice_engine.py` | `VoiceEngineRouter`: Kokoro, XTTS-v2, Higgs, Dia, MMS-TTS |
| `backend/mms_engine.py`, `language_registry.py` | MMS-TTS per-language VITS, LRU cache, ISO codes |
| `backend/romanizer.py` | Shared lazy `uroman` for MMS input and the aligner |
| `backend/alignment_engine.py` | MMS_FA forced alignment → phonemes → 15 visemes; reports `last_method` |
| `backend/emotion_engine.py` | Emotion presets and blends → prosody and render hints |
| `backend/quality_auditor.py` | SQUIM MOS/PESQ/STOI, ECAPA similarity with admissibility |
| `backend/audio_utils.py` | Probe, validate, convert references to 24 kHz mono |
| `backend/security.py` | API keys (constant-time), token-bucket rate limiter |
| `backend/provenance.py` | Consent sidecars and admissibility |
| `backend/model_registry.py` | On-disk weight audit, no downloads, no torch |
| `backend/gpu_utils.py` | VRAM guard and release registry |
| `backend/face_engine.py` | MediaPipe landmarks, pose, blendshapes, quality gate, segmenters |
| `backend/avatar_store.py` | Registry of faces in `inputs/faces`; consent + quality enforced |
| `backend/avatar_generator.py` | SD 1.5 synthetic faces, seed walk until the gate passes |
| `backend/viseme_blendshapes.py` | Visemes and emotions → ARKit blendshape weights |
| `backend/face_animation.py` | Timestamps → smoothed per-frame weights, blinks |
| `backend/face_warp.py` | `PortraitAnimator`: CPU mesh warp, mouth interior, eyelids |
| `backend/video_io.py` | ffmpeg encode/mux, ffprobe |
| `backend/render_engine.py` | `AvatarRenderJob` → MP4; preflight; engines; label |
| `backend/wav2lip_engine.py` | Wav2Lip network and inference |
| `backend/lipsync_metric.py` | SyncNet LSE-C / LSE-D / offset |
| `frontend/src/App.jsx` | Creator studio: synthesis, clone picker, emotions, visemes |
| `frontend/src/AvatarPanel.jsx` | Avatar picker, landmark canvas, registration, render, video, score |
| `scripts/` | doctor, fetchers, benchmark, render CLI, reference/avatar makers |

## Decisions that shape it

- **Contract first.** Audio and vision meet only at `AvatarRenderJob`.
- **Queue for everything heavy.** Requests return in milliseconds; a 6 GB card
  cannot hold a model per web worker.
- **Sequential GPU.** One render thread; `gpu_utils` releases before loading.
- **No silent fallback between engines.** Asking for `wav2lip` without its
  checkpoint is a 400, never a quiet `blendshape` render.
- **Local, open models only.**
