# 14 — Code guide: one request through the system

This guide follows a single piece of work, "say this sentence with this face", from the HTTP
request to the finished MP4, naming the module and function at each step. Read it with the code
open; every module also starts with a docstring saying what it is for. A concepts index at the end
says where each technique is explained in the code the first time it appears.

## 0. The shape of the system

```
browser / SDK ──HTTP──> app.py (FastAPI) ──> voice_engine.py ──> speech engines (Kokoro, XTTS-v2, MMS, Bark, OpenVoice)
                           │                      │
                           │                      ├─> emotion_engine.py (prosody)   watermark_engine.py (AudioSeal)
                           │                      └─> alignment_engine.py (phoneme timings)
                           │
                           ├──> job_queue.py ──worker thread──> render_engine.py ──> face_engine.py (MediaPipe)
                           │       (job_store.py, SQLite)              │               face_animation.py + face_warp.py
                           │                                           │               or wav2lip_engine.py
                           │                                           ├─> video_watermark.py (VideoSeal), video_io.py (ffmpeg)
                           │                                           └─> manifest.py (signed record), audit_log.py
                           │
                           └──WebSocket──> live_engine.py (sentence by sentence, or the user's microphone)
```

The one interface between the speech half and the video half is `AvatarRenderJob` in
`backend/contracts.py` (golden rule 4): everything the renderer needs, nothing else.

## 1. The request arrives — `backend/app.py`

`POST /api/v1/audio/synthesize` with `{"text": "Hello there.", "mode": "fast", "returnAlignment": true}`.

1. **Middleware** `enforce_security` runs first: it binds a request id (`request_context.py`), checks
   the API key and the rate limit (`security.py`). Every log line from here on carries the id.
2. **Validation**: FastAPI parses the body into `AudioSynthesisRequest` (`contracts.py`). A bad field
   is a 422 naming the field, before any of our code runs.
3. **Route** `create_synthesis_job` asks the router which engine would run (`select_model`) and checks
   it can (`preflight`): weights on disk, a consented voice reference, a voice not on the protected
   list. A request that could never succeed is refused here (403/503) instead of failing later.
4. The work is handed to `celery_app.synthesize_audio`. With the default `QUEUE_BACKEND=in_memory`
   Celery runs eagerly, so the POST returns the finished result; with `celery` it returns a task id
   that is polled with `GET /api/v1/audio/synthesize/{taskId}`. A failed task reports its reason in
   `error`.

## 2. Speech — `backend/voice_engine.py`

`VoiceEngineRouter.synthesize` takes a lock (one synthesis at a time: one heavy model resident,
golden rule 5) and runs `_synthesize`:

1. **Pick and load the engine** (`_synthesize_kokoro`, `_synthesize_xtts`, `_synthesize_mms`, ...).
   Models load lazily and `gpu_utils` unloads the others first when memory is short.
2. **Emotion** (`emotion_engine.EmotionProsodyEngine`): pitch, rate and loudness changes applied to
   the waveform, because none of the engines takes an emotion input.
3. **Watermark** (`_watermark_output` -> `watermark_engine.py`): AudioSeal embeds a 16-bit keyed tag
   and the file is read back to prove the mark is there. No mark, no audio (no silent fallback).
4. **Alignment** (`alignment_engine.ForcedAligner`): the text is force-aligned to the audio with
   MMS_FA, giving each phoneme a start and end in milliseconds, mapped to one of 15 visemes. If the
   aligner cannot run, the timing is *estimated* and `alignmentMethod` says `acoustic-fallback`.
5. **Records**: a speech record (`manifest.write_speech_record`) is written beside the WAV, tied to
   its SHA-256; a cloned voice gets a `voice_use` entry in the audit trail (`audit_log.py`).

The response carries `modelUsed`, `durationSeconds`, `phonemeTimestamps` and `alignmentMethod`.

## 3. The render job — `app.py` -> `job_queue.py`

The client builds an `AvatarRenderJob` from that response (the SDK's `render` does it) and posts it
to `POST /api/v1/avatar/render-job`.

1. `_submit_render` runs `render_engine.preflight`: the avatar exists and has a consent record
   (`avatar_store.require_usable`), the audio URL points inside `outputs/` or `inputs/`, the engine's
   weights exist. Then `job_queue.enqueue`.
2. `InMemoryJobQueue` stores the job in SQLite (`job_store.py`) so a restart does not lose it, and
   hands it to a single worker thread (one render at a time: one GPU-sized model).
3. The client polls `GET /api/v1/avatar/render-job/{jobId}` and sees `progress` move.

## 4. Rendering — `backend/render_engine.py`

`render_job` is the whole video pipeline in order:

1. **The photo**: `store.load_image` (consent checked again), optionally super-resolved once for
   1080p (`super_resolution.py`), resized to the quality box, optionally given a new background
   (`apply_background`, MediaPipe selfie segmenter).
2. **The face**: `face_engine.FaceMeshEngine.analyze` finds 478 landmarks, head pose and blendshapes.
3. **The motion**: `face_animation.build_animation` turns phoneme timestamps into per-frame
   blendshape weights: visemes held and smoothed, emotion added, blinks scheduled, the mouth gated by
   the audio's loudness.
4. **The frames**: the blendshape engine warps the photo per frame (`face_warp.PortraitAnimator`,
   a triangle-mesh warp); the Wav2Lip engine (`wav2lip_engine.py`) instead repaints the mouth from
   the audio. The "AI-generated" label is stamped on.
5. **The video mark**: frames are marked with VideoSeal in windows of 32 as they stream to the
   encoder (`video_watermark.py`), so memory stays bounded.
6. **Encoding**: `video_io.VideoWriter` pipes raw frames into ffmpeg and muxes the audio. The mark is
   then read back *from the encoded file*; a video that cannot show its mark is not delivered.
7. **The manifest**: `manifest.build_manifest` lists inputs (avatar, consent basis, audio and its
   speech record), processing and models, binds the video's SHA-256 and signs it with Ed25519.
   `face_use` and `manifest_issued` go on the audit trail.

The result (`RenderResult.to_dict`) carries the video URL, timings, the watermark read-back, the
manifest link and any warnings.

## 5. After the render

- `POST .../lipsync-score` -> `lipsync_metric.score_video`: SyncNet embeddings of mouth crops and
  MFCCs, the best audio/video offset, LSE-C / LSE-D and the share of seconds within one frame.
- `POST /api/v1/provenance/verify` -> `authenticity.py`: reads both watermarks, checks a manifest's
  signature and hash, finds the audit record, and gives a verdict with what it cannot prove.
- `GET /api/v1/metrics` -> `metrics.py`: queue depth, render timings, scores so far.

## 6. The other entry points

| Entry | Path through the code |
|---|---|
| Live avatar `WS /api/v1/live` | `app.live_avatar` -> `live_engine.LiveSession`: per sentence `synth_chunk` (router + aligner), `build_track`, `render_jpeg`, streamed by `stream_text`; microphone audio by `stream_audio_chunk` (mouth from loudness) |
| Your own recording | `voice_to_avatar.prepare`: decode, Whisper if no transcript, align, record as supplied -> an ordinary render job |
| Batch | `create_render_batch`: each item through `_submit_render`, refusals by index |
| New face | `avatar_generator.py` (SD 1.5 text-to-image + quality gate) or `style_transfer.py` (img2img, identity scored) -> `avatar_store.register` |
| Protected voices | `protected_voices.py`: embeddings checked in `require_voice_consent` and after a clone |
| CLI | `scripts/render_avatar.py` runs steps 2 and 4 in one process |

## 7. Concepts index

Where each idea is explained in the code the first time it appears.

| Concept | Where |
|---|---|
| Frozen contract between two halves | `contracts.py` (`AvatarRenderJob`) |
| Pydantic aliases, validation as the first line of defence | `contracts.py` |
| Request ids through threads (`contextvars`) | `request_context.py` |
| API keys, constant-time comparison, token-bucket rate limit | `security.py` |
| Lazy model loading, one heavy model resident, RAM/VRAM guards | `gpu_utils.py`, `model_registry.py` |
| Engine routing and preflight refusals | `voice_engine.py` (`select_model`, `preflight`) |
| Prosody: semitones, time-stretch, spectral tilt | `emotion_engine.py` |
| Keyed audio watermark, read-back verification | `watermark_engine.py` |
| CTC forced alignment, romanisation for non-Latin scripts | `alignment_engine.py`, `romanizer.py` |
| Phonemes to visemes to blendshapes | `viseme_blendshapes.py`, `face_animation.py` |
| Face landmarks, head pose, quality gate | `face_engine.py` |
| Mesh warping a photo | `face_warp.py` |
| Neural lip sync | `wav2lip_engine.py` |
| ffmpeg pipes, muxing, probing | `video_io.py` |
| Video watermark windows, bit accuracy | `video_watermark.py` |
| Signed manifests (Ed25519 + SHA-256) | `manifest.py` |
| Hash-chained audit log | `audit_log.py` |
| Thread-safe job queue with immutable snapshots | `job_queue.py`, `job_store.py` (SQLite WAL) |
| SyncNet, MFCCs, LSE-C / LSE-D | `lipsync_metric.py` |
| LPIPS and face-embedding identity | `visual_metrics.py` |
| Consent and provenance sidecars | `provenance.py`, `avatar_store.py` |
| Speaker embeddings and an opt-out list | `quality_auditor.py`, `protected_voices.py` |
| WebSocket media framing, presentation timestamps | `live_engine.py`, `frontend/src/LivePanel.jsx` |
| Speech recognition before alignment | `voice_to_avatar.py` |
| Img2img style transfer, provenance inheritance | `style_transfer.py` |
| Tiled super-resolution | `super_resolution.py` |
| React hooks, polling, uploads | `frontend/src/AvatarPanel.jsx`, `App.jsx` |
| Web Audio clock, drawing frames on time, microphone capture | `frontend/src/LivePanel.jsx` |
