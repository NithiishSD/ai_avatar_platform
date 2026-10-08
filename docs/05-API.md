# 05 — API

Base path `/api/v1`. JSON uses camelCase. Errors are FastAPI's
`{"detail": ...}`; validation errors are 422 with the failing field path.
Auth: `X-API-Key` header when `AUTH_ENABLED=true`. Rate limit: 429 with
`Retry-After`. Public without a key: `/health`, `/docs`, `/redoc`,
`/openapi.json`, `/outputs/*`.

## Endpoints (current)

| Method | Path | Purpose | Success |
|---|---|---|---|
| GET | `/health` | status, model weights, vision weights, render engines, security | 200 |
| GET | `/api/v1/audio/samples` | voice references in `inputs/` with provenance | 200 |
| POST | `/api/v1/audio/synthesize` | queue synthesis (`AudioSynthesisRequest`) | 202 + `taskId` |
| GET | `/api/v1/audio/synthesize/{taskId}` | poll synthesis | 200 |
| POST | `/api/v1/audio/align` | align existing audio to a transcript | 200 |
| GET | `/api/v1/audio/languages` | MMS language catalogue, `?q=` search | 200 |
| GET | `/api/v1/audio/languages/{code}` | one language | 200 / 404 |
| GET | `/api/v1/audio/emotions` | emotion presets | 200 |
| POST | `/api/v1/audio/quality-audit` | SQUIM scores for a clip | 200 |
| POST | `/api/v1/audio/voice-similarity` | ECAPA similarity with admissibility | 200 |
| GET | `/api/v1/avatar/faces` | registered faces, consent bases, engines | 200 |
| POST | `/api/v1/avatar/faces` | register a photo (multipart, consent basis) | 201 |
| GET | `/api/v1/avatar/faces/{avatarId}/image` | the image | 200 |
| DELETE | `/api/v1/avatar/faces/{avatarId}` | remove a face | 204 |
| POST | `/api/v1/avatar/face/analyze` | landmarks, pose, blendshapes, quality | 200 |
| POST | `/api/v1/avatar/render-job` | queue a render (`AvatarRenderJob`, `?engine=`); with `renderQuality: 1080P_HQ` a photo that would give less than 720p is first enlarged by Real-ESRGAN and the result's `superResolution` says so | 202 |
| GET | `/api/v1/avatar/render-job/{jobId}` | poll a render | 200 / 404 |
| POST | `/api/v1/avatar/render-job/{jobId}/lipsync-score` | SyncNet score of the result | 200 |
| POST | `/api/v1/avatar/render-batch` | queue up to 50 render jobs (`{"jobs": [AvatarRenderJob, ...]}`, `?engine=`); each item is checked on its own; returns `batchId` and, per item, `accepted` or `httpStatus` + `detail`; over 50 or empty is a 422 | 202 |
| GET | `/api/v1/avatar/render-batch/{batchId}` | state of every job in the batch, `counts` by state, `done` | 200 / 404 |
| POST | `/api/v1/avatar/voice-to-avatar` | the caller's own recording drives a face: multipart `file` (<= 50 MB, <= 120 s), `avatarId`, `consentBasis` (speaker-recorded, written-consent, open-licence), optional `transcript`, `language`, `engine`, `renderQuality`; without a transcript Whisper base recognises the words; returns the render job plus `speech` (transcript, its source, language, alignment); the manifest says `origin: supplied` | 202 / 400 / 403 / 404 / 422 |
| GET / POST / DELETE | `/api/v1/abuse/protected-voices[/{id}]` | the opt-out list: POST a recording (multipart `file`, <= 25 MB) -> `{id}`; only the speaker embedding is kept; cloning a voice that matches it is refused (403) | 200 / 201 / 204 |
| GET | `/api/v1/parameters` | every customisation parameter with type, range, default, and a `status` from measuring it (`measured`, `no-effect`, `limited`, `not-measured`); counts against the 50+ target, which is reported as not met | 200 |
| GET | `/api/v1/metrics` | queue depth by state, render time and real-time-factor spread (mean/p50/p95/max), watermarked count, lip-sync scores taken so far, audit trail size; `None` where nothing has been measured; the Celery backend cannot list jobs and says so | 200 |
| GET | `/api/v1/avatar/generate/options` | fixed attribute choices + whether weights exist | 200 |
| POST | `/api/v1/avatar/generate` | queue a synthetic face (`AvatarGenerateRequest`; 409 id taken, 503 no weights) | 202 + `taskId` |
| GET | `/api/v1/avatar/generate/{taskId}` | poll a generation | 200 / 404 |

## Planned (from 07-TASKS)

| Method | Path | Task |
|---|---|---|

| POST | `/api/v1/provenance/verify` | is this audio/video file ours? multipart `file` (<= 200 MB) or form `path`, optional `manifest`; reports audio mark, video mark, manifest and audit record separately, and a verdict | 200 / 400 / 413 / 422 |
| GET | `/api/v1/audit` | the consent audit trail, newest first; filters `event`, `subject`, `since`, `limit` | 200 / 400 |
| GET | `/api/v1/audit/verify` | recompute the trail's hash chain | 200 |

## Live avatar: `WS /api/v1/live`

One session = one socket. Client to server (JSON text): `start` (`avatarId`, `language`, `mode`
fast|clone, `emotion`, `fps` 5–30, `maxSide` 128–640, `speakerWav` + `cloneEngine` for clone mode,
`apiKey` because a browser WebSocket cannot set headers; the `X-API-Key` header also works),
then any number of `say` (`text` ≤ 2000 chars), `interrupt` (stop speaking, keep the session) and
finally `stop`. Server to client: `ready` (size, fps, sampleRate, model, requestId); per sentence an
`audio` message then binary media, a `chunk` message with its timings, and `done` with the totals;
`interrupted`; `error` (`code` + `detail`; start errors also close the socket).

Binary messages are `kind(u8) chunk(u32) index(u32) presentationMs(u32)` big-endian (13 bytes) then
the payload: kind 1 = PCM16 mono audio at `sampleRate`, kind 2 = one JPEG frame. `presentationMs` is
when it is due on the session timeline, so a client plays by that, not by arrival. Error codes:
`bad_start`, `start_timeout`, `unauthorised`, `busy` (close 1013; `LIVE_MAX_SESSIONS`, default 2),
`consent`, `avatar_not_found`, `bad_request`, `model_unavailable`, `bad_message`, `malformed`,
`speech_failed`, `idle`. Consent, weights and language are checked before any speech, as in REST.

**Streaming audio input (R-41).** Instead of text, a client can send its own speech: `audio_start`
(`sampleRate` 8000–48000, `consentBasis` one of `speaker-recorded`, `written-consent`, `open-licence`;
written to the audit trail as `audio_supplied`) answers `audio_ready` with `drive: "audio-energy"`.
Each following **binary** message is up to 2 s of 16-bit little-endian mono PCM; the server answers
with that chunk's JPEG frames (kind 2, on the session's single 1000/fps ms grid) and a `chunk`
message labelled `drive: "audio-energy"`, `note: "mouth estimated from the audio's loudness; not
phoneme-aligned"`. No audio is sent back. `audio_end` answers `done`. Errors: `audio_not_started`
(PCM before `audio_start`), `bad_audio` (empty, odd length, longer than 2 s); the session carries on.

## Request ids

Every response carries `X-Request-ID` (also on 401/429/500). Send your own
(`A-Za-z0-9._-`, up to 64 characters) to trace a call; anything else is
replaced. Every log line written for the request, including the render or
generation worker it queued, shows it as `[id]`. An unhandled error answers
500 with `{"detail": ..., "requestId": ...}`.

## Status codes used

400 bad input the schema cannot express (unsupported engine, unreadable URL) ·
401 missing/invalid key · 403 consent not satisfied · 404 unknown id ·
409 duplicate jobId · 413 upload too large · 422 schema violation ·
429 rate limited · 503 required model weights missing (message names the fix).
