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
| POST | `/api/v1/avatar/render-job` | queue a render (`AvatarRenderJob`, `?engine=`) | 202 |
| GET | `/api/v1/avatar/render-job/{jobId}` | poll a render | 200 / 404 |
| POST | `/api/v1/avatar/render-job/{jobId}/lipsync-score` | SyncNet score of the result | 200 |
| GET | `/api/v1/avatar/generate/options` | fixed attribute choices + whether weights exist | 200 |
| POST | `/api/v1/avatar/generate` | queue a synthetic face (`AvatarGenerateRequest`; 409 id taken, 503 no weights) | 202 + `taskId` |
| GET | `/api/v1/avatar/generate/{taskId}` | poll a generation | 200 / 404 |

## Planned (from 07-TASKS)

| Method | Path | Task |
|---|---|---|
| WS | `/api/v1/live` | streaming TTS + live avatar frames | M4 |
| POST | `/api/v1/provenance/verify` | does a file carry our watermark / manifest | M5 |
| GET | `/api/v1/audit` | consent audit trail | M5 |

## Status codes used

400 bad input the schema cannot express (unsupported engine, unreadable URL) ·
401 missing/invalid key · 403 consent not satisfied · 404 unknown id ·
409 duplicate jobId · 413 upload too large · 422 schema violation ·
429 rate limited · 503 required model weights missing (message names the fix).
