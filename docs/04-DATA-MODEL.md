# 04 — Data model

## The frozen contract: `AvatarRenderJob`

| Field (JSON) | Type | Rule |
|---|---|---|
| `jobId` | string | non-empty, unique per queue |
| `avatarId` | string | must exist in the avatar store and be usable |
| `audioUrl` | string | contains `://`; worker only reads this server's `/outputs/` or project `file://` |
| `sampleRate` | int | 8000–192000 |
| `durationSeconds` | float | > 0, ≤ 3600 |
| `phonemeTimestamps` | list of `PhonemeTimestamp` | non-empty, ordered by `startMs`, each `endMs` ≤ duration |
| `emotionVector` | `EmotionVector` | `happy`, `neutral` 0–1; `eyeblinkRate` 0–10; six optional emotions 0–1 |
| `renderQuality` | enum | `PREVIEW` or `1080P_HQ` |
| `targetFps` | int | 1–120 |

`PhonemeTimestamp`: `phoneme`, `viseme` (non-empty), `startMs` ≥ 0, `endMs` > `startMs`.
Unknown keys are rejected (`extra="forbid"`). Extensions must be optional fields.

## Job lifecycle

```
QUEUED ──worker picks up──► PROCESSING ──ok──► COMPLETED (videoUrl, result)
                                   └──error──► FAILED    (error)
```

Terminal states never change. Duplicate `jobId` → 409. Unknown id → 404.

## Storage

| What | Where | Tracked in git |
|---|---|---|
| Generated audio, images | `outputs/` (served at `/outputs`) | no |
| Rendered videos | `outputs/renders/` | no |
| Voice references + sidecars | `inputs/*.wav` + `.provenance.json` | no |
| Faces + sidecars | `inputs/faces/` | no |
| Vision weights | `.models/{mediapipe,syncnet,sface,wav2lip}/` | no (fetched by script) |
| HF weights | `~/.cache/huggingface/hub` | no |
| Job state (in_memory) | process dict | n/a, lost on restart until R-21 |
| Job state (celery) | Redis key `avatar:render-job:{id}` | n/a |

## Provenance sidecar (`<file>.provenance.json`)

| Field | Meaning |
|---|---|
| `source` | `human` or `synthetic` |
| `speaker` / `subject` | who it is |
| `licence` | licence text or name |
| `consentBasis` | `subject-provided`, `written-consent`, `open-licence`, `synthetic` |
| `notes`, `created` | free text, ISO timestamp |

Admissible as evidence only when `source == human` and the basis is a consent basis.
