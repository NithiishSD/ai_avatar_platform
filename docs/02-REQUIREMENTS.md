# 02 — Requirements

Every requirement has a stable ID. `docs/08-TESTING.md` maps each ID to the
test or measurement that proves it. **R-** = functional (built and tested),
**N-** = non-functional target (measured, reported met / not met).

Sources: **PS** = problem statement (repo copy corrupted; thresholds as last
transcribed, see Q-04), **RM** = roadmap PDF, **G** = golden rules.

## Functional

### Speech

| ID | Requirement | Source |
|---|---|---|
| R-01 | Text → speech through the API, with at least 5 open-source TTS engines available (Kokoro, XTTS-v2, OpenVoice V2 named) | PS, RM P1 |
| R-02 | A router picks the engine from language, speed and style; the response names the engine actually used | RM P1, G1 |
| R-03 | Zero-shot voice cloning from a 30–60 s reference clip | PS, RM P2 |
| R-04 | Cross-lingual cloning: a cloned voice speaks a language other than the reference's | RM P2 |
| R-05 | Forced alignment yields millisecond phoneme timestamps mapped to 15 visemes, and says whether timing was measured or estimated | RM P2, G1 |
| R-06 | Multilingual synthesis covering 1000+ languages (MMS-TTS) | RM P3 |
| R-07 | Emotion prosody presets (joy, anger, sorrow, authority, calm, excitement) and blends | RM P3 |
| R-08 | Speech quality auditor predicting MOS / PESQ / STOI | RM P3 |
| R-09 | Prosody controls: speed and pitch, 0.5×–2.0× | RM P2 |

### Vision

| ID | Requirement | Source |
|---|---|---|
| R-10 | Face analysis: 468+ landmarks, bounding box, head pose, blendshapes, background segmentation | RM P1 |
| R-11 | Face quality gate rejects no face, several faces, extreme pose, closed eyes, low resolution, with a reason | RM P1 |
| R-12 | Render job: audio + one registered face → lip-synced MP4 | RM P2 (Gate 2) |
| R-13 | At least two lip-sync engines selectable per job, with no silent fallback between them | RM P2, G1 |
| R-14 | Emotion vector drives the face (brows, cheeks, mouth corners, blink rate) | RM P3 |
| R-15 | Procedural avatar generation (Stable Diffusion) through the API | RM P3 |
| R-16 | Avatar customisation: background replacement in the render path | RM P3 |
| R-17 | Lip-sync quality metric (SyncNet LSE-C / LSE-D / offset) per rendered clip | RM §5 |

### Live

| ID | Requirement | Source |
|---|---|---|
| R-18 | Streaming TTS: audio delivered in chunks over a WebSocket as it is generated | RM P4 |
| R-19 | Live avatar: streamed chunks drive animated frames sent back over the same session | RM P4 (Gate 4) |

### Platform

| ID | Requirement | Source |
|---|---|---|
| R-20 | Every heavy step is an async job: QUEUED → PROCESSING → COMPLETED / FAILED, pollable by id | RM P0 |
| R-21 | Jobs survive an API restart | RM P4 |
| R-22 | API-key authentication and per-identity rate limiting | PS |
| R-23 | `/health` reports which model weights are actually present | G1 |
| R-24 | Creator studio UI: synthesise, pick a clone voice, pick emotion, see visemes, pick or register a face, see landmarks, render, play the video, see the sync score | RM P0–P3 |
| R-25 | Python SDK wrapping the public API | RM P6 |
| R-26 | Containerised deployment with a health check | RM P6 |
| R-27 | Structured logs carrying a request id | RM P6 |

### Ethics and provenance

| ID | Requirement | Source |
|---|---|---|
| R-30 | No voice or face is used without a provenance / consent record | G3, RM P5 |
| R-31 | A clone request whose reference is not admissible is refused with the reason | RM P5 |
| R-32 | Inaudible watermark embedded in generated audio, with a detector | RM P5 |
| R-33 | Invisible watermark and a signed provenance manifest on rendered video, with a verifier | RM P5 |
| R-34 | Visible "AI-generated" label on rendered video by default | RM P5 |
| R-35 | Consent audit trail: every use of a voice or face is logged with its basis | RM P5 |
| R-36 | Authenticity check endpoint reports whether a file carries this platform's marks | RM P5 |

## Non-functional targets

| ID | Target | Method | Source |
|---|---|---|---|
| N-01 | Speech quality MOS > 3.5 | SQUIM subjective, non-matching reference | PS |
| N-02 | Cloning similarity > 85% | ECAPA-TDNN cosine vs a human consented reference | PS (RM says > 90%, see Q-03) |
| N-03 | API job initiation < 500 ms | time to 202 on a running server | PS |
| N-04 | API capacity 100+ requests/min | load test against a running server | PS |
| N-05 | Lip sync "> 95%" | SyncNet: offset 0 and LSE-C, see D-12 / Q-02 | RM |
| N-06 | < 30 s to generate 60 s of video | end-to-end wall clock | RM final gate |
| N-07 | < 200 ms live latency | first frame after first chunk, measured | RM final gate |
| N-08 | 50+ concurrent tasks | concurrency test against the queue | RM final gate |
| N-09 | Temporal jitter < 2% frame-to-frame | landmark variance across output frames | RM §5 |
| N-10 | Peak VRAM < 6 GB per render | `torch.cuda.max_memory_allocated` | hardware |
| N-11 | Uptime > 99.5% | needs a deployed environment, see Q-07 | RM final gate |
