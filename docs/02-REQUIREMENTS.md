# 02 — Requirements

Every requirement has a stable ID. `docs/08-TESTING.md` maps each ID to the
test or measurement that proves it. **R-** = functional (built and tested),
**N-** = non-functional target (measured, reported met / not met).

Sources: **PS** = the problem statement, **read in full from the intact PDF on 8 Oct 2026** (an
earlier copy was corrupted and its thresholds had been transcribed from memory; the section
"Problem statement, reconciled" at the end lists every difference), **RM** = roadmap PDF,
**G** = golden rules.

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
| R-31 | A clone request whose reference may not be used (no provenance record, or a human voice without a consent basis) is refused with the reason. Synthetic voices are usable but never *admissible* as evidence (see D-30) | RM P5 |
| R-32 | Inaudible watermark embedded in generated audio, with a detector | RM P5 |
| R-33 | Invisible watermark and a signed provenance manifest on rendered video, with a verifier | RM P5 |
| R-34 | Visible "AI-generated" label on rendered video by default | RM P5 |
| R-35 | Consent audit trail: every use of a voice or face is logged with its basis | RM P5 |
| R-36 | Authenticity check endpoint reports whether a file carries this platform's marks | RM P5 |

## Non-functional targets

| ID | Target | Method | Source |
|---|---|---|---|
| N-01 | Speech quality MOS > 3.5 for efficient models and Milestone 1; **> 4.0 for premium models** (Higgs, Dia: cannot run here) | SQUIM subjective, non-matching reference | PS |
| N-02 | Cloning similarity: **> 85% at Milestone 1, > 90% as the final target** (both are in the PS) | ECAPA-TDNN cosine vs a human consented reference | PS |
| N-03 | API: status queries < 500 ms; generation initiation < 500 ms at Milestone 1, **< 2 s final** | time to 202 / status on a running server | PS |
| N-04 | API capacity 100+ requests/min | load test against a running server | PS |
| N-05 | Lip sync "> 95%" | SyncNet: offset 0 and LSE-C, see D-12 / Q-02 | RM |
| N-06 | < 30 s to generate 60 s of video | end-to-end wall clock | RM final gate |
| N-07 | < 200 ms live latency | first frame after first chunk, measured | RM final gate |
| N-08 | 50+ concurrent tasks (PS: 20+ simultaneous in the system targets, 50+ at Milestone 3) | concurrency test against the queue | PS, RM |
| N-09 | Temporal jitter < 2% frame-to-frame | landmark variance across output frames | RM §5 |
| N-10 | Peak VRAM < 6 GB per render | `torch.cuda.max_memory_allocated` | hardware |
| N-11 | Uptime > 99.5% | needs a deployed environment, see Q-07 | RM final gate |

## Problem statement, reconciled (8 Oct 2026, from the intact PDF)

The PDF in `docs/` is the owner's own copy; it has 20 pages and was read in full. What it changes:

**Corrections to what this project had recorded**

| Was recorded | The problem statement says | Effect |
|---|---|---|
| Cloning similarity target 85% (PS) vs 90% (roadmap), an open question (Q-03) | **Both, as tiers**: > 85% at Phase 1 / Milestone 1, **> 90% as the final benchmark** ("Quality Benchmarks", "Critical Performance Targets", "Success Metrics") | Q-03 resolved. XTTS-v2 62.0% misses both tiers; OpenVoice 34.2% misses both |
| API initiation < 500 ms | status queries < 500 ms; initiation **< 2 s** (final), < 500 ms at Milestone 1 | N-03 reworded |
| MOS > 3.5 | > 3.5 efficient / Milestone 1, **> 4.0 premium**; "Natural speech with MOS > 4.0, latency < 500 ms for real-time" | N-01 reworded; Kokoro 4.33 clears 4.0 as well, Higgs/Dia (premium) cannot run on this stack |
| Streaming latency "< 200 ms" only | also **"< 100 ms latency for streaming applications"** (Voice Synthesis Performance) and "< 500 ms" TTS real-time | new N-19: text to first audio is 425-461 ms warm: **not met** |
| Lip sync "> 95%" with no definition (Q-02) | still no measurement method, but three tiers: **> 90% across 10+ languages (Milestone 2)**, > 90% (Phase 2), **> 95% frame-level across all languages (final)** | N-05 needs the D-12 percentage computed over 10+ languages (T6.6); only English, Hindi, Tamil have been scored |
| Requirements exist only as a transcription (Q-04) | the real document | Q-04 resolved |

**New measurable targets the project had not been tracking**

| ID | Target (PS) | Status today |
|---|---|---|
| N-12 | Visual fidelity: **LPIPS < 0.1** | not measured (the `lpips` package is now installed, for VideoSeal) |
| N-13 | **Identity preservation: > 90% facial similarity** to the source image | not measured (SFace weights are fetched for it) |
| N-14 | Avatar from a single photo: **< 20 s (Milestone 2); < 10 s** (Avatar Engine) | not measured; photo registration is ~2 s; *generating* a face with Stable Diffusion is ~141 s on this CPU |
| N-15 | Customization: **50+ appearance parameters** with real-time preview | **not met**: 4 face attributes + background colour |
| N-16 | Output **1080p+**, optional 4K | **not met**: a render is capped at the photo's own size (no upscaling); no super-resolution |
| N-17 | TTS processing: **< 2 s for 30 s of audio** | not measured for a 30 s clip |
| N-18 | **20+ languages** supported; lip sync checked on 10+ | speech: 1,077 via MMS-TTS; lip sync scored on 3 |
| N-20 | Processing speed **< 5 s for a 30 s avatar video** (and "< 2 s generation for standard content") | not measured; the watermark made CPU renders ~3x real time, so on this machine **not met with the marks on** |
| N-21 | Temporal consistency > 95% (Milestone 2) | measured as jitter 0.28-0.35% (N-09); the 95% reading is not defined |

**Deliverables named in the PS that this project does not yet have** (each becomes a task in `07-TASKS.md`, M8):

| ID | Deliverable | Status |
|---|---|---|
| R-40 | **Voice-to-Avatar / audio + image to lip-synced video** (the user brings the audio) | not built: the renderer takes this platform's own synthesis; no audio upload |
| R-41 | **Real-time avatar with streaming *audio* input** | not built: live sessions take text only |
| R-42 | **Batch processing** of many avatar requests through the API | the queue handles many jobs; there is no batch endpoint |
| R-43 | **Style transfer** (realistic, cartoon, artistic) | not built |
| R-44 | **Quality enhancement**: super-resolution, lighting correction | not built |
| R-45 | **Custom voice training / fine-tuning** | not built (CPU-infeasible for XTTS-v2 here) |
| R-46 | **Deepfake detection integration** and abuse prevention (abuse detection, similarity-anomaly alerts) | partial: consent, rate limit, watermark, manifest; no detector, no alerts |
| R-47 | **Granular voice control**: emotion, accent, rhythm, pauses, intonation | emotion, speed, pitch only |
| R-48 | WebRTC streaming | WebSocket instead (D-46) |
| R-51 | **Monitoring**: real-time performance metrics and quality assessment | per-job timings and scores exist; no metrics endpoint |
| R-52 | **Documentation deliverables**: benchmarks, architecture, API docs, model integration guides, ethics, installation guide, tutorials, contribution guidelines | partly (docs 01-15); README and contribution guide pending (T7.6) |
| R-53 | **Testing across 20+ languages with native speakers, demographics, user studies** | not possible from here; automated coverage only |

Things in the PS this project already does: 5+ TTS models with automatic selection (Kokoro, XTTS-v2,
OpenVoice V2, Bark, MMS-TTS); zero-shot cloning from 30-60 s; cross-lingual cloning; 468-point landmarks
with 3D pose (478 with iris); phoneme-aware sync; emotion control; photo-to-avatar; procedural generation;
background customization; a real-time avatar; REST and WebSocket APIs with auth and rate limiting;
consent verification; invisible watermarking and provenance; quality metrics (MOS, PESQ, STOI, LSE-C/D, ECAPA).
