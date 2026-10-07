# 01 — Overview

## What it is

An open-source AI avatar platform. A creator gives it a script, optionally a
30–60 s voice sample, and one photo; it returns a video of that face speaking
the script in that voice, lip-synced. A live mode streams an animated avatar
back while text or speech arrives.

Every model is open source and runs locally. Nothing is sent to a hosted API.

## Who uses it

| User | Wants |
|---|---|
| Content creator | A talking-head video from a script without filming |
| Educator / trainer | The same lesson in several languages, one presenter |
| Support / corporate | A consistent branded presenter, emotion matched to the message |
| Developer | An HTTP API and a Python SDK to drive all of the above |

## The pipeline

```
Script + voice ─► Speech router ─► Forced aligner ─┐
                  (TTS / clone)    (15 visemes)    ▼
                                            AvatarRenderJob  (frozen contract)
                                                   │
Photo ─► Face analysis ─► consent gate ─► Lip-sync engine ─► Composite ─► MP4
         (478 landmarks, 52 blendshapes)            + watermark + provenance

Live:  text/audio chunks ─► streaming TTS ─► per-chunk visemes ─► frames over WebSocket
```

## Origin

Built from a two-developer roadmap (`AI_Avatar_Platform_2_Developer_Roadmap.pdf`,
on branch `m1`): Developer 1 owns audio and backend, Developer 2 owns vision
and UI, joined by a frozen render-job contract. Seven phases (0–6), each ending
in an integration gate. `docs/07-TASKS.md` maps the remaining work onto those
gates from where the code actually is.

## What "done" means here

The roadmap's final-gate numbers (< 200 ms live latency, 50+ concurrent jobs,
< 30 s per 60 s of video) were benchmarked on an RTX 4090 / A100. This project
runs on a 6 GB RTX 4050, and the sandbox it is built in has no GPU at all.
Every feature is built and verified; every target is measured and reported as
met or **not met** with its number. See `docs/11-DECISIONS.md` D-03.
