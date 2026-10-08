# 15 — The development journey

What was built, the problems met on the way and how each was solved, which models
were used (and which were not), and what the work taught. Written from the
project's own records (`12-PROGRESS.md`, `11-DECISIONS.md`, `context.md`,
`DEBUGGING.md`) and the measurements taken along the way; every number below
comes from one of them. Where something is not done or not met, it says so.

Contents: 1 The goal · 2 The solution · 3 Models · 4 Timeline · 5 Problems and
fixes · 6 Results against the targets · 7 Rules that shaped the work · 8 Mistakes
made and caught · 9 Lessons · 10 What is not done

---

## 1. The goal

An open-source **AI avatar platform**: a script, an optional voice sample and one
photo go in; a **talking, lip-synced avatar video** comes out, plus a **live
streaming avatar**. Every model is open source and runs locally. The brief was a
problem statement (requirements and thresholds) and a two-developer roadmap with
five integration gates:

| Gate | Meaning |
|---|---|
| 1 | Text becomes speech; a face detector maps landmarks on reference photos |
| 2 | A cloned voice and a single photo become an accurately lip-synced clip |
| 3 | A customised avatar speaking multiple languages with emotion |
| 4 | A live interactive avatar |
| 5 | Generated video carries verifiable watermarking and passes safety checks |

**The hardware set the shape of the whole design.** The target machine is a laptop
with an RTX 4050 (6,141 MiB of VRAM) and 14 GiB of RAM, shared with other work.
That forced: one heavy model resident at a time, lazy loading, models released
before the next loads, and a CPU path for everything (the development sandbox has
no usable GPU, so every "live" check here ran on the CPU and GPU-only numbers are
marked not measured).

## 2. The solution

```
 text ─► model router ─► speech ─► (emotion prosody) ─► WATERMARK ─► forced alignment
          Kokoro / XTTS-v2 /                                           (MMS_FA wav2vec2)
          OpenVoice V2 / Bark /                                              │
          MMS-TTS                                                            ▼
                                                              phonemes ─► 15 visemes ─► per-frame
 photo ─► consent check ─► MediaPipe 478 landmarks ──────────────────────► blendshape weights
          (provenance sidecar)        │                                          │
                                      ▼                                          ▼
                 background replace ─► portrait animator (mesh warp) ◄──────────┘
                 (selfie segmenter)             │       └─► optional Wav2Lip mouth (neural, GPU)
                                                ▼
                                    H.264 + AAC MP4 (+ "AI-generated" label) ─► SyncNet score
```

**The frozen contract.** `AvatarRenderJob` is the only audio-to-vision interface:
job id, avatar id, audio URL, sample rate, duration, phoneme timestamps (phoneme,
viseme, start ms, end ms), emotion vector, quality tier, fps. It was frozen first
and mock payloads were passed through it before either side was real, which is why
the speech half and the vision half fitted together later. Since then it has only
grown by *optional* fields (named emotions; `background`).

**Modules (backend/):**

| Area | Files |
|---|---|
| API and queue | `app.py` (REST + WebSocket), `contracts.py`, `job_queue.py` (in-process, SQLite-backed), `job_store.py`, `celery_app.py` (Redis path), `generation_jobs.py`, `security.py`, `request_context.py` |
| Speech | `voice_engine.py` (the router), `mms_engine.py`, `openvoice_engine.py`, `bark_engine.py`, `emotion_engine.py`, `language_registry.py`, `romanizer.py`, `audio_utils.py` |
| Timing | `alignment_engine.py` (MMS_FA forced aligner, viseme map), `quality_auditor.py` (SQUIM, ECAPA) |
| Vision | `face_engine.py` (MediaPipe), `face_animation.py`, `viseme_blendshapes.py`, `face_warp.py`, `wav2lip_engine.py`, `render_engine.py`, `video_io.py`, `avatar_store.py`, `avatar_generator.py` |
| Live | `live_engine.py` (sentence pipeline), the `/api/v1/live` socket in `app.py` |
| Trust | `provenance.py` (consent sidecars), `watermark_engine.py` (audio mark) |
| Operations | `model_registry.py` (weight audit), `gpu_utils.py` (VRAM and RAM guards), `lipsync_metric.py`, `jitter_metric.py` |
| Clients | `frontend/` (React + Vite studio), `sdk/` (Python client), `scripts/` (CLI tools) |

**Three flows exist end to end:**

1. *Batch*: `POST /audio/synthesize` → `POST /avatar/render-job` → poll → MP4 → `POST …/lipsync-score`.
2. *Cloning*: the same, with `mode=clone`, a consented reference in `inputs/`, and a `cloneEngine`.
3. *Live*: `WS /api/v1/live`: text in; per sentence an audio chunk, then JPEG frames, each stamped with when it is due.

## 3. Models

| Model | Role | Size on disk | Runs on | Licence / status |
|---|---|---|---|---|
| **Kokoro v1.0 (82M)** | fast English speech | 327 MB | CPU, fast | working; cold load ~7 s |
| **XTTS-v2** (Coqui) | zero-shot voice cloning, 17 languages | 1.87 GB (SHA-256 verified) | CPU 4.2 GiB RAM peak | **CPML, non-commercial**; accepted by the owner themselves, never by the tooling |
| **OpenVoice V2** | second cloner: recolours a base voice (Kokoro or MMS-TTS) | 131 MB | CPU 1.6 GiB | MIT; works across 1,077 languages through MMS |
| **Bark small** | two-speaker dialogue (`[S1]`/`[S2]`) | 1.7 GB | CPU 1.8 GiB; ~10× slower than real time | MIT; replaced Dia |
| **MMS-TTS** | 1,000+ languages, one checkpoint each | ~145 MB per language | CPU | 4 cached (hin, tam, swh, spa); others fetch on demand |
| **MMS_FA** (torchaudio wav2vec2) | forced alignment → phoneme timestamps | about 1.2 GB in memory (estimated from its parameter count, not measured separately) | CPU | measured, not estimated; shared across requests |
| **SQUIM** (torchaudio) | speech quality (MOS, PESQ, STOI estimates) | — | CPU | model estimates, always labelled with the method |
| **ECAPA-TDNN** (SpeechBrain) | speaker similarity for cloning | — | CPU | the only accepted similarity method |
| **MediaPipe** Face Landmarker, Selfie Segmenter, Multiclass Segmenter | 478 landmarks + 52 blendshapes; background cut-out; hair/clothes | 4 MB / 0.25 MB / 16 MB | CPU | MediaPipe 1.x *Tasks* API (`mp.solutions` is gone) |
| **Stable Diffusion 1.5** | synthetic avatar faces | 2.7 GB | CPU **6.6 GiB RAM peak**, ~141 s for one face | CreativeML OpenRAIL-M; faces are checked by the same quality gate as uploads |
| **Wav2Lip (GAN)** | neural mouth region | 436 MB | GPU intended; runs on CPU here | **research / non-commercial**; opt-in |
| **SyncNet v2** | lip-sync score (LSE-C / LSE-D / offset) | 55 MB | CPU | the project's lip-sync metric |
| **SFace** | face recognition features | 39 MB | CPU | fetched for the studio |
| **AudioSeal** (Meta) | inaudible audio watermark, 16-bit payload | 93 MB | CPU, 0.12 s per clip | MIT code and weights |

**Considered and not used, with the reason:**

| Model | Why not |
|---|---|
| **Higgs Audio / TTS 2 (3B)** | needs a transformers version newer than the pin Coqui TTS forces (`< 4.48`), and 11.6 GB of weights; the code now says "cannot run on this stack" instead of telling you to download it |
| **Dia-1.6B** | same pin conflict (it needs `nari-tts`, which needs numpy 2, which breaks Coqui); 12.9 GB. Bark replaced it |
| **LatentSync** | inference needs well over 6 GB of VRAM |
| **MuseTalk** | optional; not integrated |
| Kubernetes / GPU autoscaling | costs money, no measured need on one machine: Docker Compose is the target |

## 4. Timeline

**Before the plan (Sessions 1–6, 4 Sep – 3 Oct 2026; `context.md`).**
Sessions 1–2 (4 Sep): the first baseline, then the multi-model router with voice-clone
inputs and audio conversion. Session 3 (17 Sep) was an **audit that overturned earlier claims**
(see §5.1). Session 4 (18–29 Sep): provenance guard, face engine, project docs, a
`doctor` health script. Session 5 (30 Sep): the **vision pipeline end to end, first
talking avatar**, measured with SyncNet. Session 6 (3 Oct): status audit and a
config-precedence fix.

**The plan (7 Oct onward; `07-TASKS.md`).** Work was done task by task, each with
tests, a live run on real weights and a recorded result:

| Milestone | Tasks | Outcome |
|---|---|---|
| **M0** Foundation | repo hygiene, lint, types, frontend lint/build, Playwright harness, CI | gate script `scripts/check.sh`; ruff 668 raw findings → 70 under the chosen rule set → 0; 130 type errors → 0; E2E harness; CI green |
| **M1** (Gate 1) | E2E speech and landmarks, head-pose signs, honest router | **passed** |
| **M2** (Gate 2) | clone consent, Wav2Lip, clone → video, clone in the UI, five engines | demonstrated end to end; **sign-off waits on the owner's GPU and eyes (M-04)** |
| **M3** (Gate 3) | background replacement, avatar generation API, cross-lingual cloning, the Gate 3 flow, jitter metric | **passed** |
| **M4** (Gate 4) | streaming TTS, live frames, live UI, latency | **passed** |
| **M5** | audio watermark (measured; awaiting a listening check), video watermark + signed manifest, verify endpoint, consent audit trail | in progress |
| **M6** | job persistence, request ids, load and concurrency tests, security review | done except re-running benchmarks |
| **M7** | SDK, dependency audit, placeholder sweep done; Docker, documentation pass, README, final verification pending | |

## 5. Problems met, and how each was solved

Grouped by theme. Each entry: what was seen, the cause, the fix, and how it was proven.

### 5.1 Things that looked like success and were not

1. **A fallback nobody could see.** The forced aligner, when MMS_FA could not run,
   silently spread phonemes evenly across the text. The output had the same shape
   as a real alignment, so a *guessed* mouth timeline looked *measured*.
   Fix: every result records `alignmentMethod` (`mms_fa` vs `acoustic-fallback`),
   the API returns it, and the UI warns. *Lesson: a fallback must label itself.*
2. **A five-model router that was really two.** The router degrades to a fallback
   when a model is missing, and tests mocked the loaders, so for three phases
   Higgs, Dia and XTTS-v2 "worked" while having 0 bytes of weights. The Phase 3
   benchmark had only ever measured Kokoro and MMS-TTS. Fix: `model_registry.py`
   audits what is really on disk, startup prints it, `/health` serves it, and
   `scripts/fetch_models.py` re-audits after downloading.
3. **"Present" is not "complete".** The audit said XTTS-v2 was fine; its folder held
   1,299,857,408 bytes of a 1,867,929,118-byte file. The 30 Sep download had stopped
   at 70%. Fix: zip-format checkpoints must have their directory and a config; two
   rejecting tests. The finished download was later verified against Hugging
   Face's SHA-256.
4. **A router that would have downloaded 12 GB mid-request.** `from_pretrained`
   was called with no presence check, so the first `high_quality` request would have
   started a multi-gigabyte download while the log said it "falls back". Fix: a
   `require_weights` guard before any loader, a 503 that names the fetch command,
   `local_files_only` everywhere. Proven by comparing the cache folder's byte count
   before and after (unchanged).
5. **Tests that passed on local data.** A "real render" test was rendering the
   developer's own `demo.png`, not its fixture; CI (no such file) failed. Another
   test left a fake engine in a global and polluted later tests. Fixed, and the
   hermetic check now removes `.models/`, `inputs/` and the caches too.
6. **Environment variables outrank code.** `QUEUE_BACKEND=in_memory` promised no
   Redis, yet every task wrote to Redis, because Celery reads
   `CELERY_BROKER_URL` from the environment and that beats both constructor
   arguments and later config. It stayed hidden only because Redis happened to
   be running. Fixed, and verified with Redis stopped.

### 5.2 Lost or unreliable inputs

7. **The requirements PDF is unreadable.** Every byte above 0x7F was replaced with
   the UTF-8 replacement character (61,512 of them); all 132 compressed streams fail
   to inflate. It was moved through a text-mode transfer. The roadmap PDF is intact.
   Consequence: thresholds from the problem statement exist only as a transcription,
   and the project says so wherever it cites one (`Q-04`).

### 5.3 The pinned-versions trap

8. **Coqui TTS pins numpy < 2 and transformers < 4.48.** That one constraint rules
   out Higgs and Dia, ties which versions of everything else may be used, and is why
   four vulnerable packages could not be upgraded (§5.9). The response was not to
   fight it: unrunnable models are reported as unrunnable with "fix: none on this
   stack"; OpenVoice V2 was installed `--no-deps` at a reviewed commit so its stale
   pins were ignored; Bark replaced Dia; each engine's real memory use was measured.
9. **OpenVoice upstream bug**: `ToneColorConverter(enable_watermark=False)` raises
   `TypeError` (the argument is forwarded to a class that rejects it). Worked around
   and noted.
10. **A fallback that bypassed the guard.** When MMS-TTS failed, the OpenVoice path
    fell back to Higgs, after the preflight, which would have started the 11.6 GB
    download of a model that cannot run. It now raises an error naming the language.
11. **A silent fallback I introduced.** Bark's load failure fell back to Kokoro while
    `model_used` still said "bark". Caught while updating tests; `model_used` now
    names the engine that spoke.

### 5.4 Lip sync and animation

12. **The mouth barely moved.** MMS_FA emits ~20 ms CTC spans with gaps, and read
    literally the mean jaw opening was 0.05. Fix: hold each phoneme until the next
    one starts (max 240 ms).
13. **Wav2Lip led the audio by 120 ms.** SyncNet read offset −3 frames. Found by
    sweeping the mel window and watching the offset move by exactly one frame per
    step on three clips, then fixed as a named constant (`AUDIO_LEAD_SECONDS = 0.12`),
    because the root cause was not isolated. After: offset 0 on all three clips.
14. **Head-pose signs were wrong in the fallback.** Roll had the opposite sign, and
    yaw was measured along image axes so a pure 10° tilt read as 14° of yaw. The
    convention was established on real MediaPipe output with known-answer edits
    (rotations, keystone warps, mirroring) and the fallback fixed to measure in the
    face's own frame. A test that encoded the inverted sign was corrected by
    measurement, not weakened.
15. **The last video frame was trimmed.** ffmpeg's `-t <audio length>` dropped a final
    frame that started 10 ms before the cutoff, so the file was a frame shorter than
    its audio and warned about it. The owner's own run showed the warning. The cutoff
    is now the end of the last frame.
16. **An unexplained lip-sync offset** of −4 frames appeared on one short clip with a
    coloured background, against 0 on earlier clips. *Not explained; queued for the
    benchmark re-run.* It is recorded as open, not dismissed.

### 5.5 Voice cloning and consent

17. **Nothing checked consent on the cloning path.** Any WAV could be cloned. Fix: a
    provenance sidecar per recording; cloning refuses a file with no record or a
    human recording without a consent basis (403, with the reason). Voices and faces
    use the same rule.
18. **Cloning quality is far below target, and that is reported.** Scored by
    ECAPA-TDNN against speech the cloner never heard (clone from the first 30 s of
    a recording, score against the last 20 s; the same speaker's real speech scores
    91.7%): OpenVoice V2 **34.2%**, XTTS-v2 **62.0%**, XTTS-v2 cross-lingual
    Spanish 43.9%, Hindi 48.8%, French 42.8%. The target is 85%. An early API run
    that scored a clone against the *whole* reference gave a lower number than the
    unconverted base voice; unexplained, and the held-out method was adopted so no
    number is inflated by the prompt being inside the reference.
19. **The studio and XTTS disagreed about language codes.** The UI sends ISO-639-3
    (`spa`); XTTS-v2 wants `es`. Picking Spanish in clone mode crashed XTTS and
    `xttsSupported` called Spanish unsupported. Found by the Gate 3 E2E test; fixed
    in the language registry; a language XTTS cannot speak is now a 400 that names
    OpenVoice.

### 5.6 Frontend

20. **Stale responses overwrote newer ones** (language lookup). Fixed with an
    `ignore` flag in each effect's cleanup; the E2E race test fails without it.
21. **A `<textarea>` in a `<label>`** folds its value into the label's accessible name,
    so `getByLabel("Text")` never matched. Selecting by placeholder fixed it.
22. **A canvas read as black** in the browser tests because it was drawn before a
    frame was painted (`217` was exactly the distance of black from the target
    colour). The MP4 itself was correct. The read now seeks, waits and retries.

### 5.7 Performance, memory and concurrency

23. **The server was killed for lack of RAM** (exit 137) running the full browser
    suite. Measured, not guessed: Stable Diffusion adds 6.6 GiB on the CPU, XTTS-v2
    4.2 GiB; nothing was ever unloaded on a CPU host (the "one heavy model at a
    time" rule had only been enforced for VRAM); `release()` + `gc` left 1.2–2.4 GiB
    resident, ~1.0 after `malloc_trim`; and a fresh `ForcedAligner` reloaded the
    alignment model (about 1.2 GiB by parameter count) on every request. Fixes: a host-RAM guard with the
    measured sizes, `malloc_trim` on release, one shared aligner. Server peak fell
    from ~10.1 to 8.2 GiB, lowest free RAM rose from 1.3 to 3.1 GiB, and the suite
    got faster (72 s → 55 s).
24. **An `async def` that did CPU work froze the whole server.** Face analysis ran on
    the event loop; a trivial language lookup took 2.2 s whenever an analysis ran.
    It surfaced as a flaky browser test after a dependency upgrade. Fixed with a
    worker thread; the regression test uses one shared event loop (my first version
    could not fail because its clock started after the block).
25. **The speech router had no lock.** Two simultaneous syntheses shared model state.
    It now takes turns.
26. **A duplicate-id race in the queue** (two threads both see "id free") is guarded by a
    lock, shown by removing it and watching a 64-thread same-id test fail. Live: 60 jobs + 12 duplicate ids released
    together → 60 accepted and completed, 12 refused, none lost.
27. **A restart forgot every job.** Jobs now live in SQLite; after a restart finished
    jobs return, never-started ones re-run, and mid-render ones are failed with the
    reason. Proven with two real server processes.

### 5.8 Security (review of every control)

28. **The rate limiter could be bypassed by anyone** with auth off (the default): a
    fresh random `X-API-Key` on each request got a fresh bucket each time; **200 of
    200 requests were allowed** (a normal client gets 30), and memory grew with each.
    Fixed (a presented key counts only when auth is on); live: 30 of 200, same as
    a client with no key.
29. The audio endpoints accepted **any file in the project** (including `.env`); a
    voice-reference path was an **existence oracle for the host** (403 for a file
    that exists, 400 for one that does not); error bodies leaked absolute server
    paths; the dev compose published Redis (no password) and Postgres on all
    interfaces. All reproduced, fixed, each with a test that fails without the fix.

### 5.9 Dependencies

30. `pip-audit` found 10 vulnerable packages. Pillow (33 advisories; it decodes
    uploads), FastAPI/Starlette (14), PyJWT, urllib3, Werkzeug and multidict were
    upgraded; four could not be (transformers, diffusers, accelerate, nltk): each
    advisory is about loading or saving *untrusted* model repos, which this code
    never does (checked by grep), and each is recorded as an exception with the reason.

### 5.10 The watermark

31. **AudioSeal 0.2+ no longer resamples for you.** Fed 24 kHz audio it worked by
    accident. The engine now resamples to 16 kHz, marks there and adds the mark back.
32. **`torch.compile` recompiled for every clip length**: the first 5 s clip took
    42 s. With dynamo disabled, embedding takes 0.12 s.

### 5.11 The machine

33. **The host GPU is not used.** The RTX 4050 and its kernel driver are present, but
    the userspace libraries (`libcuda`, `nvidia-smi`) are not installed, so PyTorch
    sees no device. The owner's own run printed `Processing Device: CPU`. Fix needs
    `sudo apt install libnvidia-compute-595 nvidia-utils-595`; until then peak VRAM,
    N-06, N-10 and GPU latency are unmeasured.
34. **The machine is shared** (another project's Java services held ~4 GiB during the
    memory runs), which is why long jobs run detached and every memory number says
    what else was running.

## 6. Results against the targets

| Target | Result (CPU unless stated) | Status |
|---|---|---|
| N-01 Speech MOS > 3.5 | 4.33 (SQUIM, Kokoro); Bark dialogue 4.12 (self-referenced, biased up) | met |
| N-02 Cloning similarity > 85% | XTTS-v2 **62.0%**; OpenVoice 34.2%; cross-lingual 43–49%; real speech ceiling 91.7% | **not met** |
| N-03 Job initiation < 500 ms | render job → 202 in 4.5 ms mean; synthesis in the in-process queue is the whole synthesis (cold 7.2 s) | met for render; **not** for synthesis in this mode |
| N-04 100+ requests/min | default limiter admits 138/min; ~50,000/min raw (cheap read, one process) | met by policy |
| N-05 Lip sync | SyncNet offset 0; Wav2Lip LSE-C 9.30–11.18, blendshape 3.98–5.55; the "% within ±1 frame" figure not yet computed | measured; % pending |
| N-06 < 30 s for 60 s of video | needs the GPU | unmeasured |
| N-07 < 200 ms live latency | first audio → first frame **1.4–1.8 ms**; text → first audio ~0.43 s warm (7.3 s cold) | met as defined |
| N-08 50+ concurrent jobs | 60 jobs + 12 duplicates: none lost, none failed | met |
| N-09 Jitter < 2% | blendshape 0.278%, Wav2Lip 0.354% (control: 5 px shake reads 3.59%) | met |
| N-10 Peak VRAM < 6 GB | needs the GPU | unmeasured |
| N-11 Uptime > 99.5% | needs a deployment | not measured |
| R-32 Inaudible watermark | 26.6 dB under the speech; SQUIM MOS change +0.06 (worst −0.003); found after AAC 128 and 64 kbps, MP3, Opus, 8 kHz resample, trim, quieter volume (16/16 each); 20 dB noise 13/16; **lost** after 10 dB noise, a 4 kHz low-pass, a 10% speed change; false positives 0/55 | measured; *inaudible* awaits a listening check |

## 7. The rules that shaped the work

These are in `CLAUDE.md`; each was written because something went wrong without it.

1. **No silent fallbacks** (§5.1, 5.3). 2. **Every number records its method**: only
SQUIM MOS and ECAPA similarity against a human, consented reference count.
3. **Consent before use**: no voice or face without a provenance record; synthetic
faces are marked as such. 4. **The contract is frozen**; extend with optional fields only.
5. **Respect 6 GB** (and, as it turned out, 14 GiB of RAM). 6. **Tests never download
models**, and every feature still gets one live verification on real weights.
7. **Errors say how to fix themselves.** 8. **Say what is not done**: *built*,
*verified live* and *target met* are different. 9. **Inline explanation** in every file touched.

## 8. Mistakes made along the way, and caught

Kept here because they taught as much as the bugs in the code.

- **Claiming XTTS-v2 verified from a presence check** (3 Oct) when it was 70% downloaded.
- **Tests that could not fail**, found by *planting* the defect and watching for a failure:
  the pipelining test (compared against the last frame of all sentences), the
  event-loop test (started its clock after the block), the canvas read (drew before the
  frame was painted). Each was rewritten until the planted defect failed it.
- **Hiding a warning with my own filter** (`grep -v Warning`): the "82 frames rendered but
  the file holds 81" line was on my screen for hours.
- **An invalid proof**: a "restart" test where the kill missed and the old server
  answered. The "STILL UP" line gave it away; redone properly.
- **Commands that killed my own shell**: `pkill -f` / `pgrep -f` with a pattern that also
  appears in the command line itself (exit 144). Servers are now found by their listening port.
- **A fix that missed a case**: the first edit to the path check still left an
  existence oracle (a file elsewhere in the project got 400, a missing one 404).
- **Staging by directory** once, against the project's own rule (checked; nothing stray).

## 9. Lessons

- **Degrade loudly.** If a fallback returns the same shape as the real path, the caller
  cannot know it was cheated, so the fallback must say so.
- **Measure before fixing.** The 120 ms Wav2Lip lead, the memory growth, the blocked event
  loop and the limiter bypass were each *measured* before they were fixed, and each fix was
  proven by planting the old behaviour back.
- **A test that has never failed has proved nothing.** Plant the bug.
- **Pins are architecture.** One library's version constraint decided which models could
  exist here.
- **Presence is not completeness; passing is not proving.** Check sizes and hashes; check
  that the thing the test exercises is the thing that runs.
- **Hardware is a requirement.** 6 GB VRAM and 14 GiB RAM were designed for from the start;
  every later problem that mattered was a resource problem.
- **Say plainly what is not done.** Of the eleven numeric targets, N-01, N-08 and N-09 are
  met outright; N-03, N-04, N-05 and N-07 are met with stated caveats; N-02 is not met;
  N-06, N-10 and N-11 are unmeasured. Each row says exactly what is missing.

## 10. What is not done

- **The GPU numbers** (peak VRAM, N-06, N-10) wait for the owner to install two NVIDIA
  packages (M-04). Gate 2 is not signed off until then and until someone watches the video.
- **Voice similarity is below target** with both cloners; improving it is open work
  (longer or cleaner prompts, the cloner's conditioning settings, a second speaker).
- **Watermark**: video watermark, signed manifest, verify endpoint and the consent audit
  trail (T5.2–T5.4); inaudibility needs a listening check (M-06).
- **Docker** (T7.2), the line-by-line **code documentation** and code guide (T7.4), the
  **README** (T7.6), the benchmark re-run including the unexplained −4 frame offset
  (T6.6), and the **final verification** of every definition-of-done item (T7.7).
- The **problem-statement PDF** must be re-exported or its text pasted; until then its
  thresholds are the transcribed ones.
- Not measured anywhere: Celery + Redis in a real deployment, several live sessions at
  once, Wav2Lip live, anything on a GPU, a person using the live avatar (M-05).

*Where to look next:* `12-PROGRESS.md` (dated evidence for every claim above),
`11-DECISIONS.md` (D-01 to D-48), `08-TESTING.md` (requirement-to-test table),
`DEBUGGING.md` (known symptoms), `13-OPEN-QUESTIONS.md` (what only the owner can answer).
