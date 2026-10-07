# 07 — Tasks

Work strictly in order. A milestone is finished only when its gate passes:
full suite + E2E + production build. Status lives in `docs/12-PROGRESS.md`,
not here. Requirement IDs refer to `docs/02-REQUIREMENTS.md`.

Starting point (7 Oct 2026): 475 unit tests passing; text → speech → alignment
→ render job → lip-synced MP4 works by CLI and API on the blendshape engine;
XTTS-v2 and Wav2Lip weights present but never run; an admissible LJSpeech
reference exists; no lint, typecheck or E2E tooling.

---

## M0 — Foundation and quality gates

| ID | Task | Acceptance | Covers |
|---|---|---|---|
| T0.1 | Repo hygiene | `.gitignore` covers `CLAUDE.md`, `docs/`, `.claude/`, `.models/`, stray `*.md`; `docs/*`, `frontend/README.md`, `.models/*` untracked; unused `frontend/src/App.new.jsx` deleted. `git ls-files '*.md'` prints only `README.md` | DoD 13 |
| T0.2 | Dev tooling and gate script | `backend/requirements-dev.txt` (ruff, pyrefly, pip-audit); `scripts/check.sh {lint,types,test,frontend,e2e,all}` exits non-zero on any failure | DoD 1–6 |
| T0.3 | Python lint clean | `ruff check backend scripts tests` passes with a committed config; real bugs fixed, not suppressed | DoD 2 |
| T0.4 | Python typecheck clean | `pyrefly check` passes with a committed config; errors fixed in code, ignores only with a stated reason | DoD 3 |
| T0.5 | Frontend lint + build clean | `npm run lint` and `npm run build` pass with zero warnings | DoD 4–5 |
| T0.6 | E2E harness | Playwright installed; `frontend/e2e/smoke.spec.js` starts against the real backend and dev server and checks the studio loads and `/health` is ok | DoD 6 |
| T0.7 | CI | `.github/workflows/ci.yml` runs lint, typecheck, unit tests, frontend lint and build on push | — |

**Gate M0:** `scripts/check.sh all` green.

## M1 — Gate 1 sign-off: speech + landmarks, through the UI

| ID | Task | Acceptance | Covers |
|---|---|---|---|
| T1.1 | E2E synthesis | Playwright types text, synthesises with Kokoro, the audio element gets a source, phonemes appear | R-01, R-05, R-24 |
| T1.2 | E2E avatar overlay | Playwright selects `demo`, the landmark canvas draws, pose/blendshape numbers shown | R-10, R-24 |
| T1.3 | Head-pose signs | Unit + live: mirroring a face flips yaw sign; rotating in-plane by +θ changes roll by ≈ θ with the documented sign; convention written in the module | R-10 |
| T1.4 | Router honesty for missing engines | Asking for an engine with no weights returns 503 naming the fetch command; nothing tries to download; `/health` lists it unavailable | R-02, R-23 |

**Gate M1 (= roadmap Gate 1):** gates green; E2E proves text → speech and photo → landmarks in the UI.

## M2 — Gate 2: cloned voice → lip-synced video

| ID | Task | Acceptance | Covers |
|---|---|---|---|
| T2.1 | Live XTTS-v2 clone + similarity | Clone from `ljspeech_reference.wav` on real weights; ECAPA similarity measured against the admissible reference and recorded with method | R-03, N-02 |
| T2.2 | Consent before cloning | Clone with an inadmissible reference → 403 with the reason; admissible → accepted; tests for both | R-30, R-31 |
| T2.3 | Live Wav2Lip render | Wav2Lip renders `demo` on real weights; SyncNet scores recorded next to blendshape on the same audio | R-12, R-13, R-17, N-05 |
| T2.4 | Clone → video end to end | API: clone synthesis → render job → COMPLETED → MP4 → sync score, one script, evidence recorded | R-12, R-20 |
| T2.5 | Clone flow in the UI | Pick reference → synthesise cloned → render → video plays; E2E | R-24 |
| T2.6 | Five available TTS engines | Five engines with weights, each synthesising live; approach chosen in-task and logged (OpenVoice V2 if it installs under numpy<2, else fetch an open model that runs here) | R-01 |

**Gate M2 (= roadmap Gate 2):** gates green; a cloned voice drives a lip-synced video with a measured sync score.

## M3 — Gate 3: customised multilingual avatar with emotion

| ID | Task | Acceptance | Covers |
|---|---|---|---|
| T3.1 | Background replacement | Render option `background` (colour or image) composited with the selfie segmenter; unit + live MP4 | R-16 |
| T3.2 | Avatar generation API | `POST /api/v1/avatar/generate` queues an SD 1.5 face; result registered as synthetic; UI button; tests with the pipeline mocked | R-15 |
| T3.3 | Cross-lingual cloning | XTTS clone speaks a non-English language; ECAPA vs the reference recorded | R-04 |
| T3.4 | Gate 3 flow | Custom avatar + cloned multilingual voice + emotion → preview render; E2E | R-14, R-24 |
| T3.5 | Temporal jitter metric | Frame-to-frame landmark jitter computed for renders and recorded | N-09 |

**Gate M3 (= roadmap Gate 3).**

## M4 — Gate 4: live interactive avatar

| ID | Task | Acceptance | Covers |
|---|---|---|---|
| T4.1 | Streaming TTS | `WS /api/v1/live`: text in, audio chunks out as each sentence is ready; first-chunk latency measured | R-18 |
| T4.2 | Live avatar frames | Same session streams animated frames driven by each chunk's visemes | R-19 |
| T4.3 | Live UI + E2E | Start a session, send text, frames and audio arrive | R-24 |
| T4.4 | Latency measured | First-frame latency recorded with method | N-07 |

**Gate M4 (= roadmap Gate 4).**

## M5 — Gate 5: watermarking, provenance, consent trail

| ID | Task | Acceptance | Covers |
|---|---|---|---|
| T5.1 | Audio watermark | Every generated clip carries an inaudible mark; detector finds it, not on unmarked audio; MOS change measured | R-32 |
| T5.2 | Video watermark + manifest | Rendered frames carry an invisible mark; a signed manifest (inputs, models, consent basis) ships with each video | R-33 |
| T5.3 | Verify endpoint | `POST /api/v1/provenance/verify` reports marks and manifest validity; tampered manifest fails | R-36 |
| T5.4 | Consent audit trail | Every voice/face use appended with basis; `GET /api/v1/audit` | R-35 |

**Gate M5 (= roadmap Gate 5).**

## M6 — Hardening

| ID | Task | Acceptance | Covers |
|---|---|---|---|
| T6.1 | Job persistence | Jobs survive an API restart (SQLite store); test restarts the store | R-21 |
| T6.2 | Request ids + structured logs | Every log line of a request carries its id; id returned in a header | R-27 |
| T6.3 | Load test | Against a running server: initiation latency and sustained req/min recorded | N-03, N-04 |
| T6.4 | Concurrency test | 50+ concurrent submissions: no lost or duplicated jobs; throughput recorded | N-08 |
| T6.5 | Security review | Every item in `09-SECURITY.md` checked against code; findings fixed | R-22 |
| T6.6 | Benchmarks re-run | MOS, similarity, lip sync, generation speed, VRAM re-measured and dated | N-01, N-02, N-05, N-06, N-10 |

## M7 — Release

| ID | Task | Acceptance | Covers |
|---|---|---|---|
| T7.1 | Python SDK | `sdk/` package: synthesise, render, poll, score; tests against a mocked transport | R-25 |
| T7.2 | Docker | `Dockerfile` + root `docker-compose.yml`; `docker compose up --build`; `/health` 200 | R-26, DoD 7 |
| T7.3 | Dependency audit | `pip-audit` + `npm audit`; fixed or each exception logged | DoD 8 |
| T7.4 | Documentation pass (owner's spec, 8 Oct) | **Teaching-weighted** inline comments (module docstring; each new concept explained once; why-comments on non-obvious lines) in the 31 application files below 35% explanation density - 19 backend, 8 scripts, `App.jsx`, `AvatarPanel.jsx`, `main.jsx`, `playwright.config.js` (~11,200 lines; tests excluded). Plus `docs/14-CODE-GUIDE.md`: one request followed through the system, module by module, and a concepts index. Python verified comment-only by AST comparison; JSX by a byte-identical production bundle | rule 9 |
| T7.5 | Placeholder sweep | No TODO/FIXME/XXX/placeholder in tracked code | DoD 12 |
| T7.6 | README | setup, run, test, deploy | DoD 15 |
| T7.7 | Final verification | every DoD item run and reported | all |
