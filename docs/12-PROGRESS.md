# 12 — Progress

**Resume here.** Current task is the first one not marked Done. Each entry
records what was verified live, with the real command and result.

## Status

| Task | Status | Commit |
|---|---|---|
| T0.1 | Done (on main, before the branch rule) | 1ea9d51 |
| T0.2 | Done (on main, before the branch rule) | ab1b917 |
| T0.1–T0.2 on m1 | Done | 1065342 |
| T0.3 | Done | 3a603c2 |
| T0.4 | Done | 9c88e6d |
| T0.5 | Done (browser-verified in T0.6) | 8b935b2 |
| T0.6 | Done | d108e84 |
| T0.7 | Done | ebddb99, 8b1a910 |
| **M0 gate** | **Passed** | |
| T1.1 | Done | 94956b9 |
| T1.2 | Done | 0501aeb |
| T1.3 | Done | 0d68c57 |
| T1.4 | Done | b02a741 |
| **M1 gate** | **Passed** (= roadmap Gate 1) | |
| T2.1 | Done — measured, **N-02 not met** (62.0% vs 85%) | (this commit) |
| T2.2 | Done | 54686a2 |
| T2.3 | Verified on CPU — waiting on M-04 (peak VRAM on the host GPU, and a look at the video) | (this commit) |
| T2.6 | Done — 4 usable engines; the 5th (XTTS-v2) waits on M-03 | 6b94d3c, b55e538, f5d85bc |
| T2.4 | Done | (this commit) |
| T2.5 | Done (visual check of the video still in M-04) | 13f696c |
| T3.1 | Done | 63367d0 |
| T3.2 | Done | 031d475 |
| T3.5 | Done | ae92ce9 |
| T6.2 | Done | d62dd62 |
| T6.1 | Done | 0dfb64b |
| T7.1 | Done | 98902e3 |
| T7.3 | Done (4 documented exceptions, D-41) | ec5b180 |
| T7.5 | Done (two HTML `placeholder` attributes remain by decision D-40) | d62dd62 |
| T3.3 | Next | |

**Remaining order (owner asked for continuous building, small tasks first; the session is cleared between batches):**
T3.3 cross-lingual cloning (XTTS part blocked by M-03; OpenVoice cross-lingual already measured) ->
T3.4 Gate 3 flow (custom avatar + cloned multilingual voice + emotion, E2E) ->
T6.3 load test -> T6.4 concurrency test (50+ jobs) -> T6.5 security review ->
M4 live: T4.1 streaming TTS (`WS /api/v1/live`) -> T4.2 live frames -> T4.3 live UI + E2E -> T4.4 latency ->
M5: T5.1 audio watermark -> T5.2 video watermark + signed manifest -> T5.3 verify endpoint -> T5.4 audit trail ->
T6.6 re-run benchmarks (incl. the unexplained blendshape offset -4, see T7.1 entry) ->
T7.2 Docker (build is heavy: torch image) -> T7.4 documentation pass -> T7.6 README -> T7.7 final verification.
Waiting on the owner, cannot be done here: T2.1 (M-03 XTTS-v2 licence), T2.3 sign-off (M-04 host GPU + look at the video).
Environment gotchas learned: never `pkill -f` / `pgrep -f` with a pattern that also appears in your own
command line (it kills the shell, exit 144); find the server by `ss -ltnp 'sport = :8000'`.

## Log

### 2026-10-07 — planning set written

Read the roadmap PDF, the m1 docs (MILESTONES, PROJECT_DOCUMENTATION,
DEBUGGING, context) and the code. Wrote `CLAUDE.md` and docs 01–13.

Environment facts recorded for every later task:
- branch `main` at `685a317`, equal to `origin/main`; 475 unit tests pass.
- Tooling present: node 20.20.2, npm 10.8.2, oxlint, Docker 29.8.1 (daemon up),
  Playwright Chromium build 1234 cached. Absent: ruff, pyrefly, mypy, pip-audit,
  `@playwright/test`. Network to PyPI and npm works.
- `torch.cuda.is_available()` → False in this sandbox.

### 2026-10-07 — T0.1 repo hygiene (1ea9d51)

Untracked `docs/*`, `.models/*` (ECAPA entries were symlinks into the local HF
cache), deleted `frontend/README.md` and unused `App.new.jsx`. Verified:
`git ls-files '*.md'` → `README.md` only; MediaPipe task files still on disk and
covered by `fetch_vision_models.py`; ECAPA recreated by speechbrain
`from_hparams(savedir=.models/ecapa)`. 475 tests OK, doctor 33/4/0.

### 2026-10-07 — T0.2 dev tooling and gate script

`backend/requirements-dev.txt`, `scripts/check.sh` (lint/types/test/frontend/
e2e/hygiene/all). Local hooks installed (D-13). Verified:
- `scripts/check.sh bogus` → exit 2; `hygiene` failed (exit 1) on the old
  `.gitignore` lines, passes after the fix.
- `commit-msg` hook: `feat: x` → 1, a co-author trailer → 1, a plain message → 0.
- Baseline for T0.3: `ruff check` → 668 findings (529 auto-fixable).

### 2026-10-07 — owner review: develop on m1, main is production

Owner confirmed the plan with three changes: work from `m1`; working files
(`CLAUDE.md`, `docs/`) are tracked on `m1` and never on `main`; checks that
need a person or the host GPU go to the owner (Manual checks table in 13).

Ported T0.1/T0.2 to `m1`: `.models/` untracked, template README and
`App.new.jsx` removed, `requirements-dev.txt` and `scripts/check.sh` added,
`.gitignore` keeps docs tracked. `MILESTONES.md` and the old phase-1/2 plan
deleted as superseded (D-16). Hooks made branch-aware (D-13). Verified:
`check.sh hygiene main` → 0, `hygiene m1` → 1 (docs tracked, expected);
commit-msg hook: filename mention on m1 → 0, trailer → 1, `fix:` → 1.

### 2026-10-07 — T0.3 Python lint clean

`pyproject.toml` with the rule set in D-19. 70 findings → 0. Real defects fixed:
- `app.py`: two startup loops used `status` as the loop variable, making it
  local to the function and shadowing FastAPI's `status` module (F402).
- `app.py`: synthesis polling swallowed any exception and returned UNKNOWN with
  no log; `app.py` had **no logger at all**. Added one; the failure is now logged.
- 5 `except: pass` sites (native close, soundfile probe, optional Coqui import)
  now log at debug instead of vanishing.
- 6 `zip()` calls that must be length-matched get `strict=True` (silent
  truncation would drop words from the alignment or columns from a frame).
- `test_invalid_payload_is_rejected` asserted bare `Exception`; now
  `ValidationError`.
- 14 unused imports removed (none patched by tests); 5 scripts made executable.

Verified:
- `ruff check backend scripts tests` → "All checks passed!"
- unit suite → 475 OK.
- live: `render_avatar.py --text "Checking the lint changes on a real render."
  --face demo --device cpu --json` → exit 0, speech `kokoro`,
  `alignmentMethod = mms_fa` (strict zip did not trigger the fallback on real
  tokenizer output), engine `blendshape`, 75 frames / 3.0 s rendered in 3.35 s.

### 2026-10-07 — T0.4 Python typecheck clean

`pyrefly.toml` rewritten (explicit includes/excludes, `scripts` on the search
path, lenient sub-config for tests — D-20). 130 errors → 0. Production defects:
- `quality_auditor`: `_load_squim` may return either model as None but only
  `objective` was checked; a missing subjective model raised inside the try and
  produced a report labelled `dsp-estimate` that still held SQUIM's PESQ/STOI.
  Now either missing takes the honest fallback. New rejecting test.
- `alignment_engine`: annotation `tokenizer: any` used the builtin function.
- `romanizer` / `mms_engine`: uroman can be typed to return a list; added
  `as_text()` that raises on non-str. Review of my own first version found the
  check sat inside the old-uroman `except TypeError` shim, so a bad result
  would have been mistaken for an old signature and silently retried; moved
  outside the try. Test asserts exactly one call; **proved it fails on the
  earlier structure** (`AssertionError: 2 != 1`).
- `avatar_store`: `Image.LANCZOS` → `Image.Resampling.LANCZOS` (equal at
  runtime, verified); `exif_transpose` result narrowed.
- `job_queue.update`, `app` face-analyze: narrowing asserts for invariants the
  checker cannot follow. `lfilter` results pinned with `np.asarray`.
- `face_warp`: 9 OpenCV colours `255` → `(255,)`.
- `benchmark_phase3`: `timed()` made generic (TypeVar) — fixed 8 errors at once.

Verified:
- `scripts/check.sh` → ruff "All checks passed!", pyrefly "0 errors", 481 tests OK.
- gate can fail: a planted `def f() -> int: return "x"` → `check.sh types` exit 1;
  reverted → 0.
- live, real weights/data: real uroman `romanize("नमस्ते दुनिया","hin")` →
  `'namaste duniyaa'`; emotion tilt −0.45 / +0.35 → finite float32; Wav2Lip mel
  of real speech → (80, 181); SQUIM on `outputs/speech.wav` → method
  `torchaudio-squim`, MOS 4.46, PESQ 3.80, STOI 0.999.
- face_warp A/B: same aligned job (33 phonemes, 3.0 s) rendered with HEAD's and
  the new `face_warp.py` in separate processes → decoded video md5
  `6461f0c482dafc61` both, audio md5 identical. 75 frames pixel-identical.

### 2026-10-07 — T0.5 frontend lint and build clean

`npm run lint` failed (it scanned `.vite/deps`, React's pre-bundled copy, which
was also **tracked in git** — 8 files, now untracked and ignored). On `src/`
alone, 7 real warnings, and behind them real defects:
- race: language info, language search and the health check set state from
  responses with no cancellation, so an out-of-order slow reply could overwrite
  a newer one (wrong language shown). All now use an `ignore` flag in cleanup.
- `fetchSamples` read `selectedSample` and was rebuilt on every selection, so
  its effect carried an `eslint-disable` for the missing dependency. Now a
  functional update, deps `[]`, called from the mode-change handler.
- `AvatarPanel` reset analysis to null inside an effect (extra render, and the
  previous face could flash). Analysis is now stored keyed by avatarId and
  derived.
- polling effect missing `applySynthesisPayload` in deps.
Probe before fixing (scratch file): oxlint flags `useEffect(() => { load(); })`
when `load` sets state, but not inline async/`.then` setState with a flag.

Verified:
- `npm run lint` (now `--deny-warnings`) → exit 0; a planted warning → exit 1.
- `npm run build` → built in 1.05 s, no warnings.
- Browser behaviour: verified by the T0.6 E2E run (below) — all five pass,
  including the race test, which fails when the fix is removed.

### 2026-10-07 — T0.6 E2E harness

`@playwright/test` 1.63.0, Chromium 1243 downloaded. `frontend/playwright.config.js`
starts the real backend (uvicorn :8000) and Vite (:5173) itself.
`frontend/e2e/smoke.spec.js`, 5 tests: studio heading + `ok (in_memory)` health;
language catalogue populated from the API; clone mode loads references
(`ljspeech_reference` present); a delayed lookup for one language cannot
overwrite a newer one; `demo` selected with "478 landmarks · yaw" and the canvas.

Verified:
- `npx playwright test` → **5 passed (14.7 s)** against real servers.
- The race test can fail: with `if (!ignore && data)` reverted to `if (data)` in
  the language-info effect, it fails ("element(s) not found" — the stale
  language replaced the newer one); restored → passes. This is also the live
  browser check for T0.5.
- Seen in the backend log during the run: Higgs/Dia "will fall back to another
  model" at startup — that is T1.4's subject.

### 2026-10-07 — T0.7 CI

`.github/workflows/ci.yml`: backend job (CPU torch 2.5.1, pinned requirements,
`TTS==0.22.0 --no-deps`, dev tools) runs `scripts/check.sh lint|types|test`;
frontend job runs `npm ci` + `scripts/check.sh frontend`. E2E deliberately not
in CI: it needs `inputs/` (consent-gated faces and voices). `check.sh` takes
`PY`/`BIN` from the environment so CI runs the same script. Repo is public, so
Actions minutes cost nothing.

Run 1 (37631873390, ebddb99): frontend green; backend lint + typecheck green,
**unit tests failed** — 5 errors, 1 failure:
- MediaPipe's native lib needs `libEGL.so.1` (not on the runner) → apt
  `libegl1 libgles2`.
- `test_real_render_through_the_api` patched the API's avatar store but not
  `avatar_store.FACES_DIR`, the default root render_engine's own
  `AvatarStore()` reads. **Locally it had been rendering the real
  `inputs/faces/demo.png`, not its fixture.** Reproduced by moving
  `inputs/faces` aside (same AvatarNotFound), fixed by patching FACES_DIR, then
  25/25 vision API tests pass without the real faces.

My T0.7 "hermetic" pre-check had removed `.models/` and the caches but left
`inputs/`, which is why it missed this. Re-ran it properly — no `.models/`, no
`inputs/`, empty HF/torch/XDG, offline → 481 OK (1 skipped). Note: `app.py`
creates an empty `inputs/` at import; that is the voice-sample folder,
gitignored, not a data leak.

Run 2 (37632834342, 8b1a910): **success** — backend 481 tests OK (skipped=1),
frontend lint + build OK.

### 2026-10-07 — M0 gate passed

`scripts/check.sh all` (local, before run 2): ruff clean · pyrefly 0 errors ·
481 unit tests OK · frontend lint (deny-warnings) + build · E2E 5 passed (10.8 s)
· `hygiene main` ok. CI green on m1. The "Render job failed", "SQUIM inference
failed (cuda oom)" and Redis "connection refused" lines in test output are
tests exercising failure paths on purpose.

### 2026-10-07 — T1.1 E2E synthesis

`frontend/e2e/synthesis.spec.js`: type text → Generate Speech → audio element
gets `/outputs/speech.wav?t=…` → badge says `kokoro` → "Aligned Phonemes (N)",
N > 5 → no "estimated, not measured" warning → the WAV behind the player is a
real RIFF file > 24 000 bytes.

Verified: **1 passed (22.0 s)**, real Kokoro on CPU through the real API.
First attempt failed on the selector, not the app: a textarea inside its
`<label>` gets the textarea's *value* folded into its accessible name, so
`getByLabel("Text", {exact})` never matches; now by placeholder.
Can fail: with `_mms_failed = True` planted in `ForcedAligner.__init__`, the
test fails on `toHaveCount(0)` for the fallback warning (Received: 1); restored.

### 2026-10-07 — T1.2 E2E landmark overlay

`frontend/e2e/avatar.spec.js` reads the canvas pixels (the photo loads with
`crossOrigin` and the API sends CORS, so the canvas is not tainted): counts mesh
green `#6ee7b7` and iris pink `#f472b6`, checks the mesh centroid lies inside
the face box the API reports, and that unchecking "show landmarks" removes it.

Verified, real MediaPipe on `demo`: **1 passed (3.4 s)**. Numbers:
"478 landmarks · yaw -0.58° · pitch 4.98° · roll 1.21° · 52 blendshapes"
(30 Sep recorded -0.6 / +5.0 / +1.2 — consistent); 1117 mesh pixels on the
512×512 canvas, centroid (360, 325); face box x 245–460, y 187–454 → inside.
Can fail: drawing `ctx.arc(point.x, point.y, …)` without scaling by the image
size (all dots in the corner) → "Expected > 500, Received 1"; restored.

### 2026-10-07 — T1.3 head-pose sign convention

No photo at a measured angle exists, so the convention was established on real
MediaPipe output of `demo` with known-answer edits: in-plane rotation (truth:
the angle), keystone warps (the 2D projection of tilting the face plane about
one axis — shrink the right edge = face turned toward image-right; shrink the
bottom = looking down), and mirroring (must negate yaw and roll, keep pitch).

Matrix pose, measured:

| edit | yaw | pitch | roll |
|---|---|---|---|
| frontal | -0.58 | 4.98 | 1.21 |
| face toward image-right | **+2.02** | 5.26 | 1.78 |
| ... mirrored | **-2.19** | 5.72 | -1.83 |
| face toward image-left | **-2.33** | 5.99 | 0.78 |
| looking down | 0.43 | **6.13** | -0.33 |
| looking up | -0.58 | **3.06** | 2.45 |
| rotated +10° CCW | -0.48 | 4.62 | **11.22** |
| ... mirrored | 0.10 | 5.16 | **-11.43** |
| rotated -10° CW | -0.48 | 4.68 | **-8.89** |

Convention (now in `HeadPose`'s docstring): **+yaw = facing image-right (the
subject's own left); +pitch = looking down; +roll = counter-clockwise on
screen.** Roll is unambiguous (Δ ≈ ±10 for ±10°). Yaw and pitch shifts are
small (a flat warp is a weak 3D cue, ~2°) but consistent in both directions
and under mirroring.

**Bug found in the landmark-geometry fallback** (used when no matrix):
1. roll had the opposite sign: +10° CCW → matrix 11.22, geometry **-10.18**;
2. yaw/pitch were measured along image axes, so a pure 10° tilt read
   **14.12° of yaw** (matrix: -0.48).
Fixed by measuring in the face's frame (along the eye line and perpendicular to
it) and negating roll for image-y-down. After: CCW 11.22 / **10.18**, CW
-8.89 / **-10.04**; tilted-frontal geometry yaw 3.39 and 4.03 vs frontal 4.37.
Geometry keeps a constant bias (+3–5°), expected for the approximation.

Tests: `test_tilted_eyes_produce_roll` expected **positive** roll for a
clockwise tilt — it encoded the inverted sign; changed to expect negative
because the measured convention says so (D-25). New: CCW = positive roll,
rigid tilt moves roll by exactly +10 and leaves yaw/pitch unchanged,
nose-right = positive yaw. On HEAD's `face_engine.py`, 3 of the 4 fail.
Gates: ruff clean, pyrefly 0, 484 tests OK. Only the UI displays the signs;
the quality gate uses absolute values, so no behaviour beyond that changes.
Optional owner confirmation with a real turned head: Manual check M-01.

### 2026-10-07 — T1.4 router honesty for engines without weights

Found: `load_higgs` / `load_dia` call `from_pretrained` with no presence check
and no `local_files_only`. With the network up, the first `high_quality`,
`quality=high` (a UI option) or dialogue request would have **started a
multi-GB download mid-request**, while the startup log claimed such requests
"fall back to another model" (they only fell back after a failed download).

Fix: `VoiceEngineRouter.require_weights(key)` raises `ModelWeightsMissing`
(message names `python scripts/fetch_models.py --only <key>`), audited fresh
per call (5.1 ms, filesystem only) so weights fetched while running are seen.
Called in `synthesize()` right after `select_model` — the single point every
synthesis path passes before a loader — and in the API route before queueing:
503 for missing weights, 400 for unroutable requests (was: a queued job that
failed later). Startup log now says requests "are refused until it is fetched".
Runtime *load failures* (weights present, e.g. OOM) keep the existing visible
fallback (D-26).

Test isolation found on the way: `test_synthesis_task_*` patched the router
class and left a FakeVoiceEngine as the process-wide `celery_app._router`
singleton for every later test; harmless until the route used `get_router()`.
Scoped with `patch.object(celery_app, "_router", None)`. Three tests depended
on local weights — now use a fake audit (`weights_on_disk()`).

Verified:
- 490 tests OK; hermetic (no .models, no inputs, empty caches, offline) 490 OK.
- Can fail: with the guard line in `synthesize()` removed, both
  "never reaches the loader" tests fail; restored.
- live server, network up: high_quality → **503** "Fetch it with: python
  scripts/fetch_models.py --only higgs-tts-2"; dialogue → 503 (dia-1.6b);
  fast + quality=high → 503; multilingual `zzz` → **400**; fast → **202**,
  `kokoro`, audio produced. Higgs/Dia HF cache dirs 1188/1257 bytes before and
  after — **no download**; 0 download lines in the server log. `/health`:
  higgs-tts-2 and dia-1.6b present=false.
Observed for T6.3: in `in_memory` mode the synthesis POST runs the task eagerly,
so its 202 arrives after synthesis finishes — initiation time there is the full
synthesis time, not queueing time.

### 2026-10-08 — resumed; scope confirmed; M1 gate passed

The previous session ended mid M1-gate run. On resume: tree clean, m1 =
origin/m1 = b02a741, CI run 37637081306 for b02a741 **success**, no stale
servers on :8000/:5173.

Owner asked whether the plan covers one developer's roadmap or both; answer:
both (git history: the teammate has one commit, the Phase 0 mock renderer; all
later work on both pillars came from this account). Owner chose **both** (D-28).

`scripts/check.sh all` → exit 0: ruff clean · pyrefly 0 · 490 tests OK ·
frontend build · **E2E 7 passed (27.1 s)** · hygiene main ok.
Roadmap Gate 1 ("text input generates speech audio; face detector maps 468
landmarks on reference photos") is met through the UI: `synthesis.spec.js`
(real Kokoro speech, measured alignment) and `avatar.spec.js` (478 landmarks
drawn inside the face box, measured from canvas pixels). Optional human look:
M-02.

### 2026-10-08 — T2.1 blocked: the XTTS-v2 weights were never complete

Before loading XTTS-v2 I checked how Coqui gates its licence: without
`tos_agreed.txt` it prompts on stdin, and `COQUI_TOS_AGREED=1` accepts on the
user's behalf — not mine to set. The model folder then turned out to hold only
`model.pth`, **1,299,857,408 bytes, not a valid zip**, against the published
1,867,929,118; `config.json`, `vocab.json`, `speakers_xtts.pth`, `dvae.pth`,
`hash.md5` absent. The 30 Sep download stopped at 70%.

Correction to the record: on 3 Oct I marked G1-08 "Verified (weights
present)" from `doctor.py`'s presence check, and this plan counted XTTS-v2 as
an engine with weights. Neither was true; presence is not completeness.
Consequence: T1.4's guard trusted that audit, so a clone request would have
gone to Coqui, which would re-download and block on its licence prompt.

Fixed `check_coqui_model` (D-29): `config.json` required and zip checkpoints
must have their directory. Old test fixture *was* the broken case (model.pth
alone, expected present) — rewritten as a real complete download, plus
rejecting tests for "no config" and "truncated at 70%". Live: audit → present
False, "config.json is missing, so the download did not finish"; doctor →
`WARN xtts-v2 … fix: scripts/fetch_models.py --only xtts-v2`, 32 pass / 6 warn
/ 0 fail. 492 tests OK.

Integrity sweep of every other weight (HF tree sizes via the API, local files
by format): Kokoro, MMS hin/spa/swh/tam, ECAPA, SD 1.5 — all sizes match;
`wav2lip_gan.pth` loads in torch; SyncNet, SFace, MediaPipe ×3 well-formed.
**XTTS-v2 is the only broken one.** Engines with usable weights: 2 (Kokoro,
MMS-TTS).

Unblocking needs the owner (M-03: accept the CPML and run the fetcher).
Meanwhile: T2.2, T2.3 and T2.6 do not depend on it; T2.4/T2.5 do.

### 2026-10-08 — T2.2 consent before cloning

Found: nothing on the cloning path checked provenance. `provenance.usability`
existed and the avatar store used it for faces, but any WAV could be cloned.

`require_voice_consent(path)` (missing file → FileNotFoundError; no sidecar →
VoiceConsentRequired naming `scripts/make_reference.py`; human without a consent
basis → refused with the reason; synthetic or consented human → ok).
`VoiceEngineRouter.preflight(model_key, speaker_wav)` runs consent then weights,
consent only for engines that read the reference (XTTS-v2, Higgs). Called by
`synthesize()` and by the API route before queueing: 403 / 503 / 400 (D-30).

Verified:
- 11 new tests (8 `VoiceConsentTests`, 3 API); 503 OK. With the
  `require_voice_consent` call removed from `preflight`, 7 of them fail; restored.
- live server, clone requests:
  | reference | result |
  |---|---|
  | wav with no sidecar | **403** "has no provenance record … make_reference.py" |
  | human sidecar, no basis | **403** "consent basis '' is not one of [...]" |
  | `ljspeech_reference.wav` (human, open-licence) | consent passes → **503** XTTS-v2 incomplete |
  | `smoke_reference_kokoro.wav` (synthetic) | consent passes → **503** |
  | missing file | **400** "Reference audio not found" |
  Temporary test WAVs removed from `inputs/`.
The successful clone itself is exercised in T2.4, once XTTS-v2 is complete (M-03).

### 2026-10-08 — T2.3 Wav2Lip live, and a 120 ms lip lead found and calibrated out

`wav2lip_gan.pth` on real weights, CPU (no GPU here). Same aligned 3.0 s job
rendered with both engines: blendshape LSE-C 3.85 / LSE-D 10.71 / offset 0;
Wav2Lip LSE-C 8.84 / LSE-D 6.79 / **offset −3 frames (−120 ms)**, with SyncNet
warning the streams may be misaligned. By the 30 Sep controls (audio delayed
200 ms → −5), −3 means the mouth moves ~120 ms *before* the sound.

Cause hunted by measurement rather than by reading the reference code: the mel
window starts at each frame (reference inference convention). Sweeping it k
frames earlier moved the offset by exactly +1 per frame on three clips (two TTS
engines, two languages):

| clip | k=0 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| en 3.0 s Kokoro | −3, 9.01 | −1, 10.05 | **0, 10.20** | — |
| en 6.4 s Kokoro | −3, 10.62 | −1, 10.93 | **0, 10.99** | +1, 10.83 |
| hi 5.0 s MMS | −3, 11.52 | −1, 11.75 | **0, 11.71** | +1, 11.80 |
(offset frames, LSE-C)

Blendshape reads offset 0 on the same clips, so the lead is in the Wav2Lip
path, not the metric. Root cause not isolated (window convention vs STFT
centring); fixed as a named calibration constant `AUDIO_LEAD_SECONDS = 0.12`
(time-based, so fps-independent) with the evidence in its comment (D-31).
Tests: reference indexing kept as the explicit `lead_seconds=0` case; new test
for the default (frame 10 → column 22; early frames clamp).

Final, committed code, no patching — `render_job` + `score_video`:

| clip | engine | frames | render (CPU) | offset | LSE-C | LSE-D |
|---|---|---|---|---|---|---|
| en 3.0 s | blendshape | 75 | 3.27 s | 0 | 3.98 | 10.55 |
| en 3.0 s | **wav2lip** | 75 | 5.02 s | **0** | **9.84** | **6.61** |
| en 6.4 s | blendshape | 160 | 1.02 s | 0 | 4.23 | 11.56 |
| en 6.4 s | **wav2lip** | 160 | 4.97 s | **0** | **10.51** | **5.97** |
| hi 5.0 s | blendshape | 126 | 0.95 s | 0 | 5.55 | 10.20 |
| hi 5.0 s | **wav2lip** | 126 | 4.01 s | **0** | **11.18** | **5.48** |

Real talking-head video scores LSE-C ≈ 6–8; Wav2Lip exceeds it, as expected of
a model trained against a SyncNet discriminator (so LSE-C alone flatters it —
which is why the visual check M-04 matters). Gates: 504 tests OK, lint, types.

Not measurable here: peak VRAM (acceptance says < 6 GB) — M-04.

### 2026-10-08 — T2.6 (a): Higgs and Dia cannot run here at all

Before choosing how to reach five engines, checked each candidate with evidence:

| engine | HF size | licence | `AutoConfig` under transformers 4.47.1 |
|---|---|---|---|
| Higgs TTS 2 3B | 11.58 GB (`model.safetensors` 11.5 GB fp32) | other | **fails**: model type `higgs_audio_v2` not recognised |
| Dia-1.6B | 12.89 GB (two 6.4 GB copies) | apache-2.0 | **fails**: no `model_type` in config (needs `nari-tts` → numpy 2) |
| OpenVoice V2 | 0.13 GB (converter 131 MB) | MIT, ungated | n/a (own package) |

So the router declared five engines of which two could never load on this stack
— the transformers <4.48 and numpy <2 pins are forced by Coqui TTS — and their
weights exceed the 6 GB card anyway. Everything told the operator to fetch them:
`doctor.py`, the 503 body, the startup log, and `fetch_models.py --all` (which
also understated Higgs as 6.5 GB). `requirements.txt` claimed "~3 GB VRAM".

Fix (D-32): `ModelWeightStatus.fix`; `model_registry.UNRUNNABLE` holds both
reasons; the audit reports them unavailable with "fix: none on this stack";
`ModelWeightsMissing` says "Fetch it with" only when fetching is the fix; doctor
prints the registry's fix; the fetcher refuses both keys with the reason;
requirements comments corrected. Log wording "MODEL WEIGHTS MISSING" →
"MODEL UNAVAILABLE … Fix: …" (test updated to match, still one warning per model).

Verified: 4 new tests, 508 OK. Live: doctor → `WARN higgs-tts-2 … cannot run on
this stack … fix: none on this stack`; `fetch_models.py --only dia-1.6b` → the
reason, exit 1, nothing downloaded; real server: startup log "MODEL UNAVAILABLE",
`high_quality` and `dialogue` → 503 "… Downloading it would not help. Fix: none
on this stack (see detail); use another engine".

### 2026-10-08 — T2.6 (b): OpenVoice V2 as a working cloner

Installed without its stale pins (`--no-deps`, commit 74a1d14, code read first)
plus `eng_to_ipa` and `cn2an`; numpy/transformers/librosa/torch pins verified
unchanged. Converter weights (131 MB, MIT, ungated) sizes match published.
Upstream bug worked around: `ToneColorConverter(enable_watermark=False)`
raises TypeError (the kwarg is forwarded to a base class that rejects it).

`backend/openvoice_engine.py`: lazy load (local files only, cached failure
naming the fetch), per-reference embedding cache keyed on (path, mtime, size),
seeded conversion scoped with `torch.random.fork_rng`, output resampled to
24 kHz. Router: `cloneEngine` request field (`xtts-v2` default | `openvoice-v2`,
explicit — never switched silently); base voice Kokoro for English, MMS-TTS for
its 1077 languages (so cross-lingual cloning); preflight checks consent, the
converter and the base voice. Audit/doctor/fetcher know `openvoice-v2`.

Also fixed: when MMS-TTS failed, `_synthesize_mms` fell back to Higgs — after
the preflight, so it bypassed T1.4's guard and would have started an 11.6 GB
download of an unrunnable model — and set `_mms_failed`, disabling MMS for every
language. Now it raises that request's error naming the language.

Verified:
- 526 tests OK (18 new: engine, routing, preflight, consent, MMS failure, audit,
  API). Planted regressions caught: base-voice check removed → fails; old MMS
  fallback restored → fails.
- live, real server: clone EN → 202 SUCCESS, `openvoice-v2 (base: kokoro)`,
  4.92 s, alignment mms_fa, 58 phonemes, 30.6 s on CPU (cold); clone **Hindi**
  → 202 SUCCESS, `openvoice-v2 (base: mms-tts)`, 4.88 s, mms_fa, 63 phonemes.
- **Similarity (ECAPA-TDNN, admissible human reference, CPU), 6 sentences,
  cloned from the first 30 s of the LJSpeech reference, scored against the
  held-out last 20 s:** unconverted Kokoro base mean 26.9% → converted
  **34.2%** (+7.3, sd 3.1), improved on 6/6. Same speaker's real speech: 91.7%.
  **N-02 target (85%) not met by OpenVoice V2.**
- Open finding: via the API, cloned from *and* scored against the full 50 s
  reference, the EN clip scored 21.3% (base 24.7%) — lower than unconverted. Not
  explained yet; to investigate in T6.6 before any number is published.

### 2026-10-08 — documentation scope set, deferred to T7.4

Owner asked whether every line is explained. Measured: 74 source files; 8
application files fully documented (contracts, job_queue, celery_app,
security, gpu_utils, romanizer, openvoice_engine, audio_utils); 31 below 35%
explanation density (~11,200 lines). Owner chose teaching-weighted, inline + a
code guide, application code only, and to do it at the end (T7.4, D-34).

### 2026-10-08 — T2.6 (c): Bark for dialogue (built; live check pending)

Dialogue had nowhere to go: Dia cannot run here. Bark small: MIT, ungated,
1.68 GB `pytorch_model.bin`, `BarkModel` native in transformers 4.47.1; full
Bark (4.5 GB) would not fit beside other models on 6 GB. `bark_engine.py`:
`split_dialogue` ([S1]/[S2] turns; untagged text = speaker 1), `chunk_turn`
(≤220 chars at sentence ends, Bark's ~13 s limit), presets loaded as arrays
from the snapshot, seeded with `fork_rng`, 0.3 s gaps, peak-normalised. Router:
dialogue → `bark`; Dia's loader/synthesis removed as unreachable; a Bark load
failure falls back to Kokoro **and `model_used` says kokoro** (my first
version left it saying "bark" — a silent fallback, caught while updating the
old Dia tests). Audit `check_bark` requires the model files and all six preset
files (`check_hf_files`, shared with OpenVoice); fetcher spec `bark`.

Tests: 10 new (`test_bark_engine.py`) + router updates; 7 old tests encoded
"dialogue → Dia" and were rewritten to the new routing (D-35); the one Dia-only
loader test was removed with the code it tested. 535 OK. Can fail: speaking every
turn with speaker 1's preset → the dialogue test fails; restored.

Download: network to huggingface.co was intermittent (DNS failures, 15 s
responses); 1.6 GB arrived, one preset (`en_speaker_9_semantic_prompt.npy`)
still incomplete — and the audit correctly reports Bark **not present** until
it lands. Live verification (CPU timing, SQUIM MOS, two audible voices) is next.

### 2026-10-08 — T2.6 (c) verified live; T2.6 done

Download completed after DNS trouble; audit → `bark` present, 1.7 GB.
Live, real server, `mode=dialogue`, "[S1] Good morning, did you sleep well?
[S2] Not really, the storm kept me awake all night.", `auditQuality`:
202 SUCCESS, model `bark`, 9.51 s of audio, alignment `mms_fa`, 66 phonemes,
97.5 s on CPU (cold load included). SQUIM MOS **4.12** / PESQ 2.48 / STOI 0.97 —
self-referenced (no separate reference given), so biased upward; noted as such.
Two distinct voices, ECAPA on separate lines: same speaker 67.8% (S1) / 54.5%
(S2); across speakers 13.6% / 17.7%. Bark runs ~9–10× slower than real time on
CPU (3.6 s of audio in 32.7 s).

**T2.6 outcome:** usable engines with weights here — Kokoro, MMS-TTS, OpenVoice
V2, Bark (4). XTTS-v2 is the 5th once the owner completes M-03; it is also one of
the three the problem statement names, so R-01 needs it regardless. Higgs and
Dia cannot run on this stack (D-32).

### 2026-10-08 — T2.4 cloned voice → lip-synced video, through the API

`scripts/clone_to_video.py`: standard-library client of the public API — clone
(`POST /audio/synthesize`, polls under Celery) → frozen `AvatarRenderJob` from
the response (audio addressed by the server's own `/outputs/` URL) → `POST
/avatar/render-job?engine=` → poll → `POST …/lipsync-score` → `POST
/audio/voice-similarity`. Exit 0 / 1 with the server's own error text.

Verified, real server, real weights, CPU:
`clone_to_video.py --text "Hello, this is a cloned voice speaking through a
talking avatar. The words should match the lips." --voice
inputs/ljspeech_reference.wav --face demo --clone-engine openvoice-v2 --engine
wav2lip --job-id gate2-openvoice-wav2lip` → exit 0:
- voice: `openvoice-v2 (base: kokoro)`, 5.97 s, alignment `mms_fa`, 17.0 s
- video: `/outputs/renders/gate2-openvoice-wav2lip.mp4`, wav2lip, 10.1 s
- **lip sync: offset 0 frames, LSE-C 10.81, LSE-D 5.64** (syncnet-v2, 25 fps)
- **voice similarity: 27.87%** (ECAPA vs the full reference) — not met

Gate 2 ("audio from a cloned voice + a single photo → an accurately lip-synced
clip") is **demonstrated end to end** but not signed off: clone similarity is
far below target (XTTS-v2 pending M-03), and nobody has watched the video yet
(M-04). Not marked passed.

### 2026-10-08 — T2.5 clone flow in the UI

`App.jsx`: a "Cloning engine" selector in clone mode sends `cloneEngine`; it
preselects a cloner whose weights `/health` reports present and disables one
without weights. Stale hints fixed: dialogue is Bark, High Quality says it
cannot run here, the router badge follows the chosen cloner. `/health`
`capabilities.models` no longer lists Higgs/Dia (it advertised models that
503) and gains `cloneEngines`.

Verified, real servers and weights, CPU:
- `scripts/check.sh` → ruff clean, pyrefly 0 errors, 535 tests OK; `npm run lint` exit 0.
- `npx playwright test clone.spec.js` → **passed (17.2 s)**: clone mode, OpenVoice
  V2, `ljspeech_reference`, Model Used badge `openvoice-v2`, alignment measured,
  render (blendshape) → COMPLETED, `<video>` readyState ≥ 1, duration > 1 s,
  width > 0, response is an MP4 (`ftyp`).
- The test can fail: with the `cloneEngine` line commented out the API default
  (XTTS-v2, no weights) is used and the run fails at the badge
  ("Expected substring: openvoice-v2 / Received: Model Used—"); restored → passes.

Not done: XTTS-v2 through the UI (M-03); someone watching the video (M-04).

### 2026-10-08 — T3.1 background replacement

Optional `background` on the frozen contract (D-36), `render_engine.apply_background`
(selfie segmenter, composited once on the source photo), `shared_segmenter()`,
URL resolver generalised so a background image obeys the same outputs/inputs
rules as audio, `--background` on `render_avatar.py`, a colour picker in the
avatar panel, `crossOrigin` on the video so the page can read its pixels.

Verified:
- `scripts/check.sh` → ruff clean, pyrefly 0 errors, 547 tests OK (12 new).
- planted: preflight's segmenter check removed → `test_missing_segmenter_refuses…` fails; restored.
- live, real segmenter + real speech, CPU: `render_avatar.py --face demo
  --background "#0b3d91"` → 512×512, 61 frames, 2.6 s, `background = color #0b3d91`,
  no warnings; `--background inputs/bg_test_gradient.png` (640×360, cover-fit) →
  `image bg_test_gradient.png`, 0.8 s. A frame from each, looked at: clean
  hair/ear/shoulder edge, no halo of the old background, subject unchanged.
- E2E `background.spec.js` → passed (14.2 s): UI toggle + colour → render →
  both top corners of the decoded `<video>` within 60 (sum of channel diffs) of #00c800.
  Planted: spread removed from the payload → fails ("background: color #00c800" not found); restored.

Not measured: segmenter quality on photos with busy backgrounds or hair against
a similar colour — only the synthetic-looking `demo` portrait was tried.

### 2026-10-08 — T3.2 avatar generation API

`POST /api/v1/avatar/generate` (+ `/options`, `/{taskId}`), `generation_jobs.py`
(one worker thread), `avatar_generator.build_prompt` and
`generate_registered_avatar` (the CLI now shares it), a "Generate a synthetic
face" form in the avatar panel (D-37).

Verified:
- `scripts/check.sh` → ruff clean, pyrefly 0 errors, 556 tests OK (9 new).
- planted: the id-clash check removed → `test_existing_id_is_refused_before_any_generation` fails; restored.
- live, real Stable Diffusion 1.5 on CPU, real server: POST `{middle-aged, woman,
  long-dark, seed 3, attempts 3, steps 20}` → 202; COMPLETED after **141 s**, seed 3
  passed the quality gate first time, provenance `synthetic` with prompt, seed and
  steps recorded; `demo` → 409, a `prompt` field → 422. Rendered with the
  blendshape engine (60 frames, 2.9 s, no warnings); frame looked at: a clean
  frontal portrait. **The prompt asked for long hair and got short hair** —
  attribute control is approximate (SD 1.5), so the UI says "choices", not guarantees.
- E2E `generate.spec.js` → passed (6.7 s): options come from the API, taken id shows the 409 text.
  The full UI-triggered SD run is not in E2E (minutes on CPU); it was verified by hand above.

Not done: the generated face `gen-live` is left in the local `inputs/faces/`
(gitignored); delete with `scripts/make_avatar.py --delete gen-live` if unwanted.

### 2026-10-08 — T3.5 temporal jitter metric

`backend/jitter_metric.py` (D-38) and `render_avatar.py --jitter`.

Verified:
- `scripts/check.sh` → ruff clean, pyrefly 0 errors, 564 tests OK (8 new in `test_jitter_metric.py`). pyrefly first caught a real typing slip (tuple assigned to an array-typed variable), fixed.
- live, real MediaPipe on a real render: `demo`, blendshape, 3 s → mean **0.278%**,
  p95 0.758%, max 0.963% — **N-09 met** (< 2%), as an upper bound.
- negative control through the real detector: the still `demo` photo → 0.0%;
  the same photo shifted 5 px on alternate frames → **3.59%**, target missed. So the
  metric does respond to shake.

Not measured: Wav2Lip and the generated face (`gen-live`) — only the blendshape `demo`
render. The Wav2Lip result matters more (it repaints the mouth region); run
`render_avatar.py --engine wav2lip --jitter` to add it.

Download check (asked 8 Oct): XTTS-v2 is still incomplete — `model.pth` 1.3 GB,
no `config.json`, no writes since 30 Sep. It needs the owner's CPML acceptance (M-03).

### 2026-10-08 — T6.2 request ids + structured logs, T7.5 placeholder sweep

`backend/request_context.py` (D-39); `app.py` registers the middleware last
(outermost) and exposes the header to browsers via CORS; both job runners
(`job_queue.py`, `generation_jobs.py`) carry the context into their worker thread.

Verified:
- `scripts/check.sh` → ruff clean, pyrefly 0 errors, 572 tests OK (8 new).
- **the tests found a real gap:** an unhandled exception returned a 500 with no
  id (Starlette builds that response outside the middleware). Fixed by catching
  in the middleware; the test now asserts the id is in the body and the header.
- live, real server: `GET /health` with `X-Request-ID: my-trace-1` → echoed;
  none → fresh `692aaae9efee49e2`; `bad id` (space) → replaced. Synthesis +
  `POST /avatar/render-job` with `X-Request-ID: trace-render-7` → response header
  `trace-render-7`; the server log shows the render *worker thread's* lines
  `[trace-render-7] face_engine: Loaded MediaPipe FaceLandmarker…` and
  `[trace-render-7] render_engine: Rendered t62-trace… 64 frames in 2.64s`.
- T7.5: `git grep -nE "TODO|FIXME|XXX|placeholder"` outside docs → 2 hits left, the
  HTML attribute (D-40). Owner may overrule.

Not done: uvicorn's own access-log lines are not stamped (D-39). Side effect worth
knowing: importing the app now installs a log handler on the root logger when none
exists, so test output shows app log lines.

### 2026-10-08 — T7.3 dependency audit, and two bugs it uncovered

`pip-audit` found 10 vulnerable packages (about 120 advisories). `npm audit --omit=dev` → 0.
Upgraded what the pins allow (D-41); 4 packages remain as documented exceptions,
each about loading untrusted model repos, which this code never does (grep-checked).
After the upgrade: ruff clean, pyrefly 0 errors, 574 tests OK.

Then the E2E suite (the live check for new Starlette/Pillow) failed one spec in 9 of 10 runs,
and chasing it found two real problems, neither caused by the upgrade:
1. **Event loop blocked.** `analyze_face` and photo registration are `async def` but ran
   landmark detection directly on the loop; measured on a real server, a trivial
   `GET /languages/ace` took **2.2 s** whenever an analysis was running (it should take ms),
   and on a cold model the stall was long enough to time out specs. Both now use
   `run_in_threadpool`. New `tests/test_event_loop.py` (shared-loop, so the blocking is
   visible); **my first version could not fail** (the plant passed) because its clock started
   after the block, so I rebuilt it around an engine-entered signal; now both plants
   (analysis and registration back on the loop) fail it, and the real code passes.
2. **Rate limiter vs. the suite.** `smoke.spec.js` "clone mode loads references" failed only in
   the full run: with `RATE_LIMIT_RPM=100000` all 10 pass (D-42). Playwright's backend now
   sets it.

Verified: `npx playwright test` → **10 passed** three times in a row (37 s, 37 s, 35 s)
after being 9+1 failed three runs in a row before the fix.

Not done: `diffusers`, `transformers`, `accelerate`, `nltk` advisories (D-41).

### 2026-10-08 — T7.1 Python SDK

`sdk/avatar_platform/` (installable: `sdk/pyproject.toml`, httpx only): `AvatarClient`
with `synthesize`, `render`, `score_lipsync`, `voice_similarity`, `faces`, `generate_face`,
`health`; `ApiError` (status, server detail, request id), `JobFailed`. Polls hide the
async jobs; a 429 is retried after `Retry-After`, three times, then raised. ruff and pyrefly
now cover `sdk/`. Log-handler install moved from import to server start so importing the
app in tests prints nothing.

Verified:
- `scripts/check.sh` → ruff clean, pyrefly 0 errors, 584 tests OK (10 new).
- planted: the SDK's `targetFps` renamed → the contract test fails (the job it builds is
  validated by the real `AvatarRenderJob`); restored.
- live, real server, CPU: `AvatarClient` → health ok, faces `[demo, gen-live]`, synthesise
  (kokoro, 2.5 s, 22 phonemes, `mms_fa`), render `demo` with `background={"color": "#0b3d91"}`
  (blendshape, 63 frames, result reports the background), score: offset **-4** frames,
  LSE-C 2.743, LSE-D 12.073.

**Observation to chase in T6.6, not explained:** that blendshape lip-sync offset is -4, but
T2.3 measured offset 0 on 3/3 clips. Differences: a different (short) sentence and a blue
background. SyncNet crops the face from the first frame, so the background or clip length
may matter. No claim is made either way until it is re-measured.

### 2026-10-08 — T6.1 job persistence

`backend/job_store.py` (SQLite, WAL), used by `InMemoryJobQueue` (render jobs) and
`GenerationJobs` (D-43). `check.sh` runs the unit tests with `JOBS_DB=:memory:`.

Verified:
- `scripts/check.sh` → ruff clean, pyrefly 0 errors, 593 tests OK (9 new).
- planted: the save inside `update()` removed → 3 persistence tests fail; restored.
- live, two real server processes: A (pid 221130) rendered `t61-persist` → COMPLETED;
  A stopped (health unreachable, GET → HTTP 000); B (pid 226048) started → `GET
  /render-job/t61-persist` → COMPLETED, `/outputs/renders/t61-persist.mp4`, blendshape,
  61 frames; re-submitting the same jobId → **409 "jobId already exists"**.
  (My first attempt at this was invalid: the kill missed, so server B never bound the
  port and server A answered. Caught by the "STILL UP" line and redone with the server
  found by its listening port.)

Not done: synthesis task records are not persisted (D-43); a render interrupted by a restart
is failed, not resumed.

### 2026-10-08 — T2.1 XTTS-v2 live clone + similarity (M-03 resolved)

The owner ran `fetch_models.py --only xtts-v2` and answered the licence prompt themselves.
Weights verified against Hugging Face: `model.pth` 1,867,929,118 bytes and
`speakers_xtts.pth` sha256 match the published values; `doctor.py` → `PASS xtts-v2`,
36 pass / 4 warn / 0 fail. (`hash.md5` is Coqui's own folder marker, not the file's md5.)

New `scripts/measure_clone_similarity.py`: splits the admissible reference (first 30 s as the
cloner's sample, last 20 s held out), clones 6 sentences, scores each by ECAPA-TDNN against
the held-out audio, reports the real-speech ceiling, writes JSON. It refuses a synthetic
reference (checked: exit 1 with the reason); its working copies are removed afterwards.

Real XTTS-v2, CPU, `ljspeech_reference.wav` (LJSpeech, open licence):

| sentence | similarity |
|---|---|
| quick brown fox | 67.6% |
| sells sea shells | 63.2% |
| quarterly revenue | 55.1% |
| confirm shipment | 57.5% |
| thick fog | 62.0% |
| committee review | 66.9% |

**Mean 62.0%, sd 4.6.** Ceiling (same speaker's real speech) **91.7%**. OpenVoice V2 under the
identical method: 34.2%. **N-02 is not met** at 85% or 90%; XTTS-v2 gets about 68% of the way
to the ceiling. Latency: ~30 s per sentence on this CPU including the cold load (not a GPU figure).
Consequences: R-01 is now met (5 engines, each synthesised live) and R-03 passes.

Not measured: XTTS-v2 on a second speaker (only one admissible reference exists, Q-05), and
anything on the GPU. Higher similarity might come from a longer or cleaner prompt or from the
`gpt_cond_len` / `temperature` settings; not tried, so no claim.
