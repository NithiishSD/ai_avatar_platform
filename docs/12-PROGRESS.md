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
| T2.1 | Done — measured, **N-02 not met** (62.0% vs 85%) | de4a36a |
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
| T3.3 | Done — voice carries over only partly (43–49%) | 3b43262 |
| T3.4 | Done | 0b43bca |
| **M3 gate** | **Passed 8 Oct** (= roadmap Gate 3): `check.sh` 620 tests green, `npx playwright test` 11 passed, `npm run build` clean. Targets inside it: N-02 not met (62.0% English, 43–49% cross-lingual), N-09 met | |
| T6.3 | Done | (this commit) |
| T6.4 | Done | (this commit) |
| T6.5 | Done — 6 findings fixed (09-SECURITY.md); S-16 and S-18 open | (this commit) |
| T4.1 | Done | 1c88e64 |
| T4.2 | Done | 1c88e64 |
| T4.4 | Done (measured with T4.1/T4.2) | 1c88e64 |
| T4.3 | Done | (this commit) |
| **M4 gate** | **Passed 8 Oct** (= roadmap Gate 4): `check.sh` 671 tests green, `npx playwright test` 13 passed, `npm run build` clean; N-07 met as defined (first audio → first frame 1.4–1.8 ms), text → first audio ~0.43 s warm | |
| T5.2 | **Built and unit-tested; live robustness sweep and a person's look (M-07) still to do** | (this commit) |
| T5.4 | Done | 43c9c87 |
| T5.3 | Done | (this commit) |
| T5.1 | Measured, **waiting on M-06** (a person listens: is it inaudible?) | 567f07d + this commit |

**Remaining order (owner asked for continuous building, small tasks first; the session is cleared between batches):**
M5: T5.1 audio watermark -> T5.2 video watermark + signed manifest -> T5.3 verify endpoint -> T5.4 audit trail ->
T6.6 re-run benchmarks (incl. the unexplained blendshape offset -4, see the T7.1 entry) ->
T7.2 Docker (build is heavy: torch image) -> T7.4 documentation pass -> T7.6 README -> T7.7 final verification.
Waiting on the owner, cannot be done here: T2.3 sign-off and the Gate 2 sign-off (M-04: host GPU peak VRAM + a look at the video; the owner's host run failed on a wrong command, correct one is in 13). M-03 is done. Q-04: the problem-statement PDF is unrecoverable, the owner is to re-export or paste it.
Environment gotchas learned: the machine is shared (another project's Java services use ~4 GiB) and has 14 GiB, so a full `npx playwright test` needs the RAM guard (D-44); run long jobs detached (`setsid nohup … &`) and poll, because an OOM kill of the foreground shell is exit 137; never `pkill -f` / `pgrep -f` with a pattern that also appears in your own
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

### 2026-10-08 — T3.3 cross-lingual cloning (XTTS-v2)

`measure_clone_similarity.py` gained `--language` with six test sentences each in Spanish,
French and Hindi (English reference, so the cloner has never heard the speaker in these
languages). Real XTTS-v2, CPU, held-out scoring as in T2.1:

| language | mean | sd | min | max |
|---|---|---|---|---|
| English (T2.1) | 62.0% | 4.6 | 55.1 | 67.6 |
| Spanish | 43.9% | 5.4 | 33.3 | 51.1 |
| Hindi | 48.8% | 1.9 | 45.8 | 52.0 |
| French | 42.8% | 6.0 | 32.9 | 50.9 |

Real-speech ceiling 91.7%. Speaking a new language costs about 13 to 19 points. Through the
real API (`mode=clone`, `cloneEngine=xtts-v2`, via the SDK): Spanish 3.34 s, 43 phonemes,
alignment `mms_fa` (measured), ECAPA 44.42% against the full reference; Hindi 4.36 s, 39
phonemes, `mms_fa`, 49.52%. These agree with the script's numbers, and the 30 s prompt being
part of the full reference did not inflate them. OpenVoice V2 Hindi (T2.6b) is the other data
point. **Not met** against 85%/90%, like English. No cross-lingual figure for a second speaker
(Q-05).

### 2026-10-08 — T3.4 Gate 3 flow, and the memory problems it exposed

`frontend/e2e/gate3.spec.js`: generated avatar (`gate3-face`, created through the API the first
time, reused after) → clone mode, XTTS-v2, `ljspeech_reference` → language **Spanish** picked
in the UI → emotion **joy** → Generate → badge `xtts-v2`, "Emotion Applied", alignment measured
→ background colour → Render → COMPLETED → the decoded video's top corners are the new colour.
**Passed in the browser against real servers and weights**, and the same file proves it can fail:

**Bug 1 (user-visible, now fixed):** the studio sends ISO-639-3 codes (`spa`, `hin`, `fra`) but
XTTS-v2 takes `es`, `hi`, `fr`. Picking Spanish in clone mode crashed XTTS ("Language spa is not
supported") and `xttsSupported` called Spanish unsupported. New `language_registry.xtts_code()`
maps either spelling (and Coqui's `zh-cn`); a language XTTS cannot speak is now a 400 naming
`openvoice-v2`. With the old code restored, `gate3.spec.js` fails with exactly that error.

**Bug 2 (test, not product):** my first corner-colour read got black because the canvas was
drawn before the browser painted the frame (`217` = distance of black from `#0b3d91`). The MP4
itself was right (ffmpeg: corners (10,60,141)). `readFrame()` now seeks, waits, retries.

**Memory (D-44).** The full suite was OOM-killed twice (exit 137). Causes found by measuring:
SD 1.5 adds 6.6 GiB on CPU, XTTS-v2 4.2 GiB; RAM guard + `malloc_trim` + shared aligner added.
- one process loading XTTS → OpenVoice → Bark → XTTS: unguarded 3.0 → 3.8 → 5.2 GiB with all
  three resident; guarded one engine at a time, 1.5 GiB after OpenVoice.
- full E2E, server RSS: before the work peak **9.9–10.1 GiB**, lowest free RAM 1.3 GiB; after
  **8.2 GiB, lowest free 3.1 GiB**, and the suite went from 1.2 min to 55 s (no aligner reload).
- `scripts/check.sh` → ruff clean, pyrefly 0 errors, 620 tests OK (19 new). Plants: old language
  mapping → unit test and E2E fail; SD/Bark/OpenVoice/XTTS loaders each have a test that the
  guard is asked for the measured size and that a refusal is not cached.
- `npx playwright test` → **11 passed** (54.7 s), twice in a row at the end (and 11 passed earlier).

Not done: the machine is shared (another project's Java services held ~4 GiB during these runs);
on a quiet 14 GiB box the headroom is larger. Peak 8.2 GiB is still high for one server holding
Kokoro + the aligner + whichever cloner ran last; it is not a 6 GiB-VRAM-style hard limit, and
the GPU path (`ensure_vram`) is unchanged and untested here. `SpeechQualityAuditor` in `app.py`
is a second instance from the router's (SQUIM would load twice if both are used); not changed.

### 2026-10-08 — owner ran M-04 on the host: CPU, and a render bug it showed

The owner's host output (`render_avatar.py --engine wav2lip --metric`) printed
`Processing Device: CPU`, so the host run did not use the GPU either. Diagnosis (read-only):
`lspci` shows the RTX 4050 Max-Q; `lsmod` shows nvidia / nvidia_uvm / nvidia_drm; `/dev/nvidia0`
and `/dev/nvidiactl` exist; `/proc/driver/nvidia/version` says 595.91.07. **But** `libcuda.so`
and `nvidia-smi` do not exist (`dpkg` has only `nvidia-kernel-common-595`), so
`torch.cuda.is_available()` is False in `backend/.conda` (torch 2.5.1+cu121). `apt-cache`
offers the matching `libnvidia-compute-595` and `nvidia-utils-595` at 595.91.07. Installing
them is a system change, so the owner runs it (M-04 row has the exact commands). Nothing in the
code is wrong: the router already picks CUDA when torch sees it. **N-06, N-07, N-10 and peak VRAM
stay unmeasured until then.**

The same output carried `Warning: 82 frames were rendered but the file holds 81`. It was not new
(my earlier runs had it too; I had hidden it with `grep -v Warning`). ffmpeg's `-t <audio length>`
dropped a last frame that started 10 ms before the cutoff (3.25 s x 25 fps = 81.25, ceil 82), so
the file was a frame shorter than the audio. The writer's cutoff is now the end of the last
frame. New regression test (3.25 s at 25 fps; fails with the old line, passes with the fix).
Live, same command on CPU: 82 frames, video 3.28 s, audio 3.25 s (uncut), no warning, offset 0,
LSE-C 9.301, LSE-D 6.545; **jitter on the Wav2Lip render: mean 0.354%, p95 0.834%, max 1.093%**
(< 2%, closes the "Wav2Lip not measured" gap in T3.5). `check.sh`: 621 tests OK.

### 2026-10-08 — T6.3 load test against a running server

`scripts/load_test.py` (stdlib, keep-alive connections, writes `outputs/benchmarks/load-test-*.json`).
Real uvicorn on this CPU box, queue `in_memory`. Run A, default limiter (120 rpm, burst 30):
- render-job initiation, 20 jobs: all **202**, mean **4.5 ms**, p50 3.1, p95 6.6, max 20.8 ms.
- synthesis POST: 7,211 ms first (cold), 287 and 296 ms warm. In `in_memory` mode the
  synthesis task runs inside the request, so this is the whole synthesis, not an acknowledgement.
- capacity, 8 clients, 30 s: **138 requests/min accepted**, 79,785 refused 429 (the limiter
  answers a refusal quickly: 160k attempts/min), accepted p50 10.4 ms, p99 14.4 ms.
Run B, limiter lifted (`RATE_LIMIT_RPM=10000000`), 20 s each: 1 client 56,033 /min, 8 clients
48,422, 32 clients 46,542; latency p50/p95/p99 at 32 clients 36/105/116 ms.

Reading it honestly: N-03 is met for render jobs and **not** for synthesis in this queue mode
(N-03 for synthesis needs Celery + Redis, which was not run). N-04 is met by policy (138/min)
but the default of 120 is barely above the target of 100, and a UI session rendering continuously
polls ~60/min (D-42). Raw capacity of a trivial read is ~50k/min from one process, which says
nothing about heavy endpoints. Not measured: Celery mode, multiple uvicorn workers, auth enabled.

### 2026-10-08 — T6.4 concurrency test (N-08)

`tests/test_queue_concurrency.py` (4 tests, thread-switch interval forced to 1 µs so a missing
lock shows up) and `scripts/concurrency_test.py` (live). **Proved able to fail:** with the
`enqueue` lock replaced by a bare block, `test_the_same_id_submitted_by_many_threads_is_accepted_exactly_once`
fails ("the id was accepted more than once"); restored → 4 pass.
Live, real server (limiter lifted), 60 distinct jobs from 60 threads released together plus 12
repeats of used ids: 202 ×60, 409 ×12, all 60 COMPLETED, none failed or lost, 60 distinct MP4s;
all submitted in 0.22 s, drained in 18.8 s (191 jobs/min on ~1.2 s clips; renders run one at a
time by design). **N-08 met on the in-process queue.** Not run: Celery + Redis (no Redis here),
so the Redis `SET NX` uniqueness path is covered by the existing fake-Redis unit tests only.

### 2026-10-08 — T6.5 security review

Every control in `09-SECURITY.md` was checked against the code and, where it could be attacked,
probed against the real app (traversal, SSRF, oracle, upload, CORS, auth ordering, limiter).
Six findings, all reproduced first and fixed: F1 audio paths could name any project file; F2
**the rate limiter was bypassable with a rotating `X-API-Key` when auth is off (default): 200 of
200 allowed, measured; now 30 of 200 live, same as a client with no key**; F3 absolute paths in
500 bodies; F4 `speakerWav` existence oracle; F5 missing API-level tests; F6 dev Redis/Postgres on
all interfaces. Details and verification in `09-SECURITY.md`; D-45.

Verified: `check.sh` ruff clean, pyrefly 0 errors, **643 tests OK** (18 new: `test_security_probes.py`,
`RateLimitIdentityTests`). Plants, each fails the new tests and is restored: project root allowed
again (2 fail), `speakerWav` confinement removed (1 fail), `identity` using the client key again (1 fail).
Live: real server, default limiter, 200 requests with a new random key each → 30×200 + 170×429;
200 with no key → the same. `docker compose config` shows `host_ip: 127.0.0.1` for both ports.
Existing tests that used temp-folder voices now patch `app.inputs_dir` (the old behaviour was the bug).

Not done: S-16 (Dockerfile does not exist, T7.2), S-18 (M5); the limiter is per process.

### 2026-10-08 — T4.1 streaming TTS, T4.2 live frames, T4.4 latency (M4)

`backend/live_engine.py` (sentence splitter, binary framing, `LiveSession`, pipelined `stream_text`),
`WS /api/v1/live` in `app.py`, message contracts in `contracts.py`, `scripts/live_client.py`,
a lock around `VoiceEngineRouter.synthesize` (D-47). Design and protocol: D-46 and `05-API.md`.

Verified:
- `scripts/check.sh` → ruff clean, pyrefly 0 errors, **671 tests OK** (28 new: `test_live.py` 26, router lock 2).
- the tests were distrusted because they passed first time. Four plants, each caught: prefetch moved
  after the frames (**my first version of that test could not fail**: it compared against the last frame of
  *all* sentences; fixed to the first sentence's frames, then caught), session slot never released (5
  tests fail), `cleanup` removed (2), auth check skipped (1).
- live, real server, real Kokoro + MMS_FA aligner + animator, CPU, `demo`, 384 px, 25 fps, "Hello there. This is
  a live avatar speaking to you. The frames follow the words as they are spoken." (3 sentences, 7.2 s):

| | cold (models loading) | warm run 1 | warm run 2 |
|---|---|---|---|
| session open | 1,406 ms | 26 ms | n/a |
| text → first audio | 7,252 ms | 425 ms | 461 ms |
| first audio → first frame (N-07) | 6.0 ms | 1.4 ms | 1.8 ms |
| text → first frame | 7,258 ms | 426 ms | n/a |
| frames / audio received | 182 / 7.2 s | 182 / 7.2 s | 182 / 7.2 s |
| late frames (real-time client) | 2 (max 17.7 ms) | 2 (max 13.3 ms) | 2 |
| per-sentence synth+align | 7,248 / 895 / 996 ms | 421 / 799 / 862 ms | n/a |
| render per frame | 2.3–4.1 ms | 2.5–3.4 ms | n/a |

  All 182 frames decode as 384×384 JPEGs (~20.7 kB each), presentation times monotonic, alignment
  `mms_fa` (measured) for every sentence, audio plays as 24 kHz PCM16 (7.2 s written to a WAV).
  The face moves: 123 of 181 consecutive frame pairs differ (58 identical = silent holds).

Reading it honestly: N-07 is met as the roadmap defines it, but only because a blendshape frame is cheap;
the figure a person feels is text → first audio, about 0.43 s warm and 7.3 s on the very first request after
start. Not measured: Wav2Lip live (not offered), clone mode live (XTTS is ~10 s a sentence on this CPU, so
it will not feel live), several sessions at once, GPU, a real browser playing it (T4.3).

### 2026-10-08 — T4.3 live UI + E2E, and the M4 gate

`frontend/src/LivePanel.jsx` (wired into `App.jsx`): Start / Speak / Interrupt / Stop, a canvas, and counters.
Playback is by the server's `presentationMs`, not arrival: PCM16 is decoded to Web Audio buffers scheduled on
the audio clock (re-anchored if the user waits between utterances), JPEG frames are decoded with
`createImageBitmap` and drawn when the same clock reaches them. `e2e/live.spec.js`, 2 tests.

Verified:
- `npx playwright test live.spec.js` → **2 passed (22.1 s)**, real WebSocket, real Kokoro + aligner + animator: ready
  panel reads `384×384 @ 25 fps · kokoro`; 2 sentences → exactly 2 audio chunks and 60+ frames; the canvas shows a
  face (over 30% of pixels lit) whose lower half changed more than 3 times during speech; a second utterance on the
  same session finishes; with an 8-sentence text, Interrupt returns the panel to `ready`, **no frame arrives
  afterwards** (count identical 2.5 s later) and the speech was cut short (< 400 of ~480 frames); Stop → `closed`.
- both plants caught: frames never drawn → fails ("Expected > 3, Received 1"); interrupt sends nothing → fails
  ("Expected < 400, Received 625"). Restored.
- **M4 gate:** `scripts/check.sh` 671 tests OK, ruff clean, pyrefly 0 errors; `npm run lint` clean; `npm run build`
  clean; full `npx playwright test` → **13 passed (1.2 min)**; server RSS peak 7.7 GiB, lowest free RAM 3.2 GiB.

Not verified: that anything is audible. Headless Chromium runs the Web Audio graph but there is no speaker, so the
test proves audio chunks arrive and are scheduled (counted), not that sound comes out; and nobody has watched the live
canvas. Both go on the owner's manual list (M-05 below).

### 2026-10-08 — T5.1 audio watermark: built, measurement still to run (checkpoint commit)

Chosen method (D-48): **AudioSeal** (Meta, MIT code and MIT ungated weights, 94 MB), installed `--no-deps`
(`audioseal==0.2.0` plus `omegaconf`/`antlr4`, pure Python; numpy and transformers pins unchanged).
`backend/watermark_engine.py`: embeds a 16-bit platform tag = HMAC(secret key, label); a clip counts as ours when
the detector's probability >= 0.5 AND >= 14 of 16 bits equal the tag (a random message matches that well by chance
with probability 137/65536 = 0.2%). Key: `WATERMARK_KEY`, else a random key file in `outputs/` (mode 0600, logged
loudly). Wired into `VoiceEngineRouter.synthesize` after prosody and before alignment, so **every clip, including
live sessions and the audio track of every video, is marked**; the clip is read back and detected before it is
returned, and a mark that cannot be read back fails the request. `WATERMARK_ENABLED=false` is the explicit opt-out
and the result then says `applied: false` and why. Preflight refuses with the fetch command when the weights are
missing. Registry, `fetch_models.py --only audioseal`, `doctor.py` know it; the response has a `watermark` block.

Found while testing it: AudioSeal 0.2+ **no longer resamples internally** (its own warning), so the engine
resamples to 16 kHz, marks there and adds the mark back at the clip's rate; and its `torch.compile` recompiled per
clip length (first 5 s clip took **42 s**), fixed by disabling dynamo (**0.12 s** embed, 0.1 s detect, CPU).

Verified so far: `scripts/check.sh` ruff clean, pyrefly 0 errors, **697 tests OK** (26 new in `test_watermark.py`,
unit tests run with `WATERMARK_ENABLED=false` so CI needs no weights). Four plants each caught: threshold loosened to
10 bits, no 16 kHz resampling (3 tests fail), no read-back check, preflight no longer demanding the weights. First
real runs, CPU: LJSpeech human 5 s and a Kokoro clip: signal-to-mark 30.9 and 26.3 dB; marked clips detected with
16/16 bits, p=1.000; the unmarked originals 5/16 and 8/16 bits, p=0.000, not detected.

**Not done, so not claimed:** the MOS change (SQUIM before/after), robustness to AAC/MP3/Opus/noise/trim/speed, and the
false-positive rate over many clips: all in `scripts/measure_watermark.py`, written and type-checked, **never run**.
Also not done: the live-session and REST responses were not re-checked with the watermark switched on end to end, and
the full E2E suite was not re-run with it on.

### 2026-10-08 — T5.1 measured (`scripts/measure_watermark.py`, 16 clips, CPU)

Numbers in `08-TESTING.md` (R-32). In short: the mark sits **26.6 dB below the speech** (22.8 at worst); SQUIM's MOS
estimate did not fall (mean +0.06, worst -0.003; PESQ-estimate -0.07, STOI unchanged: SQUIM is a model's estimate,
and a positive MOS change is noise, not an improvement). It was detected in **16 of 16** clips after a 16-bit WAV, an
8 kHz resample, AAC at 128 *and* 64 kbps (the MP4 soundtrack case), MP3, Opus, 30 dB noise, a 0.7 s trim and a volume cut;
13 of 16 at 20 dB noise. **It is lost** after 10 dB noise, after a 4 kHz low-pass and after a 10% speed change (0 of 16 each).
The 4 kHz case is odd: the 24 -> 8 -> 24 kHz resample, which also removes everything above 4 kHz, kept it 16 of 16, so the
cause is probably the IIR filter's phase shift, not the lost band; **not isolated, no claim made**. False positives:
0 of 55 clips that should not match (human, TTS, noise, tones, silence, and our own clips with a different key).
Listening pairs for the owner are in `outputs/watermark-ab/` (M-06). T5.1 stays open until that answer is in.

### 2026-10-08 — T5.2 video watermark and signed manifest (built, tested, one real render)

**Video watermark:** `backend/video_watermark.py`, Meta's **VideoSeal 1.0** (MIT code and weights, 256-bit, 57 M parameters).
Installed `--no-deps` (+ `av lpips pytorch_msssim calflops decord pycocotools PyWavelets timm==0.9.16`, `scikit-image<0.22`,
`networkx<3`: **my first `scikit-image` install pulled networkx 3.4 and broke `gruut`, a Coqui dependency; caught by `pip`'s
own warning and reverted** — numpy, torch and transformers pins unchanged). The 228 MB checkpoint is fetched by
`scripts/fetch_vision_models.py --only videoseal` with a pinned size and SHA-256 (`3d2ff252…`), verified again on load.
Upstream gaps worked around: the wheel omits `configs/attenuation.yaml` (written next to the checkpoint), and its card paths are
repo-relative (a local card is generated). Message = 128-bit keyed tag + 128-bit manifest id; accepted when >= 96 of 128 tag
bits read back (a random video matches that well with probability ~6e-9, asserted in a test). Frames are marked in windows of
32 as they stream into the encoder (bounded memory), before the visible "AI-generated" label.
**Strength matters and was measured** (`scaling_w`): at VideoSeal's default 0.2 a textured clip lost the mark at H.264 CRF 23 and read
only 117/128 at the renderer's own CRF 20; at **0.3** it reads 128/128 at CRF 20 and 102/128 at CRF 23, the face clip holds to CRF 28
(124/128), and the mark stays ~45 dB below the picture (PSNR 44.7–46.2 dB). 0.3 is the default (`VIDEO_WATERMARK_STRENGTH`
overrides). First exploratory sweep on a real face render (strength 0.2): bit accuracy 100% uncompressed, 97% CRF 18, 96% CRF 23, 90%
CRF 28, 72% CRF 35, lost at CRF 42; an unmarked video 51.6% (chance); unmarked after H.264 54.3%.

**Signed manifest:** `backend/manifest.py`. Ed25519 (`cryptography` 50.0.2, added), key derived from the platform secret, public key in
every manifest. Lists inputs (avatar source / consent basis / licence / image hash; audio hash and its **speech record**: model,
mode, language, voice-reference consent basis and hash, emotion, audio-watermark result), processing (engine, quality,
background, label), models, both watermark results, and the video's SHA-256; **no names**. A speech record
(`<audio>.speech.json`) is written at synthesis, tied to the audio bytes; if the audio was overwritten the manifest says
"mismatched" instead of describing the wrong clip. The render writes `<video>.manifest.json`; the mark is read back from the
*encoded file* and a video that cannot show its mark is not delivered. Not C2PA (D-49).

Verified:
- `scripts/check.sh` ruff clean, pyrefly 0 errors, **738 tests OK** (new: `test_manifest.py` 15, `test_video_watermark.py` 19 incl. 4
  real-VideoSeal tests that run when the weights exist, 7 render-integration tests). **Eight plants, each caught:** `verify` always
  valid (2 fail), any issuer counts as ours (1), a manifest that leaks the speaker's name (1), tag threshold loosened to 60 (4), message without the
  manifest id (2), no read-back check in the render (1), frames never marked (1), a constant manifest id (planted; the run was killed by an out-of-memory event
  before its result, **and a `# PLANT` line was left in `manifest.py`; found and restored from backup on resume, grep confirms none remain**).
- live, real render (`render_avatar.py --face demo --metric`, CPU, all marks on): 112 frames, mark read back from the encoded MP4 with
  **128/128 tag bits**, embedded id `d54ceefd…` = read-back id = manifest id; manifest verifies (`trustworthy`), **one edited field
  breaks the signature, a different video breaks the hash match**; SyncNet offset 0, LSE-C 4.73. **Cost, not hidden: this 4.5 s video took 14.9 s to render
  (3.3x real time) because the mark runs frame by frame on the CPU; before it, blendshape rendered faster than real time.** Not measured on a GPU.
- one existing test (`GeneratorLoadTests`) failed on this memory-starved machine because my RAM guard ran in front of its missing-weights check;
  it is now hermetic (the guard has its own tests).

Not done: the robustness sweep over many clips and transforms for video (as for audio), live-session frames are **not** watermarked (each is a
JPEG pushed to a socket; stated, not claimed), a person looking for visible artefacts (M-07), the verify endpoint (T5.3), the audit trail (T5.4).

### 2026-10-08 — T5.4 consent audit trail

`backend/audit_log.py`: an append-only SQLite table (same file as the job store) in which every row stores the hash of the
row before it and its own hash (a chain), so an edited row, a deleted row or a reorder is found by `verify_chain` at the first
broken link. Events: `voice_use`, `voice_refused`, `face_use`, `face_refused`, `face_registered`, `face_generated`,
`audio_supplied` (reserved for T8.2), `manifest_issued`. Entries hold ids, hashes, the basis, the request id, the engine, never a
name (a test checks). Hooks: cloning (use, with the reference's hash and basis; refusals by file hash with the reason), every
render (`face_use` + `manifest_issued`, which is what lets a watermark's id be traced), live sessions, photo registration, face
generation, avatar refusals. `GET /api/v1/audit` and `GET /api/v1/audit/verify`.

**Stated limit:** a hash chain cannot show that the *newest* rows were deleted (nothing follows them to break); the head hash is
what to keep elsewhere, and a test pins this behaviour instead of pretending otherwise. Someone with write access to the file can
also rewrite the whole chain.

Verified: `scripts/check.sh` ruff clean, pyrefly 0 errors, 762 tests OK (24 new); four plants each caught (hash ignoring the details,
no missing-row check, refusals not recorded, no `face_use` on render). Not done: a live look at the trail after a real clone and
render through the server (the entries are asserted in unit tests; T5.3's live run will read them back).

### 2026-10-08 — T5.3 verify endpoint

`backend/authenticity.py` + `POST /api/v1/provenance/verify`. Reports four kinds of evidence separately (audio watermark, video
watermark, signed manifest, audit record) and a verdict with its meaning: `authentic_original`, `ours_modified`,
`manifest_for_another_file`, `tampered_manifest`, `foreign_manifest`, `no_evidence`. A tampered manifest outranks a present
watermark (the forgery is the finding); `no_evidence` says in words that it proves nothing about whether the content is real, and
the response always carries what the checks cannot do (heavy compression/cropping/speed changes, other systems' fakes).
Uploads stream to a temp file outside the publicly served `outputs/` with a 200 MB cap and are removed afterwards; the report
uses the upload's own filename; errors never show the temp path.

Found while testing: (1) an audio-only `.m4a` crashed the reader (`soundfile` cannot open AAC); ffmpeg decodes everything, so all
audio goes through it now. (2) the 422 message leaked the server's temp path and temp filename (ffprobe quotes the path it was given);
scrubbed and covered by a test. (3) my own test expected the exact bytes to be "modified": same bytes have the same hash, so they are
authentic; the test was wrong, the product right.

Verified: ruff clean, pyrefly 0 errors, **784 tests OK**; four plants caught (watermark outranking a tampered manifest, no-marks
reported as authentic, temp file not removed, no upload limit). **Live, real models, 10 files** (CPU, 2.5-6.5 s each):
exact render + manifest `authentic_original` (video 128/128 bits, audio 16/16); CRF 26 re-encode, a 1 s trim, a 320 px downscale
all `ours_modified` (video mark 127, 127 and 124 of 128 bits; audio 16/16); the AAC soundtrack alone `ours_modified`; an old
unmarked render `no_evidence`; someone else's video beside our manifest `manifest_for_another_file`; our video with an edited
manifest `tampered_manifest`; our speech `ours_modified` (16/16), a real human recording `no_evidence` (6/16). **Through a real
server:** render via the SDK, CRF 27 re-encode, upload: `ours_modified`, traced to job `t53-live` with 0 bit errors and "issued hash
differs: a modified copy"; exact file + manifest and a path under `outputs/` with its sidecar: `authentic_original`;
`GET /api/v1/audit` shows `manifest_issued` and `face_use`, `GET /api/v1/audit/verify` valid.

Not done / not claimed: the audio-mark false-positive and video-mark false-positive rates over many unmarked videos (audio: 0/55 earlier;
video: only the chance-level sweep and the unit-test arithmetic); detecting other systems' deepfakes (not built: R-46); a person using
the endpoint from the UI (no UI for it yet).

### 2026-10-08 — T5.3 UI: provenance line in the studio and the Verify panel

`frontend/src/ProvenancePanel.jsx` (new) posts a file, and optionally its manifest, to `POST /api/v1/provenance/verify` and shows the
verdict, its meaning, each kind of evidence on its own line and the "cannot do" caveat. `AvatarPanel.jsx` shows, under a finished
render, "invisible watermark verified (n/128 bits)" (or the reason it is not marked) and a link to the signed manifest.

Verified: `frontend/e2e/provenance.spec.js`, real Kokoro + renderer + VideoSeal + AudioSeal through the browser: the render shows
its watermark line with >= 96/128 bits and a `.mp4.manifest.json` link; the Verify panel gives `authentic_original` (exact file +
manifest), `ours_modified` (same video, no manifest; mark found), `tampered_manifest` (edited consent basis; "signature does NOT
match") and `no_evidence` for silent audio, with "does NOT show the content is real" visible. 1 passed in 38 s.
`scripts/check.sh` 784 tests OK, ruff clean, pyrefly 0 errors; `npm run lint` clean, `npm run build` clean.

Not done: a person looking at how the panel reads (add to M-07's visual pass); the full Playwright suite for the M5 gate has not been run yet.

### 2026-10-08 — M5 gate run (not passed: two manual checks open)

`scripts/check.sh all`: ruff clean, pyrefly 0 errors, 784 tests OK, frontend lint and build clean, `npx playwright test` **14 passed** (1.9 min).
An earlier run showed `live.spec.js` failing after 17 minutes against a 240 s timeout while the machine was stalled; the spec alone passed
(2 passed, 27 s) and the full rerun passed, so this was the stall, not the code. The gate is **not** declared passed: T5.1 waits on M-06
(listening) and T5.2 on M-07 (looking at the marked video; the Verify panel and the watermark line in the studio are part of that look).
