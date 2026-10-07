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
| T1.2 | Done | (this commit) |
| T1.3 | Next | |

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
