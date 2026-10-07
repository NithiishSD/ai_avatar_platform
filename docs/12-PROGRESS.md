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
| T0.4 | Done | (this commit) |
| T0.5 | Next | |

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
