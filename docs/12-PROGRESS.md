# 12 — Progress

**Resume here.** Current task is the first one not marked Done. Each entry
records what was verified live, with the real command and result.

## Status

| Task | Status | Commit |
|---|---|---|
| T0.1 | Done (on main, before the branch rule) | 1ea9d51 |
| T0.2 | Done (on main, before the branch rule) | ab1b917 |
| T0.1–T0.2 on m1 | Done | (this commit) |
| T0.3 | Next | |

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
