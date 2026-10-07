# Operating Manual

Single entry point for anyone continuing this project. Read this, then every
file in `docs/`, before changing anything.

**Branches.** Development happens on **`m1`**, which tracks this file and
`docs/`. **`main` is production**: code and `README.md` only, and nobody
pushes to it except when the owner asks for a release (see Git rules).

| Need | File |
|---|---|
| What the product is | `docs/01-OVERVIEW.md` |
| Every requirement, with an ID | `docs/02-REQUIREMENTS.md` |
| Architecture and module map | `docs/03-ARCHITECTURE.md` |
| Contracts, storage, state machines | `docs/04-DATA-MODEL.md` |
| HTTP/WebSocket API | `docs/05-API.md` |
| Frontend screens and states | `docs/06-UI.md` |
| **The task list — work it in order** | `docs/07-TASKS.md` |
| Test strategy + requirement-to-test table | `docs/08-TESTING.md` |
| Security checklist | `docs/09-SECURITY.md` |
| Docker, runbook, debugging | `docs/10-DEPLOYMENT.md` |
| Decision log | `docs/11-DECISIONS.md` |
| **Where work stopped — read first when resuming** | `docs/12-PROGRESS.md` |
| Things only the owner can answer, and manual checks for the owner | `docs/13-OPEN-QUESTIONS.md` |
| Dated history before this plan existed | `docs/context.md` |

---

## 1. Mission

An open-source AI avatar platform: script + optional voice sample + one photo
→ a talking, lip-synced avatar video, plus a live streaming avatar. Every model
is open source and runs locally.

## 2. Hard constraints

| Constraint | Value | Consequence |
|---|---|---|
| GPU (host) | RTX 4050 Laptop, 6141 MiB VRAM | One heavy model resident at a time |
| GPU (this sandbox) | **None visible** — `torch.cuda.is_available()` is False | Live verification here runs on CPU; GPU-only numbers are marked "not measured here" |
| Python | 3.10 in `backend/.conda` | Never use base conda (3.14) |
| numpy | `<2.0.0` | Coqui TTS / numba break on numpy 2 |
| transformers | `<4.48` | Dia/Higgs load through it |
| MediaPipe | 1.x Tasks API | `mp.solutions` is gone |
| Licences | XTTS-v2 (Coqui CPML) and Wav2Lip are non-commercial. Wav2Lip's checkpoint is on disk; **XTTS-v2's download is incomplete and no CPML acceptance is recorded** (no `tos_agreed.txt`) | Never set `COQUI_TOS_AGREED` or answer a licence prompt for the owner; do not fetch any licence-gated weight without asking |

## 3. Golden rules

1. **No silent fallbacks.** A missing model or failed load is logged loudly,
   reported in `/health`, and reflected in `model_used` / `alignmentMethod`.
2. **Every number records its method.** Only admissible evidence counts toward
   a threshold: MOS from SQUIM, similarity from ECAPA-TDNN against a human,
   consented reference.
3. **Consent before use.** No voice or face without a provenance sidecar.
   Never download or commit an identifiable person's photo or recording.
4. **The contract is frozen.** `AvatarRenderJob` in `backend/contracts.py` is
   the only audio→vision interface. Extend with optional fields only.
5. **Respect 6 GB.** Lazy load; one heavy model resident; release before the next.
6. **Tests never download models.** Mock loaders. Every feature still gets one
   live verification on real weights, recorded in `docs/12-PROGRESS.md`.
7. **Errors say how to fix themselves.**
8. **Say what is not done.** *Built*, *verified live* and *target met* are different.
9. **Inline explanation.** Every file touched gets teaching comments: a module
   docstring saying what it is for, each new concept explained the first time
   it appears, why-comments on non-obvious lines. Comment-only edits are
   verified by AST comparison (`scripts/check.sh` does not change code).

## 4. Workflow loop (per task)

1. Open `docs/12-PROGRESS.md`; set the task to `In progress`.
2. Read the task in `docs/07-TASKS.md` and every file it touches.
3. Implement the smallest change that meets the acceptance criteria. Comment it.
4. Unit tests with mocked models; one rejecting test per new rule.
5. Run the gates: `scripts/check.sh` (lint, typecheck, unit tests). Fix all.
6. Verify live: start the server, hit the endpoints, run real weights. Record
   the command and the real output in `docs/12-PROGRESS.md`. Anything that
   cannot be checked from here (see Manual checks) is added to the manual
   check list and the owner is told.
7. Update `docs/08-TESTING.md` traceability, `docs/11-DECISIONS.md` for any
   judgement call, `docs/13-OPEN-QUESTIONS.md` for anything only the owner knows.
8. Set the task to `Done` in `docs/12-PROGRESS.md`.
9. Commit and push (Git rules below).

**End of every milestone:** full suite + E2E (`npx playwright test`) +
production build (`npm run build`), all green, before starting the next one.

## 5. Autonomy rules

- Do not stop for routine verification or low-impact ambiguity. Pick the safe,
  reversible option, log it in `docs/11-DECISIONS.md`, keep going.
- Adding a dev tool or open-source runtime dependency is allowed; log it.
  It must respect numpy<2 and transformers<4.48.
- Never weaken a test to make it pass. A failing test is reported, then fixed
  in the code under test.
- A target that cannot be met on this hardware is still built, measured, and
  reported as **not met** with the number. It is never redefined as met.

## 6. Manual checks

Some checks need a person or the host machine, not this sandbox:

- looking at the UI or a rendered video and judging it;
- listening to audio (clone quality, watermark audibility);
- anything that needs the GPU (this sandbox has none): peak VRAM, GPU timings;
- anything on a device or account only the owner has.

For each one: add a row to the **Manual checks** table in
`docs/13-OPEN-QUESTIONS.md` with the exact command or steps and what to look
for, tell the owner in the reply, and **wait for their result** before marking
the task Done. Record their answer in `docs/12-PROGRESS.md`. Work on other,
independent tasks meanwhile.

## 7. Escalation rule

Ask the owner **only when all three hold**:

1. The action is irreversible or outward-facing beyond a normal `git push`:
   spending money, accepting a licence on the owner's behalf, using a real
   person's face or voice, deleting or force-pushing published history,
   changing deployed secrets; **and**
2. there is no safe, reversible default; **and**
3. guessing wrong would make the work useless.

Anything else: decide, log it, continue. Questions that block nothing go to
`docs/13-OPEN-QUESTIONS.md` and are reported at the end.

## 8. Git rules

- Work on **`m1`**. Commit after every task and push to `origin m1`.
- On `m1`, `CLAUDE.md` and `docs/` are tracked and committed with the work.
- **`main` is production.** Do not commit, merge or push to it unless the owner
  asks for a release. A release copies code only — never `CLAUDE.md`,
  `docs/`, `.claude/` or any `.md` other than `README.md` — and is checked
  with `scripts/check.sh hygiene main` before pushing.
- **Never force-push. Never rewrite pushed history.**
- Stage files **by explicit path**, never `git add -A` / `git add .`.
  Run `git status` before every commit.
- Commit messages: short, plain, human — what changed and why, in a sentence
  or three. **No tags, no `feat:`/`fix:` prefixes, no AI or assistant
  mentions, no `Co-Authored-By` lines.** Local hooks enforce this (D-13).

## 9. Commands

```bash
PY=./backend/.conda/bin/python
scripts/check.sh                 # lint + typecheck + unit tests (the per-task gate)
scripts/check.sh all             # + frontend lint, build, E2E (the milestone gate)
PYTHONPATH=backend $PY scripts/doctor.py
cd backend && PYTHONPATH=. ./.conda/bin/python -m uvicorn app:app --port 8000
cd frontend && npm run dev -- --port 5173
cd frontend && npx playwright test
docker compose up --build -d && curl -fsS localhost:8000/health
```

`pytest` is not installed; the suite is `unittest` run through `discover`.

## 10. Definition of Done (project)

Each item is a command with a pass condition. Completion is declared only when
every one passes, run for real.

| # | Item | Command | Pass when |
|---|---|---|---|
| 1 | Unit + integration tests | `scripts/check.sh test` | exits 0 |
| 2 | Python lint | `scripts/check.sh lint` | exits 0 |
| 3 | Python typecheck | `scripts/check.sh types` | exits 0 |
| 4 | Frontend lint | `cd frontend && npm run lint` | exits 0 |
| 5 | Frontend production build | `cd frontend && npm run build` | exits 0 |
| 6 | E2E | `cd frontend && npx playwright test` | all pass |
| 7 | Docker build + health | `docker compose up --build -d` then `curl -fsS localhost:8000/health` | HTTP 200, `"status":"ok"` |
| 8 | Dependency audit | `pip-audit` and `npm audit --omit=dev` | no unaddressed high/critical; each exception in 11-DECISIONS |
| 9 | Requirements traced | every R-/N- ID in 02 has a row in 08 | no ID without a passing test or a recorded measurement |
| 10 | Targets measured | every N- ID has a dated result + method in 08 | met or **not met** stated with the number |
| 11 | Health check honest | `PYTHONPATH=backend $PY scripts/doctor.py` | 0 FAIL |
| 12 | No placeholders | `git grep -nE "TODO|FIXME|XXX|placeholder" -- ':!*.lock'` | no hits in tracked code |
| 13 | Production hygiene | `scripts/check.sh hygiene main` | `main` tracks only `README.md` as markdown, no `docs/` |
| 14 | No assistant wording on main | `git log main` and `git grep <words> main` | no hits (on `m1` the working files are expected) |
| 15 | README complete | `README.md` | setup, run, test, deploy sections; no AI-assistant mentions |
