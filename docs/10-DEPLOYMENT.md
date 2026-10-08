# 10 — Deployment and runbook

## Local (development)

```bash
./backend/.conda/bin/python -m pip install -r backend/requirements.txt -r backend/requirements-dev.txt
# Installed without their own pins, which would break numpy<2 (see requirements.txt):
./backend/.conda/bin/python -m pip install --no-deps TTS==0.22.0
./backend/.conda/bin/python -m pip install --no-deps "git+https://github.com/myshell-ai/OpenVoice.git@74a1d147b17a8c3092dd5430504bd83ef6c7eb23"
# The watermarks (also --no-deps; the API refuses to produce unmarked media without them):
./backend/.conda/bin/python -m pip install --no-deps audioseal==0.2.0 omegaconf antlr4-python3-runtime==4.9.3 videoseal==1.0.1
./backend/.conda/bin/python -m pip install av lpips pytorch_msssim calflops decord pycocotools PyWavelets timm==0.9.16 "scikit-image<0.22" "networkx<3"
# Kokoro's English front end needs this spaCy model (it tries to pip-install it on first use otherwise):
./backend/.conda/bin/python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
cd backend && PYTHONPATH=. ./.conda/bin/python -m uvicorn app:app --port 8000
cd frontend && npm ci && npm run dev -- --port 5173
```

`QUEUE_BACKEND=in_memory` needs no Redis. Weights: `scripts/fetch_models.py`
and `scripts/fetch_vision_models.py` (`--dry-run` to see status).

## Docker (T7.2)

`docker compose up --build -d` builds the API image and serves the built
frontend; `curl -fsS localhost:8000/health` must return 200 with
`"status":"ok"`. Model weights are mounted from the host, never baked in.

## Runbook

| Symptom | First check |
|---|---|
| anything | `PYTHONPATH=backend ./backend/.conda/bin/python scripts/doctor.py` |
| known symptoms | `docs/DEBUGGING.md` (symptom → cause → fix tables) |
| GPU not used | `torch.cuda.is_available()`; the VS Code sandbox hides the GPU |
| port busy | `ss -ltnp 'sport = :8000'` and stop the old server before testing |

## Local git hooks (recreate in a fresh clone, see D-13)

```bash
# .git/hooks/pre-commit
#!/usr/bin/env bash
# Local only (lives in .git/hooks, never pushed). On main (production) blocks
# committing working notes or assistant wording. On m1 those files belong.
set -euo pipefail
[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] || exit 0
staged=$(git diff --cached --name-only --diff-filter=ACMR)
bad=$(printf '%s\n' "$staged" | grep -E '(^docs/|^\.claude/|\.md$)' | grep -vx 'README.md' || true)
if [ -n "$bad" ]; then echo "pre-commit: not for main:"; echo "$bad"; exit 1; fi
pat='claude|anthropic|co-authored-by|generated with|copilot|chatgpt'
if git diff --cached -U0 --diff-filter=ACMR | grep -E '^\+' | grep -v '^+++' | grep -niE "$pat"; then
  echo "pre-commit: assistant wording in staged changes (above)"; exit 1
fi

# .git/hooks/commit-msg
#!/usr/bin/env bash
# Local only. On every branch: no co-author trailers, no "generated with", no
# conventional-commit prefixes. On main also no assistant names at all.
set -euo pipefail
msg=$(grep -v '^#' "$1")
if printf '%s' "$msg" | grep -niE 'co-authored-by|generated with'; then
  echo "commit-msg: attribution line (above)"; exit 1
fi
if printf '%s' "$msg" | head -1 | grep -qE '^[a-z]+(\([^)]*\))?!?: '; then
  echo "commit-msg: no 'type:' / 'type(scope):' prefixes"; exit 1
fi
if [ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] && printf '%s' "$msg" | grep -niE 'claude|anthropic|copilot|chatgpt'; then
  echo "commit-msg: assistant name in a production commit (above)"; exit 1
fi
```

Then `chmod +x .git/hooks/pre-commit .git/hooks/commit-msg`.
