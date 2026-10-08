#!/usr/bin/env bash
# The quality gates, one command each, so "is it green?" has exactly one
# answer no matter who runs it.
#
#   scripts/check.sh            lint + types + test   (run after every change)
#   scripts/check.sh all        everything below      (run at a milestone end)
#   scripts/check.sh lint|types|test|frontend|e2e
#   scripts/check.sh hygiene [branch]   what a branch publishes (default: main)
#
# Exits non-zero as soon as any gate fails, so it can sit in CI or a hook.

# -e: stop on the first failing command. -u: an unset variable is an error,
# not an empty string. -o pipefail: a pipeline fails if any stage fails, not
# just the last one - without it `cmd | tail` would hide cmd's failure.
set -euo pipefail

# Always run from the repo root, whatever directory the caller is in, so the
# relative paths below mean the same thing every time.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The project interpreter, never whatever `python` is on PATH: base conda is
# Python 3.14 and cannot import this project's pinned dependencies.
# `${PY:-...}` keeps a value already set in the environment, which is how CI
# (no conda env, a plain virtualenv) runs this same script unchanged.
PY=${PY:-./backend/.conda/bin/python}
BIN=${BIN:-./backend/.conda/bin}

lint() {
  echo "== lint (ruff)"
  "$BIN/ruff" check backend scripts tests sdk
}

types() {
  echo "== types (pyrefly)"
  # pyrefly.toml names the local conda interpreter; pass the one in use.
  "$BIN/pyrefly" check --output-format min-text --python-interpreter-path "$PY"
}

test_() {
  # Named test_ because `test` is a shell builtin.
  echo "== unit tests"
  # unittest through discover, the only way the imports resolve: tests import
  # backend modules by bare name, so backend/ and tests/ go on PYTHONPATH.
  # JOBS_DB=:memory: keeps importing the app from touching the real job file.
  # WATERMARK_ENABLED=false: unit tests mock the speech models and must not need the watermark
  # weights (CI has none); the watermark has its own tests that mock the detector or load the real one.
  JOBS_DB=:memory: WATERMARK_ENABLED=false PYTHONPATH=backend:tests "$PY" -m unittest discover -s tests -p 'test_*.py'
}

frontend() {
  echo "== frontend lint + build"
  # A subshell `( ... )` so the cd does not leak into the gates that follow.
  (cd frontend && npm run lint && npm run build)
}

e2e() {
  echo "== e2e (playwright)"
  (cd frontend && npx playwright test)
}

hygiene() {
  # Checks a branch's committed tree, not the working copy. Development
  # happens on m1, which tracks the planning docs on purpose; production
  # (main) publishes code and a single README.md, nothing else.
  local ref="${1:-main}" bad
  echo "== repo hygiene ($ref)"
  bad=$(git ls-tree -r --name-only "$ref" | grep -E '(^docs/|\.md$)' | grep -vx 'README.md' || true)
  if [ -n "$bad" ]; then
    echo "$ref publishes files that belong on the development branch only:"
    echo "$bad"; return 1
  fi
  echo "ok"
}

case "${1:-default}" in
  default) lint; types; test_ ;;
  lint) lint ;;
  types) types ;;
  test) test_ ;;
  frontend) frontend ;;
  e2e) e2e ;;
  hygiene) hygiene "${2:-main}" ;;
  all) lint; types; test_; frontend; e2e; hygiene ;;
  *) echo "usage: $0 [lint|types|test|frontend|e2e|hygiene|all]" >&2; exit 2 ;;
esac
