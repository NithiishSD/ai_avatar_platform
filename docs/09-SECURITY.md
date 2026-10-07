# 09 — Security

Checked item by item in T6.5 and again by the final audit. Each line: the
control, where it lives, and how it is verified.

| # | Control | Where | Verify |
|---|---|---|---|
| S-01 | API keys compared in constant time | `security.py` `verify_key` | unit test + code read |
| S-02 | Auth before rate limit (unauthenticated floods cannot drain a user's bucket) | `security.py` `inspect` | unit test |
| S-03 | Rate limit with `Retry-After` | `security.py` | unit test |
| S-04 | No secrets in code; all from environment | repo-wide | `git grep` for keys/passwords |
| S-05 | `/health` exposes no secret (key count only) | `security.describe` | unit test |
| S-06 | Upload size limit on face registration | `app.py` | test with oversize file → 413 |
| S-07 | Uploaded images decoded and re-validated, never served raw from user path | `avatar_store.py` | unit test |
| S-08 | Path traversal: ids and filenames cannot escape `inputs/` / `outputs/` | `avatar_store.py`, `render_engine.py`, `app.py` | tests with `../` |
| S-09 | Render worker reads only this server's `/outputs/` URLs or project `file://` (no SSRF) | `render_engine.py` preflight | tests with external URLs |
| S-10 | JSON-only Celery serialisation (no pickle RCE) | `celery_app.py` | config read |
| S-11 | Consent enforced before any voice or face use | `provenance.py`, `avatar_store.py`, clone path | tests |
| S-12 | CORS restricted to configured origins (dev: localhost only) | `app.py` | test |
| S-13 | Errors do not leak stack traces to clients | `app.py` handlers | test |
| S-14 | Sensitive data not logged (keys, raw uploads) | logging calls | `git grep` |
| S-15 | Dependencies audited | `pip-audit`, `npm audit` | T7.3 |
| S-16 | Container runs as non-root | `Dockerfile` | `docker run whoami` |
| S-17 | Default dev passwords only in the dev compose, overridable, never used in prod config | `docker-compose.yml`, `.env.example` | read |
| S-18 | Watermark/manifest signing key from environment | M5 | test |
