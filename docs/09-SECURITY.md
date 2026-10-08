# 09 — Security

Checked item by item in T6.5 and again by the final audit. Each line: the
control, where it lives, and how it is verified.

| # | Control | Where | Verify | Status (T6.5, 8 Oct) |
|---|---|---|---|---|
| S-01 | API keys compared in constant time | `security.py` `verify_key` | unit test + code read | **pass** — `hmac.compare_digest` over every key; `test_security.py` |
| S-02 | Auth before rate limit (unauthenticated floods cannot drain a user's bucket) | `security.py` `inspect` | unit test | **pass** — auth runs before the limiter; new test: 500 wrong-key requests leave a valid key's bucket untouched |
| S-03 | Rate limit with `Retry-After` | `security.py` | unit test | **pass** — 429 + `Retry-After`; seen live in the T6.3 load test (79,785 refusals answered fast) |
| S-04 | No secrets in code; all from environment | repo-wide | `git grep` for keys/passwords | **pass** — `git grep` for key/secret/password/token literals outside docs and tests: none; `.env` is gitignored and untracked |
| S-05 | `/health` exposes no secret (key count only) | `security.describe` | unit test | **pass** — `/health` shows `configuredKeys` (a count); `test_health_never_contains_a_key` |
| S-06 | Upload size limit on face registration | `app.py` | test with oversize file → 413 | **pass** — limit is 15 MB (`MAX_UPLOAD_BYTES`; this table said 10); new test: 15 MB + 1 → 413 before decoding; plus a 40-megapixel cap |
| S-07 | Uploaded images decoded and re-validated, never served raw from user path | `avatar_store.py` | unit test | **pass** — an upload is decoded and re-encoded to PNG (`Image.fromarray(...).save`), the original bytes are never stored or served |
| S-08 | Path traversal: ids and filenames cannot escape `inputs/` / `outputs/` | `avatar_store.py`, `render_engine.py`, `app.py` | tests with `../` | **fixed** — static `/outputs/..`, avatar ids, `audioPath`, `speakerWav`: see findings F1, F4; every traversal form tested |
| S-09 | Render worker reads only this server's `/outputs/` URLs or project `file://` (no SSRF) | `render_engine.py` preflight | tests with external URLs | **pass** — `audioUrl` and `background.imageUrl`: metadata IP, `file:///etc/passwd`, `s3://`, `..` all 400; new tests |
| S-10 | JSON-only Celery serialisation (no pickle RCE) | `celery_app.py` | config read | **pass** — `task_serializer`, `accept_content`, `result_serializer` all `json` (config read) |
| S-11 | Consent enforced before any voice or face use | `provenance.py`, `avatar_store.py`, clone path | tests | **pass** — consent checked before weights for faces and voices (T2.2); with F4 fixed a recording outside `inputs/` is refused before the consent check |
| S-12 | CORS restricted to configured origins (dev: localhost only) | `app.py` | test | **pass** — an `https://evil.example` origin gets no `Access-Control-Allow-Origin`; localhost gets its own origin; new test |
| S-13 | Errors do not leak stack traces to clients | `app.py` handlers | test | **fixed** — no tracebacks reach clients (the request-id middleware turns an unhandled error into a generic 500 naming the id). F3: audio endpoints echoed absolute server paths in 500 bodies; now scrubbed to `<project>` / `~`, unreadable audio is a 422 |
| S-14 | Sensitive data not logged (keys, raw uploads) | logging calls | `git grep` | **pass** — no log call formats a key, token or password; request logs carry ids, not bodies |
| S-15 | Dependencies audited | `pip-audit`, `npm audit` | T7.3 | **pass with exceptions** — T7.3, D-41 (4 packages, each about untrusted model repos, which this code never loads) |
| S-16 | Container runs as non-root | `Dockerfile` | `docker run whoami` | **open** — no `Dockerfile` exists yet; verified in T7.2 |
| S-17 | Default dev passwords only in the dev compose, overridable, never used in prod config | `docker-compose.yml`, `.env.example` | read | **fixed** — the dev compose published Redis (no password) and Postgres (default password) on every interface; now `127.0.0.1` only (F6). Default password remains dev-only and overridable by `POSTGRES_PASSWORD` |
| S-18 | Watermark/manifest signing key from environment | M5 | test | **partly**: the watermark tag key is `WATERMARK_KEY` or a 0600 key file (tested); manifest signing arrives with T5.2 |

## T6.5 review: findings

Probed against the real app (`tests/test_security_probes.py` is the probe set; each
finding below has a test that fails if the fix is removed, and was shown to).

| ID | Severity | Finding | Fix |
|---|---|---|---|
| F1 | medium | `audioPath` (align, quality-audit, voice-similarity) accepted **any file under the project root** (`.env`, source, docs) although its message said inputs/outputs; a non-audio one returned a 500 | Containment is `inputs/` and `outputs/` only. A name that exists elsewhere in the project gets the same 404 as a name that exists nowhere |
| F2 | **high for the default setup** | With auth off (the default), the rate-limit bucket was keyed on whatever `X-API-Key` the client sent. A fresh random key per request got a fresh bucket each time: **200 of 200 requests allowed** (a normal client gets 30), and memory grew by a bucket per request | The presented key names a bucket only when auth is on (i.e. verified). Measured live after: 30 of 200 for a rotating key, identical to a client with none. Buckets that have refilled are also dropped past 10,000 so rotating IPs cannot grow the store without bound |
| F3 | low | 500 bodies from the audio endpoints echoed the library's message with the absolute server path (`Error opening '/home/…/.env'`) | Messages are scrubbed (`<project>`, `~`); a file that is not audio is a 422 |
| F4 | medium | `speakerWav` was an existence oracle for the whole host: `/etc/passwd` → 403 (exists, no record), a missing path → 400 | It must be a file in `inputs/`; outside, missing and traversal all get one identical 400 |
| F5 | info | The 413 upload test, CORS, static-file and SSRF controls had no API-level tests | Added |
| F6 | medium (dev) | The dev `docker-compose.yml` published Redis (no auth) and Postgres (default password) on all interfaces | Bound to `127.0.0.1` |

Not fixed, by design or out of scope: the rate limiter is per process (the `.env.example`
note already says so; N workers multiply the limit); `TrustedHost`/TLS belong to the
deployment in front of uvicorn; S-16 and S-18 wait for T7.2 and M5.

