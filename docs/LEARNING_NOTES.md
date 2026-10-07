# Learning Notes

Study material built up one task at a time. Append after each task; never
rewrite history here. Newest entry at the top.

Format for each entry:

```
## <task ID> - <title> (<date>)

**Goal:** what the task was for, in one line.

**Concepts met for the first time**
| Concept | What it is | Why here | Alternative |
|---|---|---|---|

**How to say it in an interview:** one or two sentences.

**What I got wrong / what surprised me:** the useful part.

**Verified by:** the command run and the real output.
```

---

## Seed entry: the two bugs worth remembering (2026-10-07)

Not a task, but the first two lessons this repo produced. Both are the same
shape: something that looked like success.

**1. A fallback that cannot be seen is a lie.**
`ForcedAligner` used to fall back to spreading phonemes evenly across the text
when MMS_FA could not run. The output had the same type, the same shape and the
same field names as a real alignment, so a *guessed* mouth timeline was
indistinguishable from a *measured* one. Lip sync quietly got worse and nothing
reported why. Fix: record `last_method` (`mms_fa` vs `acoustic-fallback`), warn
loudly, and carry it out through the API as `alignmentMethod`.

> Interview line: "Degrade loudly. If a fallback returns the same shape as the
> real path, the caller cannot tell it was cheated, so the fallback must label
> itself."

**2. Environment variables can outrank your code.**
`celery_app.py` built its app with `broker="memory://"` so that
`QUEUE_BACKEND=in_memory` would need no Redis. It got Redis anyway. Celery reads
`CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` from the environment and those beat
*both* the constructor arguments *and* a later `conf` assignment. `.env` set them
for the celery backend and `load_dotenv()` put them in `os.environ`. Every eager
task then wrote its result to a Redis that in_memory mode promises you do not
need; it stayed hidden only because `start-docker.sh` had Redis running.

> Interview line: "Know your config precedence. A library that reads the
> environment itself can override the arguments you passed it, so 'I set it in
> code' is not proof."

**Verified by:** `synthesize_audio.delay(...)` reaching SUCCESS and reading its
result back with Redis stopped — `CacheBackend`, Kokoro on CPU, 14.4 s cold.
