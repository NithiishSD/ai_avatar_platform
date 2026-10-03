# Debugging Guide

Start every investigation the same way:

```bash
PYTHONPATH=backend ./backend/.conda/bin/python scripts/doctor.py
```

It checks the interpreter, packages, GPU, disk, model weights, vision
bundles, reference provenance, `.env`, and whether the API is up. Fix every
`FAIL` before looking further; most problems below show up there.

---

## Procedure

1. **Reproduce** with the smallest input (one short sentence, one photo).
2. **Read the startup output** of the backend: the `[Model Weights]` block
   says which models really exist. A request "working" on a missing model
   means it fell back — check `modelUsed` in the response.
3. **Check `/health`** (`curl localhost:8000/health`): queue mode, security
   settings, `modelWeights`.
4. **Isolate the layer**: call the engine directly in Python before blaming
   the API, and the API directly (`/docs`) before blaming the UI.
5. **Write a failing unit test** for the bug, fix it, run the full suite.
6. **Log it** in `docs/context.md` — and if the symptom is new, add a row
   below.

---

## Known symptoms

### Environment

| Symptom | Cause | Fix |
|---|---|---|
| `No module named torch/mediapipe/fastapi` | Running base conda (Python 3.14, empty) | Use `./backend/.conda/bin/python` or `conda activate ./backend/.conda` |
| `python -m unittest tests.test_x` fails on imports | Modules resolved relative to `backend/` | `PYTHONPATH=backend:tests … -m unittest discover -s tests -p 'test_x.py'` |
| `No module named pytest` | The project uses `unittest` | Use the discover command above |
| Coqui TTS or numba crash after an install | Something pulled in numpy 2 | `pip install "numpy<2.0.0"`; never install `nari-tts` |
| IDE shows red imports that run fine | LSP using the system Python | `pyrefly.toml` / VS Code interpreter → `backend/.conda/bin/python` |

### Speech models

| Symptom | Cause | Fix |
|---|---|---|
| Clone / high-quality / dialogue request "works" but sounds like Kokoro | Router fell back; weights missing | Startup audit or `fetch_models.py --dry-run`; fetch the model |
| XTTS-v2 downloaded but audit says missing | `XDG_DATA_HOME` differs between VS Code snap terminal and normal shell | Audit lists paths searched; run backend from the same kind of shell you fetched in |
| XTTS-v2 fetch hangs at a prompt | Coqui licence prompt | A human accepts CPML; `fetch_models.py` sets `COQUI_TOS_AGREED` after the note |
| Higgs / Dia `from_pretrained` errors offline | Only `config.json` cached | `fetch_models.py --all` (≈ 6.5 GB each) |
| `CUDA out of memory` | Two heavy models resident on 6 GB | Unload the first (`del`, `torch.cuda.empty_cache()`); set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| MMS-TTS error "expects romanized input" | `uroman` not installed | `pip install uroman` |
| Clone rejects reference audio | Too short/long, silent, or corrupt | Message from `AudioValidationError` says which; 1–120 s accepted, 30–60 s recommended |

### Alignment and visemes

| Symptom | Cause | Fix |
|---|---|---|
| Hindi/Tamil visemes don't match speech | `uroman` missing → acoustic fallback (warning in log) | Install `uroman`; confirm `_prepare_for_alignment` returns Latin text |
| Timestamps evenly spaced | MMS_FA failed to load → acoustic fallback | Check log for "Using acoustic fallback"; needs torchaudio pipelines + network once |
| Render job rejected with timestamp error | Unordered, negative, or beyond `durationSeconds` | Contract validation is correct; fix the producer |

### Face and vision

| Symptom | Cause | Fix |
|---|---|---|
| `module 'mediapipe' has no attribute 'solutions'` | MediaPipe 1.x removed it | Use `face_engine.FaceMeshEngine` (Tasks API) |
| `FaceEngineUnavailable: … not found` | `.task`/`.tflite` bundle missing | Fetch into `.models/mediapipe/` (see G1-03) |
| `NoFaceDetected` | No face, face too small, occluded, or extreme angle | Use a frontal, well-lit photo; face ≥ ~20% of the image |
| Landmark count 478, docs say 468 | 468 mesh + 10 iris points | Expected; `classicMeshPoints` reports 468 |
| Head pose left/right looks inverted | Sign convention not yet validated | Task G1-07 |
| Background composite has 4-D shape error | Mask not squeezed | Fixed: `foreground_mask` returns (H, W) |

### API, queue and frontend

| Symptom | Cause | Fix |
|---|---|---|
| Job stuck in `QUEUED` forever | `QUEUE_BACKEND=celery` with no worker running, or `.env` not loaded | Use `in_memory` in development, or start a Celery worker + Redis |
| `ConnectionRefusedError` on `localhost:6379` although `QUEUE_BACKEND=in_memory` | Celery reads `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` from the environment, and they **outrank** the broker/backend passed in code. `.env` sets both to Redis, so in_memory mode stored eager results in Redis anyway. Fixed 3 Oct 2026 by clearing them in that mode | Confirm `celery.conf.result_backend` is `cache+memory://`. If you add a `CELERY_*` setting to `.env`, remember it beats the code |
| `401` from every call | `AUTH_ENABLED=true` | Send `X-API-Key`, or disable auth in `.env` for local work |
| `429 Too Many Requests` | Rate limit (default 120/min per key or IP) | Raise `RATE_LIMIT_RPM` for load tests; per process, so × workers |
| Browser CORS error | Frontend origin not allowed | Leave `CORS_ORIGINS` empty in dev (any localhost port allowed) or list the origin |
| Audio player shows nothing | Output path not under `outputs/` | Files are served from `/outputs/...` |

### Benchmarks and evidence

| Symptom | Cause | Fix |
|---|---|---|
| Similarity shows "NOT evidence" / PIPELINE_TEST | Reference is synthetic or has no consent record | Register a human reference with `make_reference.py --human` |
| API initiation latency looks huge | Eager mode runs synthesis inside the POST | Measure with `QUEUE_BACKEND=celery` and a worker |
| Problem-statement PDF extracts as empty text | File corrupted (bytes replaced by U+FFFD) | Get a fresh copy from the original source |

---

## Where to look

| What | Where |
|---|---|
| Backend log and startup audit | Terminal running uvicorn |
| Generated audio / images / video | `outputs/` |
| Benchmark reports | `docs/benchmarks/` |
| Model caches | `~/.cache/huggingface/hub`, Coqui dir (see audit), `.models/` |
| Session history of past fixes | `docs/context.md` |

### Avatar rendering and lip sync

| Symptom | Cause | Fix |
|---|---|---|
| SyncNet LSE-C suddenly lower, best offset -1, `alignmentMethod` is `acoustic-fallback` | The forced aligner could not run, so phoneme timing was estimated from the text. Seen on 30 Sep 2026 when another process (an Ollama server) held 4.6 GB of VRAM | Check `nvidia-smi`. The aligner now retries on CPU by itself; if it still falls back, the warning `FORCED ALIGNMENT FELL BACK` gives the reason |
| `InsufficientVRAM` / `CUDA out of memory` when rendering or cloning | Another process holds the GPU, or this process still holds a TTS model | Close the other process, or run on CPU: `scripts/render_avatar.py --device cpu`, or `AVATAR_DEVICE=cpu` in `.env` |
| Render job rejected: "the 'wav2lip' engine has no checkpoint" | Licence-gated weights not fetched | `scripts/fetch_vision_models.py --only wav2lip --accept-licence wav2lip`, or use `engine=blendshape` |
| Render job rejected: "audioUrl ... is not renderable here" | The worker only reads this server's own `/outputs/` URLs and `file://` paths inside the project | Pass the URL returned by `POST /api/v1/audio/synthesize` |
| 403 "no provenance record" for an avatar | An image was copied into `inputs/faces/` by hand | Register it: `scripts/make_avatar.py --human FILE --avatar-id ID --subject NAME --consent ...` |
| Mouth barely moves | Timestamps are sparse and unheld, or the clip is silent | Check `phonemeTimestamps` is non-empty and `alignmentMethod` is `mms_fa`; `RenderResult.warnings` lists unknown visemes |
| `ffmpeg was not found on PATH` | System ffmpeg missing | `sudo apt install ffmpeg` |
