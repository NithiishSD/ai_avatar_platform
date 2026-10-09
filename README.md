# AI Avatar Platform

Text, an optional voice sample and one photo in; a talking, lip-synced avatar video out, plus a
live avatar that speaks while it is still being generated. Every model is open source and runs on
your own machine. Every output is invisibly watermarked and ships with a signed record of how it
was made, and no voice or face is used without a recorded consent basis.

## What it does

| Area | What you get |
|---|---|
| Speech | Kokoro (fast English), XTTS-v2 voice cloning (17 languages), OpenVoice V2 tone-colour cloning, Bark dialogue, MMS-TTS (1,077 languages); six emotion presets, speed and pitch control; forced alignment to millisecond phoneme timings |
| Avatar video | A registered photo animated by a blendshape rig (CPU), Wav2Lip (neural lip sync) or SadTalker (whole head, jaw and expression from the audio; `fetch_vision_models.py --only sadtalker`); background colour or image replacement; 512 px preview or 1080p (a small photo is enlarged once with Real-ESRGAN) |
| Your own audio | Upload a recording and a photo; without a transcript the words are recognised (Whisper) and aligned |
| Faces | Register a consented photo, generate a synthetic face (Stable Diffusion 1.5), or restyle one (realistic, cartoon, painting, sketch) with its identity similarity reported |
| Live avatar | WebSocket session: type text, or speak into the microphone, and get audio and frames back sentence by sentence |
| Provenance | Inaudible audio watermark (AudioSeal), invisible video watermark (VideoSeal), Ed25519-signed manifest per video, hash-chained consent audit trail, a verify endpoint that says whether a file came from this platform, and an opt-out list of protected voices |
| Operations | REST API with OpenAPI docs, Python SDK, batch rendering (50 jobs), metrics endpoint, request ids in every log line, API keys and rate limiting |

## Requirements

- Linux or macOS, **Python 3.10** (the speech stack needs `numpy<2` and `transformers<4.48`)
- Node.js 18+ and npm
- `ffmpeg` and `espeak-ng` on the PATH
- Docker (for the container build, optional)
- An NVIDIA GPU is optional. Everything runs on a CPU, more slowly (numbers below).

## Setup

```bash
cp .env.example .env

# Python environment inside backend/
conda create --prefix ./backend/.conda python=3.10 -y
PY=./backend/.conda/bin/python
$PY -m pip install -r backend/requirements.txt -r backend/requirements-dev.txt

# Installed without their own dependency pins, which would pull numpy 2 or old libraries:
$PY -m pip install --no-deps TTS==0.22.0
$PY -m pip install --no-deps "git+https://github.com/myshell-ai/OpenVoice.git@74a1d147b17a8c3092dd5430504bd83ef6c7eb23"
$PY -m pip install --no-deps audioseal==0.2.0 omegaconf antlr4-python3-runtime==4.9.3 videoseal==1.0.1
$PY -m pip install av lpips pytorch_msssim calflops decord pycocotools PyWavelets timm==0.9.16 "scikit-image<0.22" "networkx<3"
$PY -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Frontend
(cd frontend && npm ci)
```

### Model weights

Weights are downloaded once, never at request time. See what is present and what each costs first:

```bash
PYTHONPATH=backend $PY scripts/fetch_models.py --dry-run          # speech
PYTHONPATH=backend $PY scripts/fetch_vision_models.py --dry-run   # vision
PYTHONPATH=backend $PY scripts/fetch_models.py
PYTHONPATH=backend $PY scripts/fetch_vision_models.py
```

Some weights carry **non-commercial licences**: XTTS-v2 (Coqui Public Model Licence) and Wav2Lip
(research use) are only fetched when you name them yourself; MMS-TTS is CC-BY-NC 4.0. Without
XTTS-v2 and Wav2Lip the platform still works: Kokoro, OpenVoice and MMS speak, and the blendshape
engine animates. Stable Diffusion 1.5 (CreativeML OpenRAIL-M) has use-based restrictions.

`PYTHONPATH=backend $PY scripts/doctor.py` checks the whole installation and says how to fix
anything that fails.

## Run

```bash
# API on :8000 (OpenAPI docs at http://localhost:8000/docs)
cd backend && PYTHONPATH=. ./.conda/bin/python -m uvicorn app:app --port 8000

# Studio on :5173
cd frontend && npm run dev -- --port 5173
```

Register a face before animating it. A generated (synthetic) face needs no consent record; a photo
of a real person needs `--human FILE --subject NAME --consent <basis>`:

```bash
PYTHONPATH=backend $PY scripts/make_avatar.py --synthetic --avatar-id demo
PYTHONPATH=backend $PY scripts/render_avatar.py --text "Hello, I am a synthetic avatar." --face demo --metric
```

The default queue runs jobs inside the API process (`QUEUE_BACKEND=in_memory`, jobs kept in SQLite
across restarts). For the Celery queue, start Redis with `./start-docker.sh` and set
`QUEUE_BACKEND=celery`.

### Python SDK

```python
import sys; sys.path.insert(0, "sdk")
from avatar_platform import AvatarClient

client = AvatarClient("http://localhost:8000")
speech = client.synthesize("Hello there.", emotion="joy")
video = client.render(speech, "demo")
print(video.video_url, client.score_lipsync(video.job_id)["lseC"])
```

The SDK also covers batches, your own recordings (`voice_to_avatar`), restyling, metrics and the
protected-voice list.

## Test

```bash
scripts/check.sh            # lint (ruff), types (pyrefly), unit + integration tests
scripts/check.sh all        # plus frontend lint, production build and the browser tests
cd frontend && npx playwright test
```

Unit tests never download or load models; heavy engines are replaced by fakes. The browser tests
run the real models on a real server.

## Deploy

```bash
docker compose up --build -d
curl -fsS localhost:8000/health        # {"status": "ok", ...}
```

One container serves the API and the built studio at `http://localhost:8000`. It runs as a
non-root user. Model weights are not baked into the image: `docker-compose.yml` mounts the host's
`.models/`, Hugging Face cache, Torch cache and Coqui folder (override with `HF_CACHE`,
`TORCH_CACHE`, `TTS_DATA`), plus `inputs/` and `outputs/`. The image uses CPU PyTorch; for a GPU
build pass `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121` and run with `--gpus all`.

Before exposing it, set in `.env`:

| Variable | Why |
|---|---|
| `AUTH_ENABLED=true`, `API_KEYS=...` | API keys (the studio itself does not send keys; use the API or SDK when auth is on) |
| `WATERMARK_KEY` | keeps watermarks and manifests verifiable across rebuilds |
| `CORS_ORIGINS` | the origins allowed to call the API from a browser |
| `RATE_LIMIT_RPM`, `RATE_LIMIT_BURST` | requests per minute per key or client (per process) |

## Measured results

Measured in October 2026 on a laptop: rows marked "GPU" on its RTX 4050 Laptop GPU (6 GB) with
`scripts/gpu_benchmark.py`, the rest on its CPU. Watermarks are on unless a row says otherwise.
"Not met" means the target was measured and missed.

| Target | Result | |
|---|---|---|
| Speech quality MOS > 3.5 (> 4.0 premium) | 4.89 (SQUIM, Kokoro) | met |
| Voice cloning similarity > 85% | 62% (ECAPA-TDNN, XTTS-v2, English) | **not met** |
| Lip sync > 95% of seconds within one frame, 10+ languages | Wav2Lip 100% over 13 languages; blendshape 63% | met with Wav2Lip |
| Visual fidelity LPIPS < 0.1 | 0.045-0.080 at preview size | met |
| Identity preservation > 90% | 96-97% (SFace) | met |
| 50+ concurrent jobs, no loss | 60 jobs, none lost or duplicated | met |
| First frame within 200 ms of first audio (live) | 1.4-1.8 ms; text to first audio ~0.43 s | met |
| 60 s of video in < 30 s | GPU: 13.7 s with watermarks (CPU: 134 s) | met on GPU |
| Speech < 2 s per 30 s of audio | GPU: 0.94 s with the watermark (CPU: 13.1 s) | met on GPU |
| 30 s video render < 5 s | GPU: 4.2 s unmarked, 7.2 s with the watermark | **not met** with the watermark |
| 50+ appearance parameters | 10 verified | **not met** |
| Peak VRAM < 6 GB | 3.3 GB allocated, 4.7 GB reserved (RTX 4050 Laptop) | met |

## Limits

- Detecting deepfakes made by *other* systems is not built; the verify endpoint answers "did this
  platform make this file".
- Microphone input to the live avatar moves the mouth by loudness; it is an estimate, not
  phoneme-aligned.
- Style transfer loses identity as the style gets stronger (68% realistic, 39% sketch).
- Higgs TTS 2 and Dia cannot run on this dependency stack.

## Licence notes

The code is the project's own. Each model keeps its own licence: Kokoro Apache-2.0, MMS-TTS
CC-BY-NC 4.0 (non-commercial), XTTS-v2 CPML (non-commercial), Wav2Lip research-only, OpenVoice MIT, AudioSeal and
VideoSeal MIT, Real-ESRGAN BSD-3, SyncNet MIT, MediaPipe Apache-2.0, Stable Diffusion 1.5
CreativeML OpenRAIL-M, Whisper MIT. Check them before any commercial use.
