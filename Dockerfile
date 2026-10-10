# AI Avatar Platform: the API and the built studio in one image (T7.2).
#
#   docker compose up --build -d && curl -fsS localhost:8000/health
#
# Model weights are NOT baked in (several GB, some under licences only the owner can accept):
# docker-compose.yml mounts the host's .models/ and Hugging Face cache read-write at run time.
# PyTorch defaults to the CPU wheels so the image builds and runs anywhere; for the GPU build pass
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121   (and run with --gpus all).

# ---- stage 1: build the React studio --------------------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- stage 2: the API -------------------------------------------------------------------------
# Ubuntu 22.04 rather than python:3.10-slim: it ships Python 3.10 (this stack needs it: numpy<2,
# transformers<4.48), and its package archive was reachable from the build host where Debian's was not.
FROM ubuntu:22.04
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV DEBIAN_FRONTEND=noninteractive
# ffmpeg: every audio/video read and write; espeak-ng: phonemes for Kokoro and the aligner;
# libsndfile1: soundfile; git: the pinned OpenVoice install; curl: the health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.10 python3.10-venv python3-pip ffmpeg espeak-ng libsndfile1 git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m venv /opt/venv
# The virtualenv's python and pip come first on PATH, so every later "python"/"pip" is 3.10's.
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
# torch first, from the chosen index, so requirements.txt's ">=" finds it already satisfied.
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir -r backend/requirements.txt \
    # Installed without their own pins, which would pull numpy 2 / old librosa (see requirements.txt).
    && pip install --no-cache-dir --no-deps TTS==0.22.0 \
       "git+https://github.com/myshell-ai/OpenVoice.git@74a1d147b17a8c3092dd5430504bd83ef6c7eb23"

# The audio and video watermarks (every output is marked; without these the API refuses to make
# unmarked media). Separate layer: the same --no-deps rule as above, and it keeps the big layer cached.
RUN pip install --no-cache-dir --no-deps audioseal==0.2.0 omegaconf antlr4-python3-runtime==4.9.3 videoseal==1.0.1 \
    && pip install --no-cache-dir av lpips pytorch_msssim calflops decord pycocotools PyWavelets timm==0.9.16 "scikit-image<0.22" "networkx<3" \
    # Kokoro's English front end otherwise pip-installs this spaCy model on first use, at run time.
    && pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# MediaPipe's face landmarker opens EGL/GLES even when it runs on the CPU. Its own layer, so adding it
# did not invalidate the large pip layers above.
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgles2 libgl1 && rm -rf /var/lib/apt/lists/*

COPY backend/ backend/
COPY scripts/ scripts/
COPY sdk/ sdk/
COPY --from=frontend /src/frontend/dist frontend/dist

# Not root (S-16). uid 1000 so files written to the mounted outputs/ belong to the usual host user.
RUN useradd --create-home --uid 1000 app && mkdir -p inputs outputs .models && chown -R app:app /app
USER app

ENV PYTHONPATH=/app/backend \
    PYTHONUNBUFFERED=1 \
    FRONTEND_DIST=/app/frontend/dist \
    QUEUE_BACKEND=in_memory \
    # Weights come from the mounted host cache; never reach for the network at run time.
    HF_HUB_OFFLINE=1
WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fsS localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
