FROM docker.io/nvidia/cuda:13.2.0-devel-ubuntu24.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        git curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python venv
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip

# PyTorch (CUDA 13.x compatible)
RUN pip install --no-cache-dir torch torchvision torchaudio

# Flash Attention 2 (needs psutil + ninja for build)
RUN pip install --no-cache-dir psutil ninja packaging setuptools wheel
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# TTS model + CUDA-graph-accelerated inference + serving
RUN pip install --no-cache-dir qwen-tts faster-qwen3-tts soundfile fastapi uvicorn

# HuggingFace cache (mount a volume here to persist downloaded weights
# across container restarts).
ENV HF_HOME=/root/.cache/huggingface
VOLUME ["/root/.cache/huggingface"]

# Server script
COPY server.py /app/server.py

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8001/health || exit 1

# Use server.py's main() so --cpu / --port etc. work via container CMD.
ENTRYPOINT ["python3", "/app/server.py"]
CMD ["--host", "0.0.0.0", "--port", "8001"]
