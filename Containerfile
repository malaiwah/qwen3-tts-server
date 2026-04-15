# Qwen3-TTS server (Qwen3-TTS-12Hz-1.7B-Base + voice-clone path)
#
# Base: CUDA 12.8 + Ubuntu 24.04
#   - PyTorch cu128 wheels are pre-built → CI builds in < 5 min (no GPU needed)
#   - Compatible with drivers ≥ 525 (CUDA 12.x runtime)
#   - Ubuntu 24.04 LTS (Qwen3-recommended OS baseline)
#
# Performance tiers selected at runtime:
#   Tier 1 (default): faster-qwen3-tts + CUDA graphs  → ~3-3.4x real-time
#   Tier 2 (fallback if flash-attn wheel found):        → ~2x real-time
#   Tier 3 (fallback if no flash-attn):  qwen_tts+SDPA → ~1.5x real-time
#   CPU (--cpu flag):  transformers, very slow          → smoke tests only
#
# flash-attn is installed with --only-binary to prevent source compilation.
# If no prebuilt wheel matches this torch+CUDA combo, the step succeeds
# silently and tier-3 (SDPA) inference is used instead.

FROM docker.io/nvidia/cuda:12.8.0-devel-ubuntu24.04

WORKDIR /app

# System dependencies
# sox is required by qwen-tts for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        git curl ffmpeg libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

# Python virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip

# PyTorch (CUDA 12.8 pre-built wheels — no GPU needed at build time)
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# TTS inference + serving stack
# faster-qwen3-tts: tier-1 CUDA-graph-accelerated backend (~3.4x RT)
# qwen-tts:         reference backend (tier 2/3 fallback, also a dep of faster-qwen3-tts)
# safetensors:      loads the preset-voice bundle + persisted clone sidecars
# python-multipart: required by FastAPI for /v1/voices file uploads
RUN pip install --no-cache-dir \
        faster-qwen3-tts qwen-tts \
        soundfile fastapi uvicorn prometheus-client \
        safetensors python-multipart

# Flash Attention 2 (optional — tier-2 speedup for the qwen_tts fallback path).
# Only installs a pre-built binary wheel; never compiles from source so CI stays fast.
# If no prebuilt wheel is available for this torch+CUDA combo, the step is skipped
# and the server automatically falls back to PyTorch SDPA (tier 3, ~1.5x RT).
RUN pip install --no-cache-dir --only-binary :all: flash-attn \
    || echo "INFO: flash-attn prebuilt wheel not available for this torch+CUDA combination." \
            "Server will use PyTorch SDPA (tier 3 — ~1.5x real-time, vs ~2x for tier 2)." \
            "To enable tier 2: rebuild with a torch+CUDA combination that has a flash-attn wheel."

# HuggingFace weight cache (mount a volume here to persist across container restarts)
ENV HF_HOME=/root/.cache/huggingface
VOLUME ["/root/.cache/huggingface"]

# Persistent storage for user-registered voice clones (POST /v1/voices).
# Mount a volume here to keep clones across container restarts.  The preset
# voices are baked into the image under /app/assets and do not need this.
ENV QWEN3_TTS_VOICES_DIR=/data/voices
VOLUME ["/data/voices"]

COPY server.py /app/server.py
COPY voice_registry.py /app/voice_registry.py
COPY assets/ /app/assets/

EXPOSE 8001

# Health check waits up to 2 min for model load + CUDA warmup
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -sf http://localhost:8001/health | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d['model_ready'] else 1)"

ENTRYPOINT ["python3", "/app/server.py"]
CMD ["--host", "0.0.0.0", "--port", "8001"]
