# qwen3-tts-server

OpenAI-compatible HTTP server for [**Qwen3-TTS-12Hz-1.7B-CustomVoice**](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) with **token-level PCM streaming**, **sentence chunking**, **emotion control**, and **bilingual** generation.

Built for low-latency conversational voice agents — the first audio frame arrives in **~130 ms** on an RTX 4080 SUPER and the model generates speech at **~3–3.4× real-time**.

> Companion project: [**qwen3-asr-server**](https://github.com/malaiwah/qwen3-asr-server) — the matching speech-to-text server.

---

## What you get

- 🎙️ `/v1/audio/speech` — OpenAI-compatible single-shot synthesis (WAV/MP3/Opus)
  - Accepts **JSON body** (`input` field) **or** query parameters — drop-in for OpenAI SDK
  - OpenAI voice aliases (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) mapped automatically
- 📦 `/v1/audio/speech/stream` — sentence-chunked streaming (length-prefixed framing)
- ⚡ `/v1/audio/speech/pcm-stream` — **token-level** PCM streaming (~130 ms first chunk)
- 🎭 `instruct=` parameter for emotion/style control (`"Excited and speak quickly."`, `"Whisper softly."`)
- 🗣️ 9 built-in voices (`ryan`, `aiden`, `dylan`, `eric`, `serena`, `vivian`, `sohee`, `ono_anna`, `uncle_fu`)
- 🌍 Multilingual — English, French, German, Spanish, Italian, Portuguese, Japanese, Korean, Chinese
- 🚦 Barge-in friendly — streams abort cleanly on client disconnect
- 🐳 Docker/Podman — single-container deploy, Ubuntu 24.04, CUDA 12.8
- 🐌 CPU fallback for smoke tests (`--cpu`)

---

## Performance tiers

| Tier | Backend | Requirement | ~Real-time factor | First PCM chunk |
|------|---------|-------------|-------------------|-----------------|
| **1 (default)** | `faster-qwen3-tts` + CUDA graphs | `pip install faster-qwen3-tts` (in container by default) | **~3–3.4×** | **~130 ms** |
| 2 | `qwen_tts` + flash_attention_2 | `pip install flash-attn` | ~2× | ~400 ms |
| 3 | `qwen_tts` + SDPA | PyTorch 2.x built-in, no extra install | ~1.5× | ~500 ms |
| CPU | transformers | no GPU required | ~0.01–0.05× | 30+ s |

The server selects the best available tier at startup and logs which one it's using.
The container image always installs `faster-qwen3-tts` (tier 1).

---

## Host prerequisites (GPU)

Before running the container, install the NVIDIA driver and container toolkit on the host.

```bash
# --- NVIDIA driver (Ubuntu 24.04, from the official CUDA repo) ---
# Open kernel module — recommended for Turing, Ampere, Ada Lovelace, Blackwell
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
  | sudo gpg --dearmor -o /etc/apt/keyrings/nvidia-cuda.gpg
printf 'Types: deb\nURIs: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/\nSuites: /\nSigned-By: /etc/apt/keyrings/nvidia-cuda.gpg\n' \
  | sudo tee /etc/apt/sources.list.d/nvidia-cuda.sources
sudo apt-get update
sudo apt-get install -y nvidia-driver-open nvidia-container-toolkit

# For Docker — configure runtime and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# For Podman — generate CDI specs (enables --device nvidia.com/gpu=all)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Verify
nvidia-smi
```

> **Driver note**: `nvidia-driver-open` uses the open kernel module and works on Turing+.
> For Pascal and older GPUs use the proprietary variant (`nvidia-driver-XXX`).

---

## Quickstart (Docker / Podman)

```bash
# 1. Run the container — mount a volume for the HF model cache (~3 GB on first run)
docker run -d --name qwen3-tts \
  --gpus all \
  -p 8001:8001 \
  -v qwen3-hf-cache:/root/.cache/huggingface \
  ghcr.io/malaiwah/qwen3-tts-server:latest

# Podman equivalent (requires CDI — see Host prerequisites above):
# podman run -d --name qwen3-tts \
#   --device nvidia.com/gpu=all \
#   -p 8001:8001 \
#   -v qwen3-hf-cache:/root/.cache/huggingface \
#   ghcr.io/malaiwah/qwen3-tts-server:latest

# (Optional) pass a HuggingFace token if needed:
#   -e HF_TOKEN=hf_xxx

# 2. Wait for the model to load (first run downloads ~3 GB; subsequent runs reuse volume)
docker logs -f qwen3-tts   # look for "✓ Server ready — backend: faster-qwen3-tts"

# 3. Try it — using OpenAI-compatible JSON body
curl -s -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello from Qwen3","voice":"alloy"}' \
  -o hello.mp3
mpv hello.mp3

# Or query-parameter style (also supported):
curl -s -X POST "http://localhost:8001/v1/audio/speech?text=Hello+from+Qwen3&voice=ryan" \
  -o hello.mp3
```

---

## Quickstart (uv, no container)

```bash
git clone https://github.com/malaiwah/qwen3-tts-server.git
cd qwen3-tts-server
uv venv && source .venv/bin/activate

# GPU (tier 1 — fastest, ~3.4× real-time):
uv pip install torch --index-url https://download.pytorch.org/whl/cu128  # match your CUDA
uv pip install -e ".[gpu]"   # installs faster-qwen3-tts
python server.py              # starts on :8001

# CPU fallback (very slow, smoke tests only):
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e "."
python server.py --cpu

# In another shell:
./test-tts.py "Hello from Qwen3"                              # writes out.mp3
./test-tts.py "Bonjour le monde" --language French --voice aiden
./test-tts.py "Speak slowly" --instruct "Calm and slow." -o slow.wav --format wav
```

---

## OpenAI SDK drop-in

The server accepts the exact JSON body format the OpenAI Python SDK sends:

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8001/v1",
)

# OpenAI voice aliases are remapped automatically:
# alloy→ryan, echo→aiden, fable→dylan, onyx→eric, nova→serena, shimmer→vivian
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="The quick brown fox jumps over the lazy dog.",
)
response.stream_to_file("output.mp3")
```

---

## Round-trip with qwen3-asr-server

```mermaid
flowchart LR
    A[Text prompt] -->|POST /v1/audio/speech| B[qwen3-tts-server<br/>:8001]
    B -->|MP3/WAV/PCM| C[Audio file]
    C -->|POST /v1/audio/transcriptions| D[qwen3-asr-server<br/>:8002]
    D -->|JSON transcript| E[Recovered text]
    E -.compare.-> A
```

```bash
# 1. Synthesise
./test-tts.py "The quick brown fox jumps over the lazy dog." -o sample.wav --format wav

# 2. Transcribe (using qwen3-asr-server)
./test-asr.py sample.wav   # see github.com/malaiwah/qwen3-asr-server
```

To run both services together with automatic GPU sharing and startup ordering:

```bash
# Clone either repo or save the docker-compose.yml, then:
HF_TOKEN=hf_xxx docker compose up -d
```

See [`docker-compose.yml`](docker-compose.yml) for the full configuration including VRAM budget notes.

---

## Hardware reference (tested)

### Primary (benchmarks below)

| Component | Spec |
|---|---|
| **GPU** | NVIDIA GeForce RTX 4080 SUPER (16 GB VRAM, Ada Lovelace) |
| **CPU** | Intel Core i7-14700 KF (20 cores / 28 threads) |
| **RAM** | 32 GB DDR5 |
| **OS** | Ubuntu 24.04.4 LTS |
| **Driver** | NVIDIA 595.58.03 (CUDA 13.x, driver-compat tier) |

### Also validated on

| GPU | VRAM | Driver | CUDA compat | Notes |
|-----|------|--------|-------------|-------|
| NVIDIA GRID A100D-20C (Vultr vGPU) | 20 GB | 550.90.07 | ≤ 12.4 | cu128 image works fine; driver locks at 550 |
| CPU-only (no GPU) | — | — | — | `--cpu` flag; very slow |

### Performance (RTX 4080 SUPER, tier-1 backend)

| Workload | Length | Wall time | Real-time factor |
|---|---|---|---|
| First PCM chunk (320 ms audio) | 7 680 samples | **131 ms** | ~2.4× |
| Short reply (~9 s audio) | 27 chunks | 2.66 s | ~3.4× |
| Medium reply (~40 s audio) | 126 chunks | 12.58 s | ~3.2× |
| Long reply (~80 s audio) | 255 chunks | 25.47 s | ~3.2× |

VRAM footprint: **~4.4 GB** (bfloat16 + CUDA-graph warmup).

### Performance (GRID A100D-20C vGPU, tier-1 backend)

*Measured on Vultr GRID A100D-20C (20 GB vGPU), driver 550.90.07, CUDA 12.4,
cu128 container, faster-qwen3-tts (CUDA graphs, tier 1).*

The A100D-20C is a virtualised 20 GB slice of an A100 80 GB. The vGPU hypervisor
allocates ≈25% of the compute budget, so single-stream inference is slower than a
consumer card despite the HBM2e memory:

| Workload | Audio | Wall time | Real-time factor | Step time |
|---|---|---|---|---|
| Short sentence (9 words) | ~3 s | ~2.3 s | ~1.4× | ~57 ms/step |
| First PCM chunk | 320 ms | **~350 ms** | — | — |

VRAM footprint: **~14.4 GB** when co-located with the ASR vLLM server (18.8 GB / 20 GB total).

---

## API reference

### `POST /v1/audio/speech` — single-shot synthesis

Accepts **JSON body** or **query parameters**.

| Param | Default | Notes |
|---|---|---|
| `input` / `text` | *(required)* | Text to synthesise. `input` preferred (OpenAI SDK compat). |
| `model` | `tts-1` | Accepted for OpenAI compat; always uses Qwen3-TTS-CustomVoice. |
| `voice` | `ryan` | Speaker name or OpenAI alias. See `GET /voices`. |
| `language` | `English` | `English`, `French`, `Chinese`, `German`, `Spanish`, … |
| `instruct` | *(none)* | Style hint, e.g. `"Excited and speak quickly."` |
| `response_format` | `mp3` | `mp3`, `wav`, or `opus` |
| `speed` | `1.0` | Accepted for OpenAI compat; Qwen3-TTS does not support speed control. |

### `POST /v1/audio/speech/stream` — sentence-chunked

Same parameters as above (JSON body or query params). Streams one self-contained audio file
per sentence with length-prefixed framing:

```
[4 bytes BE uint32: chunk_length][chunk_length bytes: audio]
[4 bytes BE uint32: chunk_length][chunk_length bytes: audio]
... (stream ends on socket close)
```

### `POST /v1/audio/speech/pcm-stream` — token-level PCM (tier-1 only)

Yields raw **24 kHz mono int16 PCM** frames as the model generates them.
First chunk latency ≈ **130 ms** on RTX 4080 SUPER.
A zero-length frame marks end of stream.
**Requires `faster-qwen3-tts`** (tier-1 backend — installed by default in the container).

| Param | Default | Notes |
|---|---|---|
| `chunk_size` | `4` | Codec frames per yield (lower = lower latency, more overhead) |

### `GET /voices` — available speakers

Returns all supported speaker names plus the OpenAI alias mapping.

### `GET /health`, `GET /v1/models`

Standard introspection endpoints.  `/health` includes the active backend tier.

---

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `QWEN3_TTS_MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Override the HF model |
| `QWEN3_TTS_MAX_TEXT_LENGTH` | `10000` | Maximum input characters (returns HTTP 413 if exceeded) |
| `HF_HOME` | `/root/.cache/huggingface` | Where weights are cached. **Mount a volume here.** |
| `HF_TOKEN` | *(unset)* | HuggingFace token for gated downloads |

CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8001` | Listen port |
| `--cpu` | *(off)* | Force CPU mode. **Very slow.** |
| `--log-level` | `info` | Uvicorn log level |

---

## Building from source

```bash
docker build -t qwen3-tts-server:latest -f Containerfile .
# or
podman build -t qwen3-tts-server:latest -f Containerfile .
```

The CI workflow builds and pushes to `ghcr.io/malaiwah/qwen3-tts-server:latest`
on every push to `main` and version tags (`v0.1.0`, etc.).

---

## Tests

Smoke tests run without a GPU or model load and are CI-safe:

```bash
uv pip install -e ".[test]"
pytest -q
```

---

## CPU mode

CPU inference is **orders of magnitude slower** — 30 s+ per short sentence.
Use it only for smoke tests, API surface exploration, or environments without a GPU.

For conversational use, a CUDA GPU with ≥ 6 GB VRAM is required.

---

## Acknowledgements

- [Qwen team @ Alibaba](https://huggingface.co/Qwen) for the [Qwen3-TTS-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) model
- [`faster-qwen3-tts`](https://pypi.org/project/faster-qwen3-tts/) for CUDA-graph-accelerated inference
- [`qwen-tts`](https://pypi.org/project/qwen-tts/) for the reference inference path

---

## License

[MIT](LICENSE) — free for any use; please credit the upstream Qwen model card and follow its license terms separately.
