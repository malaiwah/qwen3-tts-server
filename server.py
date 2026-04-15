"""Qwen3-TTS server — OpenAI-compatible /v1/audio/speech with voice cloning.

Serves Qwen3-TTS-12Hz-1.7B-**Base** with a unified voice-cloning path that
handles both preset speakers and user-registered clones:

- **Preset voices** — nine named speakers (aiden, dylan, eric, ono_anna,
  ryan, serena, sohee, uncle_fu, vivian) lifted from the CustomVoice
  checkpoint's ``codec_embedding`` and shipped as a ~37 KB sidecar bundle.
  No CustomVoice checkpoint download needed.  Provenance documented at
  ``assets/preset/bundle.json`` and on HF:
  https://huggingface.co/datasets/malaiwah/qwen3-tts-preset-voices
- **Custom clones** — ``POST /v1/voices`` with a reference audio clip
  creates a persistent clone (`vc_<8hex>` ID) usable via the same ``voice``
  parameter.  Optional ``ref_text`` enables full in-context learning mode;
  without it we fall back to x-vector-only mode.

Other features:

- Flash Attention 2 for faster GPU inference (optional, auto-detected)
- Optional CUDA-graph acceleration via ``faster-qwen3-tts`` (~5× speedup vs CPU)
- Emotion/style control via ``instruct`` parameter
- WAV, MP3, and Opus output formats
- Sentence-chunked HTTP streaming for low first-audio latency
- Token-level PCM streaming for continuous, gap-free playback (~130 ms first chunk)
- OpenAI-compatible ``/v1/models`` and ``/v1/audio/speech`` endpoints
  (accepts both JSON body with ``input`` field *and* query-parameter style)
- OpenAI voice alias mapping (alloy→ryan, echo→aiden, …)
- Optional CPU fallback (slow — for smoke tests / no-GPU environments)

Performance tiers (GPU, descending):
  1. faster-qwen3-tts + CUDA graphs  → ~3–3.4× real-time, ~130 ms first PCM chunk
  2. qwen_tts + flash_attention_2    → ~2× real-time                 (flash-attn wheel required)
  3. qwen_tts + SDPA                 → ~1.5× real-time               (no extra install needed)
  4. CPU (transformers)              → 0.01–0.05× real-time           (smoke tests only)

Tier 1 is always selected when ``faster-qwen3-tts`` is installed (default in container).
Tier 2 is selected when flash-attn is installed and faster-qwen3-tts is absent.
Tier 3 is the automatic fallback — no installation needed, but ~30% slower than tier 2.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import re
import struct
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from voice_registry import KIND_CUSTOM, KIND_PRESET, VoiceRegistry


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

MODEL_ID = os.getenv("QWEN3_TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
MAX_TEXT_LENGTH = int(os.getenv("QWEN3_TTS_MAX_TEXT_LENGTH", "10000"))

# Upper bound on reference audio upload size (bytes).  10 MB comfortably
# covers a 60 s 16-bit 48 kHz mono WAV clip plus headers.
MAX_REF_AUDIO_BYTES = int(os.getenv("QWEN3_TTS_MAX_REF_AUDIO_BYTES", str(10 * 1024 * 1024)))

# Optional auto-transcription for POST /v1/voices — when the client omits
# ``ref_text`` we can forward the audio to an ASR service (qwen3-asr-server
# or any OpenAI-compatible /v1/audio/transcriptions) and use the returned
# text to enable full in-context learning mode.  When unset, we silently
# fall back to x-vector-only mode for prompt-less registrations.
ASR_URL = os.getenv("QWEN3_TTS_ASR_URL", "") or ""
ASR_API_KEY = os.getenv("QWEN3_TTS_ASR_API_KEY", "") or ""
ASR_MODEL = os.getenv("QWEN3_TTS_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
ASR_TIMEOUT_S = float(os.getenv("QWEN3_TTS_ASR_TIMEOUT", "30"))

# Voice registry layout.  Presets ship in the repo as a ~37 KB bundle;
# custom clones persist in a writable volume (typically a Docker volume
# mounted at /data).
_REPO_ROOT = Path(__file__).resolve().parent
PRESET_BUNDLE_PATH = Path(
    os.getenv("QWEN3_TTS_PRESET_BUNDLE", str(_REPO_ROOT / "assets" / "preset" / "bundle.safetensors"))
)
PRESET_META_PATH = Path(
    os.getenv("QWEN3_TTS_PRESET_META", str(_REPO_ROOT / "assets" / "preset" / "bundle.json"))
)
VOICES_DIR = Path(os.getenv("QWEN3_TTS_VOICES_DIR", "/data/voices"))
CUSTOM_VOICES_DIR = VOICES_DIR / "custom"

# Optional bearer-token auth.  Empty/None disables auth entirely.
_API_KEY: str = os.getenv("QWEN_API_KEY", "") or ""

# Paths exempt from auth (standard ops endpoints + OpenAPI docs).
_AUTH_EXEMPT_PATHS = frozenset({
    "/", "/health", "/metrics",
    "/docs", "/redoc", "/openapi.json",
})

# OpenAI voice aliases → preset voice IDs (the nine speakers lifted from
# CustomVoice).  Unknown voices are looked up directly in the registry — so
# ``voice="vc_deadbeef"`` or ``voice="my_custom"`` flow through unchanged.
_VOICE_MAP: dict[str, str] = {
    "alloy": "ryan",
    "echo": "aiden",
    "fable": "dylan",
    "onyx": "eric",
    "nova": "serena",
    "shimmer": "vivian",
}

# Default voice when the client omits one — must match a preset or an ID
# present in the registry at startup.
_DEFAULT_VOICE = os.getenv("QWEN3_TTS_DEFAULT_VOICE", "ryan")

# Sentence boundary pattern — split on .!? followed by whitespace.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Backend selection
# -----------------------------------------------------------------------

def _select_backend(prefer_faster: bool) -> tuple:
    """Return (TTSModelClass, faster_flag, attn_impl).

    Performance tiers (from fastest to slowest):
      1. FasterQwen3TTS  → CUDA-graph accelerated (~5× faster than tier 3 on GPU)
      2. Qwen3TTSModel + flash_attention_2  → ~30% faster than tier 3
      3. Qwen3TTSModel + sdpa              → PyTorch built-in, no extra install
    """
    if prefer_faster:
        try:
            from faster_qwen3_tts import FasterQwen3TTS  # type: ignore
            return FasterQwen3TTS, True, None
        except ImportError:
            logger.warning(
                "⚠  faster-qwen3-tts not installed — falling back to qwen_tts.\n"
                "   Performance impact: ~5× slower (no CUDA-graph acceleration).\n"
                "   To restore best performance: pip install faster-qwen3-tts"
            )

    # Choose best available attention implementation for qwen_tts path.
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("Using flash_attention_2 (tier 2 performance).")
    except ImportError:
        attn_impl = "sdpa"
        logger.warning(
            "⚠  flash-attn not installed — using PyTorch SDPA (tier 3 performance).\n"
            "   Performance impact: ~30%% slower than flash_attention_2.\n"
            "   To improve: pip install flash-attn --no-build-isolation"
        )

    from qwen_tts import Qwen3TTSModel  # type: ignore
    return Qwen3TTSModel, False, attn_impl


# -----------------------------------------------------------------------
# Application state
# -----------------------------------------------------------------------

_model = None
_model_ready = False
_FASTER_TTS = False
_DEVICE = "cuda"
_INFER_SEM: asyncio.Semaphore  # initialized in lifespan

# Voice registry — constructed at import time so the module is always
# importable (tests use this), populated during lifespan startup.
_voice_registry = VoiceRegistry(
    preset_bundle_path=PRESET_BUNDLE_PATH,
    preset_meta_path=PRESET_META_PATH,
    custom_dir=CUSTOM_VOICES_DIR,
    device=_DEVICE,
    dtype=torch.bfloat16,
)


def _load_model_sync(device: str, prefer_faster: bool) -> None:
    """Synchronous model loader — runs in a background thread via asyncio.to_thread."""
    global _model, _model_ready, _FASTER_TTS

    TTSModelClass, _FASTER_TTS, attn_impl = _select_backend(prefer_faster)
    if _FASTER_TTS:
        backend_desc = "faster-qwen3-tts (CUDA graphs, tier 1)"
    elif attn_impl == "flash_attention_2":
        backend_desc = "qwen_tts + flash_attention_2 (tier 2)"
    else:
        backend_desc = "qwen_tts + SDPA (tier 3)"

    logger.info("Loading %s via %s on %s …", MODEL_ID, backend_desc, device)
    start = time.time()
    torch.set_float32_matmul_precision("high")

    if _FASTER_TTS:
        model_obj = TTSModelClass.from_pretrained(MODEL_ID, device=device, dtype="bfloat16")
    else:
        kwargs: dict = {"device_map": device, "dtype": torch.bfloat16}
        if attn_impl and device == "cuda":
            kwargs["attn_implementation"] = attn_impl
        model_obj = TTSModelClass.from_pretrained(MODEL_ID, **kwargs)

    logger.info("Model loaded in %.1fs via %s", time.time() - start, backend_desc)

    # Load voice registry now that the model is up — presets and any
    # previously-registered custom clones need the model device/dtype.
    _voice_registry.device = device
    _voice_registry.bind_model(model_obj)
    _voice_registry.load()

    # Warmup — capture CUDA graphs / JIT-compile kernels so the first real
    # request doesn't pay a 5–10 s setup penalty.  Warms up through the
    # same clone path we use in production.
    if device == "cuda":
        logger.info("Warming up CUDA graphs …")
        warmup_start = time.time()
        warmup_record = _voice_registry.get(_DEFAULT_VOICE)
        if warmup_record is None:
            presets = _voice_registry.list(kind=KIND_PRESET)
            if presets:
                warmup_record = _voice_registry.get(presets[0]["id"])
        try:
            if warmup_record is None:
                logger.warning("No preset voices loaded — skipping warmup.")
            else:
                warmup_kwargs = {
                    "text": "Warmup.",
                    "language": "English",
                    "voice_clone_prompt": [warmup_record.prompt_item],
                }
                model_obj.generate_voice_clone(**warmup_kwargs)
                if _FASTER_TTS and hasattr(model_obj, "generate_voice_clone_streaming"):
                    for _ in model_obj.generate_voice_clone_streaming(**warmup_kwargs, chunk_size=4):
                        pass
                logger.info("CUDA graph warmup done in %.1fs", time.time() - warmup_start)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUDA graph warmup failed (non-fatal): %s", exc)

    _model = model_obj
    _model_ready = True
    logger.info(
        "✓ Server ready — backend: %s — voices: %d preset, %d custom",
        backend_desc,
        len(_voice_registry.list(kind=KIND_PRESET)),
        len(_voice_registry.list(kind=KIND_CUSTOM)),
    )


# -----------------------------------------------------------------------
# Prometheus metrics (in-process, no external deps beyond prometheus_client)
# -----------------------------------------------------------------------

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    _METRICS_AVAILABLE = True
except ImportError:  # pragma: no cover — prometheus_client is a hard dep in prod
    _METRICS_AVAILABLE = False

if _METRICS_AVAILABLE:
    _M_REQUESTS = Counter(
        "qwen3_tts_requests_total",
        "Total HTTP requests handled by qwen3-tts-server.",
        ["method", "path", "status"],
    )
    _M_LATENCY = Histogram(
        "qwen3_tts_request_duration_seconds",
        "HTTP request latency (wall clock, seconds).",
        ["method", "path"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    _M_INFLIGHT = Gauge(
        "qwen3_tts_requests_in_flight",
        "HTTP requests currently being processed.",
    )
    _M_MODEL_READY = Gauge(
        "qwen3_tts_model_ready",
        "1 if the model is loaded and ready to serve, else 0.",
    )
    _M_BACKEND_INFO = Gauge(
        "qwen3_tts_backend_info",
        "Backend descriptor (value is always 1; dimensions in labels).",
        ["device", "backend", "model_id"],
    )


def _route_template(request: Request) -> str:
    """Return the registered route template for a request (low-cardinality label)."""
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return route.path
    return request.url.path


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _INFER_SEM
    # One inference at a time: CUDA-graph captures are not re-entrant.
    _INFER_SEM = asyncio.Semaphore(1)
    if _METRICS_AVAILABLE:
        _M_MODEL_READY.set(0)
    # Load model off the event loop so uvicorn stays responsive.
    await asyncio.to_thread(_load_model_sync, _DEVICE, _DEVICE == "cuda")
    if _METRICS_AVAILABLE:
        backend = "faster-qwen3-tts" if _FASTER_TTS else "qwen_tts"
        _M_BACKEND_INFO.labels(device=_DEVICE, backend=backend, model_id=MODEL_ID).set(1)
        _M_MODEL_READY.set(1 if _model_ready else 0)
    yield
    # Cleanup on shutdown (releases GPU memory for clean container stop).
    global _model
    _model = None
    if _METRICS_AVAILABLE:
        _M_MODEL_READY.set(0)
    if _DEVICE == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(title="Qwen3-TTS", lifespan=lifespan)


# -----------------------------------------------------------------------
# Auth + metrics middleware
# -----------------------------------------------------------------------

@app.middleware("http")
async def _auth_and_metrics(request: Request, call_next):
    path = request.url.path

    # Bearer-token auth (enabled when QWEN_API_KEY / --api-key is set).
    if _API_KEY and path not in _AUTH_EXEMPT_PATHS:
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {_API_KEY}"
        if auth != expected:
            return JSONResponse(
                {"error": {"message": "Invalid or missing API key.", "type": "invalid_request_error", "code": "invalid_api_key"}},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

    if not _METRICS_AVAILABLE:
        return await call_next(request)

    t0 = time.perf_counter()
    _M_INFLIGHT.inc()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        _M_INFLIGHT.dec()
        template = _route_template(request)
        elapsed = time.perf_counter() - t0
        _M_LATENCY.labels(method=request.method, path=template).observe(elapsed)
        _M_REQUESTS.labels(method=request.method, path=template, status=str(status_code)).inc()


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus exposition endpoint.  Not authenticated (standard ops practice)."""
    if not _METRICS_AVAILABLE:
        return PlainTextResponse(
            "# prometheus_client not installed\n",
            status_code=501,
        )
    _M_MODEL_READY.set(1 if _model_ready else 0)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# -----------------------------------------------------------------------
# Request parsing — supports both JSON body (OpenAI SDK) and query params
# -----------------------------------------------------------------------

class _SpeechBody(BaseModel):
    """OpenAI-compatible request body for /v1/audio/speech."""
    model: str = "tts-1"
    input: Optional[str] = None      # OpenAI canonical field name
    text: Optional[str] = None       # our original alias (kept for backward compat)
    voice: str = "ryan"
    response_format: str = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instruct: Optional[str] = None   # Qwen3-TTS style/emotion control
    language: str = "English"


async def _parse_params(request: Request, **query_defaults) -> dict:
    """Return merged params, preferring JSON body over query params.

    Handles:
    - OpenAI SDK: ``Content-Type: application/json`` with ``input`` field
    - Legacy/direct: query parameters with ``text`` field
    - Mixed: JSON body overrides query params where present
    """
    ct = request.headers.get("content-type", "")
    if "application/json" in ct:
        try:
            body = await request.json()
        except Exception:
            body = {}
        return {
            "text": body.get("input") or body.get("text") or query_defaults.get("text"),
            "voice": body.get("voice", query_defaults.get("voice", "ryan")),
            "response_format": body.get("response_format", query_defaults.get("response_format", "mp3")),
            "instruct": body.get("instruct") or query_defaults.get("instruct"),
            "language": body.get("language", query_defaults.get("language", "English")),
            "speed": float(body.get("speed", query_defaults.get("speed", 1.0))),
        }
    return {**query_defaults, "speed": float(query_defaults.get("speed", 1.0))}


def _resolve_voice(voice: str) -> str:
    """Resolve a user-supplied voice string to a registry ID.

    Tries in order:
      1. exact match on a registered voice ID (preset name or ``vc_<hex>``)
      2. OpenAI alias (alloy, echo, …) → preset name
      3. lowercased exact match (tolerate case)

    Raises HTTPException(400) if no resolution is found.
    """
    if voice in _voice_registry:
        return voice
    mapped = _VOICE_MAP.get(voice.lower())
    if mapped and mapped in _voice_registry:
        return mapped
    if voice.lower() in _voice_registry:
        return voice.lower()
    known = sorted([v["id"] for v in _voice_registry.list()])
    raise HTTPException(
        400,
        f"Unknown voice '{voice}'. Known voices: {known}. "
        "Register a new clone via POST /v1/voices.",
    )


# -----------------------------------------------------------------------
# Core synthesis helpers
# -----------------------------------------------------------------------

def _generate_audio_sync(text: str, voice_id: str, instruct: str | None, language: str) -> tuple:
    """Synchronous TTS generation — call via asyncio.to_thread.

    ``voice_id`` must already have been resolved through :func:`_resolve_voice`.
    """
    record = _voice_registry.get(voice_id)
    if record is None:
        # Shouldn't happen if _resolve_voice was used, but guard defensively.
        raise HTTPException(400, f"Unknown voice '{voice_id}'.")
    kwargs: dict = {
        "text": text,
        "language": language or "English",
        "voice_clone_prompt": [record.prompt_item],
    }
    if instruct:
        kwargs["instruct"] = instruct
    wavs, sr = _model.generate_voice_clone(**kwargs)
    return wavs[0], sr


def _wav_to_bytes_sync(wav: np.ndarray, sr: int, fmt: str) -> bytes:
    """Encode WAV array to the requested format — call via asyncio.to_thread."""
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    wav_bytes = buf.getvalue()
    if fmt in ("mp3", "opus"):
        return _transcode_wav(wav_bytes, fmt)
    return wav_bytes


def _transcode_wav(wav_bytes: bytes, fmt: str) -> bytes:
    """Convert WAV bytes to mp3/opus via ffmpeg using stdin/stdout pipes (no temp files)."""
    if fmt == "mp3":
        codec_args = ["-codec:a", "libmp3lame", "-q:a", "2"]
        out_fmt = "mp3"
    else:
        codec_args = ["-codec:a", "libopus", "-b:a", "64k"]
        out_fmt = "ogg"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", "pipe:0",      # read WAV from stdin
        *codec_args,
        "-f", out_fmt,
        "pipe:1",            # write encoded audio to stdout
    ]
    result = subprocess.run(cmd, input=wav_bytes, capture_output=True, timeout=60, check=True)
    return result.stdout


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _validate_text(text: str | None, *, required: bool = True) -> str:
    if not text or not text.strip():
        if required:
            raise HTTPException(400, "text (or 'input') is required")
        return ""
    text = text.strip()
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            413,
            f"text length {len(text)} exceeds maximum {MAX_TEXT_LENGTH}. "
            "Split into smaller chunks or use the /v1/audio/speech/stream endpoint.",
        )
    return text


# -----------------------------------------------------------------------
# Non-streaming endpoint  (OpenAI-compatible)
# -----------------------------------------------------------------------

@app.post("/v1/audio/speech")
async def speech(
    request: Request,
    text: Optional[str] = Query(None, description="Text to synthesise (use 'input' for OpenAI SDK compat)"),
    voice: str = Query("ryan", description="Speaker voice. OpenAI aliases (alloy, echo, …) are mapped automatically."),
    response_format: str = Query("mp3", description="mp3 | wav | opus"),
    instruct: Optional[str] = Query(None, description="Style/emotion hint: 'Excited and fast', 'Whisper softly', …"),
    language: str = Query("English", description="Language name, e.g. English, French, Japanese"),
    speed: float = Query(1.0, description="Speed multiplier (accepted for OpenAI compat; Qwen3-TTS does not support arbitrary speed control)"),
):
    """Single-shot TTS — returns the full audio file.

    Accepts both OpenAI-style JSON body (``Content-Type: application/json``, ``input`` field)
    and direct query parameters (``?text=...``).
    """
    if not _model_ready:
        raise HTTPException(503, "Model is still loading. Check /health and retry.")

    p = await _parse_params(
        request, text=text, voice=voice, response_format=response_format,
        instruct=instruct, language=language, speed=speed,
    )
    text_in = _validate_text(p["text"])
    voice_in = _resolve_voice(p["voice"])
    fmt = p["response_format"].lower()

    if fmt not in ("mp3", "wav", "opus"):
        raise HTTPException(400, f"response_format must be mp3, wav, or opus; got '{fmt}'")

    if p["speed"] != 1.0:
        logger.warning(
            "speed=%.2f requested but Qwen3-TTS does not support arbitrary speed control. "
            "Ignoring. Consider using a resampler in the client if speed control is critical.",
            p["speed"],
        )

    logger.info(
        "TTS: voice=%s fmt=%s lang=%s len=%d instruct=%r",
        voice_in, fmt, p["language"], len(text_in), p["instruct"],
    )

    async with _INFER_SEM:
        t0 = time.time()
        wav, sr = await asyncio.to_thread(
            _generate_audio_sync, text_in, voice_in, p["instruct"], p["language"]
        )
        gen_ms = int((time.time() - t0) * 1000)
        output = await asyncio.to_thread(_wav_to_bytes_sync, wav, sr, fmt)
        total_ms = int((time.time() - t0) * 1000)

    media_type = {"mp3": "audio/mpeg", "opus": "audio/ogg"}.get(fmt, "audio/wav")
    logger.info("TTS done: gen=%dms total=%dms output=%d bytes", gen_ms, total_ms, len(output))
    return Response(
        content=output,
        media_type=media_type,
        headers={"X-Latency-Ms": str(total_ms), "X-Gen-Time-Ms": str(gen_ms)},
    )


# -----------------------------------------------------------------------
# Sentence-chunked streaming endpoint
# -----------------------------------------------------------------------

@app.post("/v1/audio/speech/stream")
async def speech_stream(
    request: Request,
    text: Optional[str] = Query(None),
    voice: str = Query("ryan"),
    response_format: str = Query("opus"),
    instruct: Optional[str] = Query(None),
    language: str = Query("English"),
):
    """Stream TTS audio sentence-by-sentence with length-prefixed framing.

    Frame format: ``[4 bytes BE uint32 length][length bytes audio]``.
    Each frame is a self-contained audio file.  Stream ends on socket close.
    Aborts mid-stream on client disconnect (barge-in friendly).

    Also accepts JSON body (same fields as /v1/audio/speech).
    """
    if not _model_ready:
        raise HTTPException(503, "Model is still loading. Check /health and retry.")

    p = await _parse_params(
        request, text=text, voice=voice, response_format=response_format,
        instruct=instruct, language=language,
    )
    text_in = _validate_text(p["text"])
    voice_in = _resolve_voice(p["voice"])
    fmt = p["response_format"].lower()
    sentences = _split_sentences(text_in)
    if not sentences:
        raise HTTPException(400, "No sentences found in text")

    logger.info(
        "TTS stream: voice=%s fmt=%s sentences=%d len=%d instruct=%r",
        voice_in, fmt, len(sentences), len(text_in), p["instruct"],
    )

    async def generate_chunks() -> AsyncGenerator[bytes, None]:
        async with _INFER_SEM:
            for i, sentence in enumerate(sentences):
                if await request.is_disconnected():
                    logger.info("TTS stream: client disconnected at chunk %d/%d", i, len(sentences))
                    return
                t0 = time.time()
                wav, sr = await asyncio.to_thread(
                    _generate_audio_sync, sentence, voice_in, p["instruct"], p["language"]
                )
                chunk = await asyncio.to_thread(_wav_to_bytes_sync, wav, sr, fmt)
                logger.info(
                    'TTS stream chunk %d/%d: %.2fs %d bytes "%s"',
                    i + 1, len(sentences), time.time() - t0, len(chunk), sentence[:60],
                )
                yield struct.pack(">I", len(chunk)) + chunk

    return StreamingResponse(
        generate_chunks(),
        media_type="application/octet-stream",
        headers={"X-TTS-Sentences": str(len(sentences)), "X-TTS-Stream": "chunked"},
    )


# -----------------------------------------------------------------------
# Token-level PCM streaming endpoint (CUDA graphs / faster-qwen3-tts only)
# -----------------------------------------------------------------------

@app.post("/v1/audio/speech/pcm-stream")
async def speech_pcm_stream(
    request: Request,
    text: Optional[str] = Query(None),
    voice: str = Query("ryan"),
    instruct: Optional[str] = Query(None),
    chunk_size: int = Query(4, description="Codec frames per yield. Lower = lower latency, more overhead."),
    language: str = Query("English"),
):
    """Stream raw 24 kHz mono int16 PCM at the codec-frame level.

    Frame format: ``[4 bytes BE uint32 length][length bytes int16 PCM]``.
    A zero-length frame marks end of stream.

    **Requires ``faster-qwen3-tts``** (tier 1 backend).  Provides the lowest
    first-chunk latency (~130 ms on RTX 4080 SUPER / ~300 ms on A100D-20C vGPU).

    Also accepts JSON body (same fields as /v1/audio/speech).
    """
    if not _model_ready:
        raise HTTPException(503, "Model is still loading. Check /health and retry.")
    if not _FASTER_TTS:
        raise HTTPException(
            501,
            "Token-level PCM streaming requires the tier-1 backend (faster-qwen3-tts + CUDA). "
            "Ensure a CUDA GPU is present and faster-qwen3-tts is installed, then restart the server. "
            "See GET /health for current backend.",
        )

    p = await _parse_params(
        request, text=text, voice=voice, instruct=instruct, chunk_size=chunk_size, language=language,
    )
    text_in = _validate_text(p["text"])
    voice_in = _resolve_voice(p["voice"])

    logger.info("TTS PCM stream: voice=%s chunk_size=%d len=%d", voice_in, chunk_size, len(text_in))

    async def generate_pcm() -> AsyncGenerator[bytes, None]:
        async with _INFER_SEM:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            record = _voice_registry.get(voice_in)
            if record is None:
                raise HTTPException(400, f"Unknown voice '{voice_in}'.")

            def _stream_in_thread() -> None:
                """Run the synchronous CUDA-graph generator in a thread, post chunks to queue."""
                try:
                    kwargs: dict = {
                        "text": text_in,
                        "voice_clone_prompt": [record.prompt_item],
                        "language": p["language"] or "English",
                        "chunk_size": chunk_size,
                    }
                    if p.get("instruct"):
                        kwargs["instruct"] = p["instruct"]
                    for item in _model.generate_voice_clone_streaming(**kwargs):
                        loop.call_soon_threadsafe(queue.put_nowait, item)
                except Exception as exc:  # noqa: BLE001
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

            thread_task = asyncio.ensure_future(asyncio.to_thread(_stream_in_thread))
            t0 = time.time()
            chunk_num = 0
            try:
                while True:
                    if await request.is_disconnected():
                        logger.info("TTS PCM: client disconnected at chunk %d", chunk_num)
                        break
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    audio_chunk, _sr, _timing = item
                    if audio_chunk.dtype != np.int16:
                        audio_chunk = (audio_chunk * 32767).clip(-32768, 32767).astype(np.int16)
                    pcm_bytes = audio_chunk.tobytes()
                    chunk_num += 1
                    if chunk_num <= 2:
                        logger.info(
                            "TTS PCM first chunk %d: %d samples %.3fs",
                            chunk_num, len(audio_chunk), time.time() - t0,
                        )
                    yield struct.pack(">I", len(pcm_bytes)) + pcm_bytes
            finally:
                if not thread_task.done():
                    thread_task.cancel()

            yield struct.pack(">I", 0)  # end-of-stream sentinel
            logger.info("TTS PCM stream: %d chunks in %.2fs", chunk_num, time.time() - t0)

    return StreamingResponse(
        generate_pcm(),
        media_type="application/octet-stream",
        headers={
            "X-TTS-Stream": "pcm",
            "X-TTS-Sample-Rate": "24000",
            "X-TTS-Channels": "1",
            "X-TTS-Sample-Width": "2",
        },
    )


# -----------------------------------------------------------------------
# Info / introspection endpoints
# -----------------------------------------------------------------------

@app.get("/health")
async def health():
    backend = "faster-qwen3-tts (CUDA graphs, tier 1)" if _FASTER_TTS else "qwen_tts (tier 2/3)"
    return {
        "status": "healthy" if _model_ready else "loading",
        "model_ready": _model_ready,
        "model_id": MODEL_ID,
        "backend": backend,
        "device": _DEVICE,
        "voices": {
            "preset": len(_voice_registry.list(kind=KIND_PRESET)),
            "custom": len(_voice_registry.list(kind=KIND_CUSTOM)),
        },
    }


@app.get("/voices")
async def voices_legacy():
    """Legacy voices endpoint — kept for backward compatibility.

    New callers should prefer ``GET /v1/voices``.
    """
    return await list_voices()


@app.get("/v1/voices")
async def list_voices():
    """List all voices — presets and user-registered clones — plus aliases.

    Response shape::

        {
          "data": [
            {"id": "aiden", "kind": "preset", "name": "aiden", "gender": "m", "lang": "en,zh"},
            {"id": "vc_ab12cd34", "kind": "custom", "name": "Alice", "created_at": 1734567890, …},
            ...
          ],
          "openai_voice_aliases": {"alloy": "ryan", …}
        }
    """
    return {
        "data": _voice_registry.list(),
        "openai_voice_aliases": _VOICE_MAP,
        "note": (
            "OpenAI voices (alloy, echo, fable, onyx, nova, shimmer) are accepted "
            "and mapped automatically to preset speakers. Register custom clones "
            "via POST /v1/voices."
        ),
    }


async def _transcribe_ref_audio(
    audio_bytes: bytes,
    filename: str,
    language: Optional[str],
) -> Optional[str]:
    """Best-effort transcription of the reference clip via an ASR service.

    Returns the transcribed text on success, ``None`` on any failure
    (network, auth, malformed response).  Failures are logged but non-fatal
    — the caller falls back to x-vector-only clone mode.

    Configured by ``QWEN3_TTS_ASR_URL`` (required; disables this helper when
    empty), ``QWEN3_TTS_ASR_API_KEY`` (bearer token), ``QWEN3_TTS_ASR_MODEL``,
    and ``QWEN3_TTS_ASR_TIMEOUT``.  Protocol is OpenAI-compatible
    ``POST /v1/audio/transcriptions`` with multipart form, matching
    qwen3-asr-server and the reference OpenAI endpoint.
    """
    if not ASR_URL:
        return None
    # Local import so httpx is only loaded when ASR integration is actually
    # wired up.  httpx is already a transitive dep via starlette's TestClient
    # and is installed explicitly in production.
    import httpx

    url = ASR_URL.rstrip("/") + "/v1/audio/transcriptions"
    files = {"file": (filename, audio_bytes, "application/octet-stream")}
    data: dict[str, str] = {"model": ASR_MODEL, "response_format": "json"}
    if language:
        data["language"] = language
    headers: dict[str, str] = {}
    if ASR_API_KEY:
        headers["Authorization"] = f"Bearer {ASR_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=ASR_TIMEOUT_S) as client:
            resp = await client.post(url, data=data, files=files, headers=headers)
        if resp.status_code >= 400:
            logger.warning(
                "Auto-transcription returned HTTP %d; falling back to x-vector-only. body=%s",
                resp.status_code, resp.text[:300],
            )
            return None
        body = resp.json()
        text = (body.get("text") or "").strip()
        if not text:
            logger.warning("Auto-transcription returned empty text; falling back.")
            return None
        logger.info("Auto-transcription: %d chars (lang=%s)", len(text), language)
        return text
    except Exception as exc:  # noqa: BLE001
        # Network errors, JSON decode errors, timeouts — all non-fatal.
        logger.warning("Auto-transcription failed (%s); falling back to x-vector-only.", exc)
        return None


@app.post("/v1/voices")
async def register_voice(
    name: str = Form(..., description="Human-friendly label for the voice."),
    audio: UploadFile = File(..., description="Reference audio clip (WAV/MP3/FLAC/OGG). 3–30 s recommended."),
    ref_text: Optional[str] = Form(
        None,
        description="Optional transcript of the reference audio. When provided (or when auto-transcription via QWEN3_TTS_ASR_URL is configured), enables full in-context learning mode (closer clone, prosody-aware).",
    ),
    language: Optional[str] = Form(None, description="Language hint, e.g. 'English'. Forwarded to ASR when auto-transcribing."),
    auto_transcribe: Optional[bool] = Form(
        None,
        description="Per-request override. Defaults to True when QWEN3_TTS_ASR_URL is set. Pass False to force x-vector-only mode even when ASR is configured.",
    ),
):
    """Register a new custom voice from a reference audio clip.

    Returns the voice record (including the assigned ``id``, e.g.
    ``vc_ab12cd34``) which can then be used as the ``voice`` parameter on
    ``/v1/audio/speech`` and its streaming variants.

    Transcription handling:
      - ``ref_text`` provided  → full ICL mode with that text.
      - ``ref_text`` omitted, ASR configured, auto_transcribe != False
        → forward ``audio`` to the ASR service, use returned text.
      - ``ref_text`` omitted, ASR unavailable/disabled/failed
        → x-vector-only mode (no prosody conditioning).
    """
    if not _model_ready:
        raise HTTPException(503, "Model is still loading. Check /health and retry.")
    if not name.strip():
        raise HTTPException(400, "name is required")

    data = await audio.read()
    if not data:
        raise HTTPException(400, "audio file is empty")
    if len(data) > MAX_REF_AUDIO_BYTES:
        raise HTTPException(
            413,
            f"audio file exceeds {MAX_REF_AUDIO_BYTES} bytes. "
            "Trim to under ~30 s or increase QWEN3_TTS_MAX_REF_AUDIO_BYTES.",
        )

    suffix = Path(audio.filename or "").suffix.lower() or ".wav"
    if suffix not in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"):
        logger.info("Unusual audio suffix %r — soundfile/ffmpeg will attempt decode.", suffix)

    transcribed = False
    if (
        (not ref_text or not ref_text.strip())
        and ASR_URL
        and auto_transcribe is not False
    ):
        asr_text = await _transcribe_ref_audio(
            data, audio.filename or f"sample{suffix}", language,
        )
        if asr_text:
            ref_text = asr_text
            transcribed = True

    # Clone-prompt creation runs a model forward (ECAPA + optional tokeniser)
    # on the GPU, so acquire the inference semaphore.
    async with _INFER_SEM:
        try:
            record = await asyncio.to_thread(
                _voice_registry.register,
                name=name,
                ref_audio_bytes=data,
                ref_audio_suffix=suffix,
                ref_text=ref_text,
                language=language,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            # create_voice_clone_prompt is the most likely failure mode
            # (unsupported audio, ref_text mismatch).  Surface as 422.
            logger.exception("register_voice failed")
            raise HTTPException(422, f"could not create clone prompt: {exc}") from exc

    payload = record.summary()
    # Signal to the caller whether we auto-filled ref_text so they can
    # surface that in their UI (e.g. "Transcribed via ASR: ...").
    payload["auto_transcribed"] = transcribed
    return JSONResponse(status_code=201, content=payload)


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a registered custom voice.  Presets cannot be deleted."""
    try:
        removed = _voice_registry.delete(voice_id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not removed:
        raise HTTPException(404, f"voice '{voice_id}' not found")
    return {"id": voice_id, "deleted": True}


@app.get("/v1/models")
async def list_models():
    if not _model_ready:
        raise HTTPException(503, "Model is still loading")
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qwen_tts",
                "root": MODEL_ID,
                "parent": None,
                "max_model_len": 4096,
            },
            # Expose OpenAI model IDs so drop-in clients work unchanged.
            {"id": "tts-1",    "object": "model", "created": int(time.time()), "owned_by": "openai-compat"},
            {"id": "tts-1-hd", "object": "model", "created": int(time.time()), "owned_by": "openai-compat"},
        ],
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    if not _model_ready:
        raise HTTPException(503, "Model is still loading")
    openai_aliases = {"tts-1", "tts-1-hd"}
    if model_id not in (MODEL_ID, *openai_aliases):
        raise HTTPException(404, f"Model '{model_id}' not found. See GET /v1/models.")
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "openai-compat" if model_id in openai_aliases else "qwen_tts",
        "root": MODEL_ID,
        "parent": None,
    }


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS server (voice cloning on Qwen3-TTS-12Hz-1.7B-Base)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Performance tiers (GPU, fastest first):
  Tier 1: faster-qwen3-tts + CUDA graphs  ~3-3.4x real-time, ~130ms first PCM chunk
  Tier 2: qwen_tts + flash_attention_2    ~2x real-time  (needs: pip install flash-attn)
  Tier 3: qwen_tts + SDPA                 ~1.5x real-time (built into PyTorch 2.x)
  CPU:     transformers fallback           0.01-0.05x RT  (smoke tests only)

Tier is selected automatically based on what is installed.
""",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode. WARNING: orders of magnitude slower; for smoke tests only.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Require 'Authorization: Bearer <key>' on /v1/* endpoints. "
             "Overrides the QWEN_API_KEY env var when provided. "
             "/health and /metrics remain unauthenticated.",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    global _DEVICE, _API_KEY
    if args.api_key is not None:
        _API_KEY = args.api_key
    if _API_KEY:
        logger.info("API-key auth enabled — /v1/* endpoints require 'Authorization: Bearer <key>'.")

    if args.cpu or not torch.cuda.is_available():
        if not args.cpu:
            logger.warning("No CUDA device detected; falling back to CPU. Performance will be very slow.")
        else:
            logger.warning("--cpu flag set; running on CPU. Generation will be very slow.")
        _DEVICE = "cpu"
    else:
        _DEVICE = "cuda"

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
