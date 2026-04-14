"""Qwen3-TTS CustomVoice server — OpenAI-compatible /v1/audio/speech endpoint.

Serves Qwen3-TTS-12Hz-1.7B-CustomVoice with:

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
from typing import AsyncGenerator, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

MODEL_ID = os.getenv("QWEN3_TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
MAX_TEXT_LENGTH = int(os.getenv("QWEN3_TTS_MAX_TEXT_LENGTH", "10000"))

# OpenAI voice aliases → Qwen3 speaker names
# Unknown voices are passed through unchanged (allows custom speakers).
_VOICE_MAP: dict[str, str] = {
    "alloy": "ryan",
    "echo": "aiden",
    "fable": "dylan",
    "onyx": "eric",
    "nova": "serena",
    "shimmer": "vivian",
}

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

    # Warmup — capture CUDA graphs / JIT-compile kernels so the first real
    # request doesn't pay a 5–10 s setup penalty.
    if device == "cuda":
        logger.info("Warming up CUDA graphs …")
        warmup_start = time.time()
        try:
            warmup_kwargs = {"text": "Warmup.", "language": "English", "speaker": "ryan"}
            model_obj.generate_custom_voice(**warmup_kwargs)
            if _FASTER_TTS and hasattr(model_obj, "generate_custom_voice_streaming"):
                for _ in model_obj.generate_custom_voice_streaming(**warmup_kwargs, chunk_size=4):
                    pass
            logger.info("CUDA graph warmup done in %.1fs", time.time() - warmup_start)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUDA graph warmup failed (non-fatal): %s", exc)

    _model = model_obj
    _model_ready = True
    logger.info("✓ Server ready — backend: %s", backend_desc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _INFER_SEM
    # One inference at a time: CUDA-graph captures are not re-entrant.
    _INFER_SEM = asyncio.Semaphore(1)
    # Load model off the event loop so uvicorn stays responsive.
    await asyncio.to_thread(_load_model_sync, _DEVICE, _DEVICE == "cuda")
    yield
    # Cleanup on shutdown (releases GPU memory for clean container stop).
    global _model
    _model = None
    if _DEVICE == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(title="Qwen3-TTS CustomVoice", lifespan=lifespan)


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
    """Map OpenAI voice names to Qwen3 speaker names; pass others through."""
    return _VOICE_MAP.get(voice.lower(), voice)


# -----------------------------------------------------------------------
# Core synthesis helpers
# -----------------------------------------------------------------------

def _generate_audio_sync(text: str, voice: str, instruct: str | None, language: str) -> tuple:
    """Synchronous TTS generation — call via asyncio.to_thread."""
    kwargs: dict = {"text": text, "language": language or "English", "speaker": voice}
    if instruct:
        kwargs["instruct"] = instruct
    wavs, sr = _model.generate_custom_voice(**kwargs)
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

            def _stream_in_thread() -> None:
                """Run the synchronous CUDA-graph generator in a thread, post chunks to queue."""
                try:
                    kwargs: dict = {
                        "text": text_in,
                        "speaker": voice_in,
                        "language": p["language"] or "English",
                        "chunk_size": chunk_size,
                    }
                    if p.get("instruct"):
                        kwargs["instruct"] = p["instruct"]
                    for item in _model.generate_custom_voice_streaming(**kwargs):
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
    }


@app.get("/voices")
async def voices():
    """List available speaker voices plus OpenAI voice alias mapping."""
    speakers = _model.get_supported_speakers() if _model_ready else []
    return {
        "speakers": speakers,
        "openai_voice_aliases": _VOICE_MAP,
        "note": "OpenAI voices (alloy, echo, fable, onyx, nova, shimmer) are accepted and mapped automatically.",
    }


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
        description="Qwen3-TTS CustomVoice server",
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
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    global _DEVICE
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
