"""Qwen3-TTS CustomVoice server — OpenAI-compatible /v1/audio/speech endpoint.

Serves Qwen3-TTS-12Hz-1.7B-CustomVoice with:

- Flash Attention 2 for faster GPU inference
- Optional CUDA-graph acceleration via ``faster-qwen3-tts`` (~5× speedup)
- Emotion/style control via ``instruct`` parameter
- WAV, MP3, and Opus output formats
- Sentence-chunked HTTP streaming for low first-audio latency
- Token-level PCM streaming for continuous, gap-free playback
- OpenAI-compatible ``/v1/models`` and ``/v1/audio/speech`` endpoints
- Optional CPU fallback (slow — for smoke tests / no-GPU environments)
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import struct
import subprocess
import tempfile
import time
from typing import AsyncGenerator

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

MODEL_ID = os.getenv("QWEN3_TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")

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

def _select_backend(prefer_faster: bool):
    """Return (TTSModelClass, faster_flag, attn_impl)."""
    if prefer_faster:
        try:
            from faster_qwen3_tts import FasterQwen3TTS  # type: ignore
            return FasterQwen3TTS, True, None
        except ImportError:
            logger.warning(
                "faster-qwen3-tts not installed; falling back to qwen_tts. "
                "Install with: pip install faster-qwen3-tts"
            )
    from qwen_tts import Qwen3TTSModel  # type: ignore
    return Qwen3TTSModel, False, "flash_attention_2"


# -----------------------------------------------------------------------
# Application state
# -----------------------------------------------------------------------

app = FastAPI(title="Qwen3-TTS CustomVoice")
model = None
model_ready = False
_FASTER_TTS = False
_DEVICE = "cuda"


def _load_model(device: str, prefer_faster: bool) -> None:
    """Load the TTS model onto the requested device."""
    global model, model_ready, _FASTER_TTS

    TTSModelClass, _FASTER_TTS, attn_impl = _select_backend(prefer_faster)
    backend = "faster-qwen3-tts (CUDA graphs)" if _FASTER_TTS else "qwen_tts (standard)"

    logger.info("Loading %s via %s on %s ...", MODEL_ID, backend, device)
    start = time.time()
    torch.set_float32_matmul_precision("high")

    if _FASTER_TTS:
        model_obj = TTSModelClass.from_pretrained(MODEL_ID, device=device, dtype="bfloat16")
    else:
        kwargs = {"device_map": device, "dtype": torch.bfloat16}
        if attn_impl and device == "cuda":
            kwargs["attn_implementation"] = attn_impl
        model_obj = TTSModelClass.from_pretrained(MODEL_ID, **kwargs)

    logger.info("Model loaded in %.1fs via %s", time.time() - start, backend)

    # Warmup — capture CUDA graphs / JIT-compile kernels so the first real
    # request doesn't pay a 5-10s setup penalty.
    if device == "cuda":
        logger.info("Warming up CUDA graphs ...")
        warmup_start = time.time()
        try:
            warmup_kwargs = {"text": "Warmup.", "language": "English", "speaker": "ryan"}
            model_obj.generate_custom_voice(**warmup_kwargs)
            if _FASTER_TTS and hasattr(model_obj, "generate_custom_voice_streaming"):
                for _ in model_obj.generate_custom_voice_streaming(**warmup_kwargs, chunk_size=4):
                    pass
            logger.info("CUDA graph warmup done in %.1fs", time.time() - warmup_start)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUDA graph warmup failed: %s", exc)

    globals()["model"] = model_obj
    globals()["model_ready"] = True


@app.on_event("startup")
async def startup_load_model() -> None:
    _load_model(_DEVICE, prefer_faster=(_DEVICE == "cuda"))


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _generate_audio(text: str, voice: str, instruct: str | None, language: str) -> tuple:
    kwargs = {"text": text, "language": language or "English", "speaker": voice}
    if instruct:
        kwargs["instruct"] = instruct
    wavs, sr = model.generate_custom_voice(**kwargs)
    return wavs[0], sr


def _wav_array_to_bytes(wav: np.ndarray, sr: int, fmt: str) -> bytes:
    wav_buf = io.BytesIO()
    sf.write(wav_buf, wav, sr, format="WAV")
    wav_bytes = wav_buf.getvalue()
    if fmt in ("mp3", "opus"):
        return _convert_wav_bytes(wav_bytes, fmt)
    return wav_bytes


def _convert_wav_bytes(wav_bytes: bytes, fmt: str) -> bytes:
    """Convert WAV bytes to mp3 or opus via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as inp:
        inp.write(wav_bytes)
        inp_path = inp.name
    out_suffix = ".mp3" if fmt == "mp3" else ".ogg"
    out_path = inp_path.replace(".wav", out_suffix)
    try:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", inp_path]
        if fmt == "mp3":
            cmd += ["-codec:a", "libmp3lame", "-q:a", "2"]
        else:
            cmd += ["-codec:a", "libopus", "-b:a", "64k"]
        cmd.append(out_path)
        subprocess.run(cmd, capture_output=True, timeout=30, check=True)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (inp_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


# -----------------------------------------------------------------------
# Non-streaming endpoint
# -----------------------------------------------------------------------

@app.post("/v1/audio/speech")
async def speech(
    text: str | None = None,
    voice: str = "ryan",
    response_format: str = "mp3",
    instruct: str | None = None,
    language: str = "English",
):
    if not model_ready:
        raise HTTPException(503, "Model loading")
    if not text:
        raise HTTPException(400, "text required")

    logger.info(
        "TTS: voice=%s fmt=%s lang=%s len=%d instruct=%r",
        voice, response_format, language, len(text), instruct,
    )

    start = time.time()
    wav, sr = _generate_audio(text, voice, instruct, language)
    gen_time = time.time() - start

    fmt = response_format.lower()
    output = _wav_array_to_bytes(wav, sr, fmt)

    media_type = {
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
    }.get(fmt, "audio/wav")

    total_time = time.time() - start
    logger.info(
        "TTS done: gen=%.2fs total=%.2fs output=%d bytes",
        gen_time, total_time, len(output),
    )
    return Response(
        content=output,
        media_type=media_type,
        headers={
            "X-Latency-Ms": str(int(total_time * 1000)),
            "X-Gen-Time-Ms": str(int(gen_time * 1000)),
        },
    )


# -----------------------------------------------------------------------
# Sentence-chunked streaming endpoint
# -----------------------------------------------------------------------

@app.post("/v1/audio/speech/stream")
async def speech_stream(
    request: Request,
    text: str | None = None,
    voice: str = "ryan",
    response_format: str = "opus",
    instruct: str | None = None,
    language: str = "English",
):
    """Stream TTS audio sentence-by-sentence, length-prefixed framing.

    Frame format: ``[4 bytes BE uint32 length][length bytes audio]``.
    Each frame is a self-contained audio file.  Stream ends on socket close.
    Aborts mid-stream on client disconnect (barge-in friendly).
    """
    if not model_ready:
        raise HTTPException(503, "Model loading")
    if not text:
        raise HTTPException(400, "text required")

    sentences = _split_sentences(text)
    if not sentences:
        raise HTTPException(400, "No sentences found in text")

    fmt = response_format.lower()
    logger.info(
        "TTS stream: voice=%s fmt=%s sentences=%d len=%d instruct=%r",
        voice, fmt, len(sentences), len(text), instruct,
    )

    async def generate_chunks() -> AsyncGenerator[bytes, None]:
        for i, sentence in enumerate(sentences):
            if await request.is_disconnected():
                logger.info(
                    "TTS stream: client disconnected after chunk %d/%d, aborting",
                    i, len(sentences),
                )
                return
            start = time.time()
            wav, sr = _generate_audio(sentence, voice, instruct, language)
            chunk = _wav_array_to_bytes(wav, sr, fmt)
            elapsed = time.time() - start
            logger.info(
                'TTS stream chunk %d/%d: %.2fs, %d bytes, "%s"',
                i + 1, len(sentences), elapsed, len(chunk), sentence[:60],
            )
            yield struct.pack(">I", len(chunk)) + chunk

    return StreamingResponse(
        generate_chunks(),
        media_type="application/octet-stream",
        headers={
            "X-TTS-Sentences": str(len(sentences)),
            "X-TTS-Stream": "chunked",
        },
    )


# -----------------------------------------------------------------------
# Token-level PCM streaming endpoint (CUDA graphs only)
# -----------------------------------------------------------------------

@app.post("/v1/audio/speech/pcm-stream")
async def speech_pcm_stream(
    request: Request,
    text: str | None = None,
    voice: str = "ryan",
    instruct: str | None = None,
    chunk_size: int = 4,
    language: str = "English",
):
    """Stream raw 24 kHz mono int16 PCM at the codec-frame level.

    Frame format: ``[4 bytes BE uint32 length][length bytes int16 PCM]``.
    A zero-length frame marks end of stream.  Requires ``faster-qwen3-tts``.
    """
    if not model_ready:
        raise HTTPException(503, "Model loading")
    if not _FASTER_TTS:
        raise HTTPException(501, "Token-level streaming requires faster-qwen3-tts")
    if not text:
        raise HTTPException(400, "text required")

    logger.info(
        "TTS PCM stream: voice=%s chunk_size=%d len=%d instruct=%r",
        voice, chunk_size, len(text), instruct,
    )

    async def generate_pcm() -> AsyncGenerator[bytes, None]:
        start = time.time()
        kwargs = {
            "text": text,
            "speaker": voice,
            "language": language or "English",
            "chunk_size": chunk_size,
        }
        if instruct:
            kwargs["instruct"] = instruct

        chunk_num = 0
        for audio_chunk, sr, _timing in model.generate_custom_voice_streaming(**kwargs):
            if await request.is_disconnected():
                logger.info("TTS PCM stream: client disconnected at chunk %d", chunk_num)
                return
            if audio_chunk.dtype != np.int16:
                audio_chunk = (audio_chunk * 32767).clip(-32768, 32767).astype(np.int16)
            pcm_bytes = audio_chunk.tobytes()
            chunk_num += 1
            if chunk_num <= 2:
                logger.info(
                    "TTS PCM chunk %d: %d samples, %.3fs elapsed",
                    chunk_num, len(audio_chunk), time.time() - start,
                )
            yield struct.pack(">I", len(pcm_bytes)) + pcm_bytes

        yield struct.pack(">I", 0)
        logger.info("TTS PCM stream complete: %d chunks in %.2fs", chunk_num, time.time() - start)

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
# Info endpoints
# -----------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model_ready else "loading",
        "model_ready": model_ready,
        "model_id": MODEL_ID,
        "backend": "faster-qwen3-tts" if _FASTER_TTS else "qwen_tts",
        "device": _DEVICE,
    }


@app.get("/voices")
async def voices():
    return {"speakers": model.get_supported_speakers() if model_ready else []}


@app.get("/v1/models")
async def list_models():
    if not model_ready:
        raise HTTPException(503, "Model loading")
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID, "object": "model", "created": int(time.time()),
            "owned_by": "qwen_tts", "root": MODEL_ID, "parent": None,
            "max_model_len": 4096,
        }],
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    if not model_ready:
        raise HTTPException(503, "Model loading")
    if model_id != MODEL_ID:
        raise HTTPException(404, f"Model {model_id} not found")
    return {
        "id": MODEL_ID, "object": "model", "created": int(time.time()),
        "owned_by": "qwen_tts", "root": MODEL_ID, "parent": None,
        "max_model_len": 4096,
    }


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-TTS CustomVoice server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run on CPU instead of CUDA. WARNING: orders of magnitude "
             "slower; intended for smoke tests / dev environments without a GPU.",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    global _DEVICE
    if args.cpu or not torch.cuda.is_available():
        if not args.cpu:
            logger.warning("No CUDA device detected; falling back to CPU.")
        else:
            logger.warning("--cpu set; running on CPU. Generation will be SLOW.")
        _DEVICE = "cpu"
    else:
        _DEVICE = "cuda"

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
