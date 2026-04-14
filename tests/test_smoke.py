"""Lightweight smoke tests — no GPU, no model load required.

Validates module structure, helpers, API routing, and request parsing
so CI can catch regressions without any CUDA hardware.
"""
from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest  # noqa: F401 — kept for pytest.raises / pytest.importorskip

# Ensure the repo root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))


def _import_server():
    """Import server module without triggering model loading.

    lifespan/startup only runs when the ASGI app receives its first lifespan
    event, which TestClient / direct import do not trigger.
    """
    return importlib.import_module("server")


# -----------------------------------------------------------------------
# Module-level smoke
# -----------------------------------------------------------------------

def test_module_imports():
    server = _import_server()
    assert hasattr(server, "app")
    assert hasattr(server, "MODEL_ID")
    assert server.MODEL_ID.startswith("Qwen/Qwen3-TTS")


# -----------------------------------------------------------------------
# Sentence splitter
# -----------------------------------------------------------------------

def test_split_sentences_basic():
    server = _import_server()
    out = server._split_sentences("Hello world. This is a test! Right?")
    assert out == ["Hello world.", "This is a test!", "Right?"]


def test_split_sentences_strips_whitespace():
    server = _import_server()
    out = server._split_sentences("   One.    Two.    ")
    assert out == ["One.", "Two."]


def test_split_sentences_empty():
    server = _import_server()
    assert server._split_sentences("") == []
    assert server._split_sentences("   ") == []


def test_split_sentences_no_terminator():
    server = _import_server()
    # Single chunk with no sentence terminator should still pass through.
    assert server._split_sentences("hello world") == ["hello world"]


# -----------------------------------------------------------------------
# Voice alias mapping
# -----------------------------------------------------------------------

def test_resolve_voice_openai_aliases():
    server = _import_server()
    assert server._resolve_voice("alloy") == "ryan"
    assert server._resolve_voice("echo") == "aiden"
    assert server._resolve_voice("fable") == "dylan"
    assert server._resolve_voice("onyx") == "eric"
    assert server._resolve_voice("nova") == "serena"
    assert server._resolve_voice("shimmer") == "vivian"


def test_resolve_voice_passthrough():
    server = _import_server()
    assert server._resolve_voice("ryan") == "ryan"
    assert server._resolve_voice("sohee") == "sohee"
    assert server._resolve_voice("uncle_fu") == "uncle_fu"


# -----------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------

def test_validate_text_empty_raises():
    from fastapi import HTTPException
    server = _import_server()
    with pytest.raises(HTTPException) as exc_info:
        server._validate_text("")
    assert exc_info.value.status_code == 400


def test_validate_text_none_raises():
    from fastapi import HTTPException
    server = _import_server()
    with pytest.raises(HTTPException) as exc_info:
        server._validate_text(None)
    assert exc_info.value.status_code == 400


def test_validate_text_too_long_raises():
    from fastapi import HTTPException
    server = _import_server()
    with pytest.raises(HTTPException) as exc_info:
        server._validate_text("x" * (server.MAX_TEXT_LENGTH + 1))
    assert exc_info.value.status_code == 413


def test_validate_text_ok():
    server = _import_server()
    assert server._validate_text("Hello") == "Hello"
    assert server._validate_text("  Hello  ") == "Hello"  # strips whitespace


# -----------------------------------------------------------------------
# Route registration
# -----------------------------------------------------------------------

def test_routes_registered():
    server = _import_server()
    paths = {route.path for route in server.app.routes}
    expected = {
        "/v1/audio/speech",
        "/v1/audio/speech/stream",
        "/v1/audio/speech/pcm-stream",
        "/v1/models",
        "/v1/models/{model_id}",
        "/health",
        "/voices",
    }
    missing = expected - paths
    assert not missing, f"Missing routes: {missing}"


# -----------------------------------------------------------------------
# Health endpoint (no model loaded)
# -----------------------------------------------------------------------

def test_health_payload_when_loading():
    server = _import_server()
    payload = asyncio.run(server.health())
    assert "model_ready" in payload
    assert "model_id" in payload
    assert "backend" in payload
    assert "device" in payload
    assert payload["model_ready"] is False
    assert payload["status"] == "loading"


# -----------------------------------------------------------------------
# JSON body parsing
# -----------------------------------------------------------------------

class _FakeRequest:
    """Minimal Request stand-in for _parse_params testing."""

    def __init__(self, json_body: dict | None = None):
        self._json = json_body
        self.headers = {
            "content-type": "application/json" if json_body is not None else "multipart/form-data"
        }

    async def json(self):
        return self._json or {}


def test_parse_params_json_body_input_field():
    """OpenAI SDK sends 'input' in JSON body; should map to 'text'."""
    async def _run():
        server = _import_server()
        req = _FakeRequest({"input": "Hello world", "voice": "alloy"})
        p = await server._parse_params(req, text=None, voice="ryan", response_format="mp3",
                                        instruct=None, language="English", speed=1.0)
        assert p["text"] == "Hello world"
        assert p["voice"] == "alloy"
    asyncio.run(_run())


def test_parse_params_json_body_text_alias():
    """Our 'text' alias in JSON body should also work."""
    async def _run():
        server = _import_server()
        req = _FakeRequest({"text": "Bonjour", "language": "French"})
        p = await server._parse_params(req, text=None, voice="ryan", response_format="mp3",
                                        instruct=None, language="English", speed=1.0)
        assert p["text"] == "Bonjour"
        assert p["language"] == "French"
    asyncio.run(_run())


def test_parse_params_query_fallback():
    """Query params should be used when no JSON body is present."""
    async def _run():
        server = _import_server()
        req = _FakeRequest(None)  # no JSON body
        p = await server._parse_params(req, text="Query text", voice="eric",
                                        response_format="wav", instruct=None,
                                        language="English", speed=1.0)
        assert p["text"] == "Query text"
        assert p["voice"] == "eric"
        assert p["response_format"] == "wav"
    asyncio.run(_run())


# -----------------------------------------------------------------------
# ffmpeg pipe transcoder (mocked — no ffmpeg binary needed in CI)
# -----------------------------------------------------------------------

def test_transcode_wav_mp3_calls_ffmpeg():
    server = _import_server()
    fake_wav = b"RIFF....WAVEfmt "  # minimal fake WAV header
    fake_mp3 = b"\xff\xfb\x90\x00"  # fake MP3 header

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = fake_mp3
        mock_run.return_value.returncode = 0
        result = server._transcode_wav(fake_wav, "mp3")

    assert result == fake_mp3
    cmd = mock_run.call_args[0][0]
    assert "pipe:0" in cmd   # stdin input
    assert "pipe:1" in cmd   # stdout output
    assert "libmp3lame" in cmd


def test_transcode_wav_opus_calls_ffmpeg():
    server = _import_server()
    fake_wav = b"RIFF....WAVEfmt "
    fake_ogg = b"OggS"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = fake_ogg
        mock_run.return_value.returncode = 0
        result = server._transcode_wav(fake_wav, "opus")

    assert result == fake_ogg
    cmd = mock_run.call_args[0][0]
    assert "libopus" in cmd


# -----------------------------------------------------------------------
# Metrics endpoint and auth middleware (in-process via TestClient)
# -----------------------------------------------------------------------

def test_metrics_endpoint_exposed_without_auth():
    from fastapi.testclient import TestClient
    server = _import_server()
    original = server._API_KEY
    server._API_KEY = ""
    try:
        client = TestClient(server.app)
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.text
        assert "qwen3_tts_requests_total" in body or "qwen3_tts_request_duration_seconds" in body
        assert "qwen3_tts_model_ready" in body
    finally:
        server._API_KEY = original


def test_auth_blocks_v1_when_key_set():
    from fastapi.testclient import TestClient
    server = _import_server()
    original = server._API_KEY
    server._API_KEY = "secret-key"
    try:
        client = TestClient(server.app)
        # /v1/models requires model_ready; here we only check auth — any
        # non-200 is fine as long as unauthenticated hits 401 first.
        r = client.get("/v1/models")
        assert r.status_code == 401
        r = client.get("/v1/models", headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401
        # Health and metrics must remain unauthenticated
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200
    finally:
        server._API_KEY = original


def test_auth_disabled_when_no_key():
    from fastapi.testclient import TestClient
    server = _import_server()
    original = server._API_KEY
    server._API_KEY = ""
    try:
        client = TestClient(server.app)
        # With auth off, unauth requests hit the endpoint logic (which may
        # return 503 "model loading" — that's fine, we're only checking that
        # the middleware doesn't block with 401).
        r = client.get("/health")
        assert r.status_code == 200
    finally:
        server._API_KEY = original
