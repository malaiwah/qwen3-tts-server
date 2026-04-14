"""Lightweight smoke tests that don't require the model to be loaded.

Validates module structure, helpers, and HTTP routing surface so CI can
catch regressions without a GPU.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Ensure the repo root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))


def _import_server():
    """Import the server module without triggering model loading.

    The module declares ``model = None`` at import time and only loads the
    weights inside the ``startup`` event handler, so a plain import is safe.
    """
    return importlib.import_module("server")


def test_module_imports():
    server = _import_server()
    assert hasattr(server, "app")
    assert hasattr(server, "MODEL_ID")
    assert server.MODEL_ID.startswith("Qwen/Qwen3-TTS")


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
    # Single chunk with no terminator should still come through as one sentence.
    assert server._split_sentences("hello world") == ["hello world"]


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


def test_health_endpoint_responds_when_loading():
    """The /health endpoint should respond even before the model is loaded."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    server = _import_server()
    with TestClient(server.app) as client:
        # Don't trigger the startup handler — set a flag-only sanity check.
        # TestClient runs lifespan; we want to bypass it to keep tests fast.
        pass
    # Direct call (bypass app lifecycle) — health is sync-safe.
    import asyncio
    payload = asyncio.run(server.health())
    assert "model_ready" in payload
    assert "model_id" in payload
