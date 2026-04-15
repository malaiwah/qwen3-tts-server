"""Lightweight smoke tests — no GPU, no model load required.

Validates module structure, helpers, API routing, and request parsing
so CI can catch regressions without any CUDA hardware.

The voice registry is loaded at session start from the committed preset
bundle.  No network and no qwen_tts are needed — missing qwen_tts is
tolerated and produces voice records with ``prompt_item=None`` (sufficient
for membership and routing tests).
"""
from __future__ import annotations

import asyncio
import importlib
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the repo root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))


def _import_server():
    """Import server module without triggering model loading.

    lifespan/startup only runs when the ASGI app receives its first lifespan
    event, which TestClient / direct import do not trigger.
    """
    return importlib.import_module("server")


@pytest.fixture(scope="session", autouse=True)
def _load_registry():
    """Populate the registry from the committed preset bundle once per run.

    Runs for every test transparently (autouse) so voice-resolution tests
    don't each need to remember to load it.  No model is bound — preset
    records still carry valid IDs, their ``prompt_item`` is ``None`` when
    qwen_tts is unavailable (fine for smoke tests).
    """
    server = _import_server()
    server._voice_registry.load()
    yield


# -----------------------------------------------------------------------
# Module-level smoke
# -----------------------------------------------------------------------

def test_module_imports():
    server = _import_server()
    assert hasattr(server, "app")
    assert hasattr(server, "MODEL_ID")
    assert server.MODEL_ID.startswith("Qwen/Qwen3-TTS")


def test_default_model_is_base():
    """Voice cloning path requires -Base, not -CustomVoice."""
    server = _import_server()
    # Accept either the exact default or a user-set env override that still
    # points at a -Base-compatible checkpoint.
    assert "CustomVoice" not in server.MODEL_ID, (
        "Default model must not be -CustomVoice — voice cloning requires -Base."
    )


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
# Voice alias mapping + registry resolution
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


def test_resolve_voice_unknown_raises_400():
    from fastapi import HTTPException
    server = _import_server()
    with pytest.raises(HTTPException) as exc_info:
        server._resolve_voice("not_a_real_voice")
    assert exc_info.value.status_code == 400


def test_resolve_voice_case_insensitive():
    server = _import_server()
    # Uppercase alias should still resolve.
    assert server._resolve_voice("ALLOY") == "ryan"


def test_registry_has_nine_presets():
    server = _import_server()
    presets = server._voice_registry.list(kind="preset")
    ids = {p["id"] for p in presets}
    assert ids == {
        "aiden", "dylan", "eric", "ono_anna", "ryan",
        "serena", "sohee", "uncle_fu", "vivian",
    }


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
        "/v1/voices",
        "/v1/voices/{voice_id}",
        "/health",
        "/voices",  # legacy alias
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
    # Voice counts should surface even before the model is loaded.
    assert payload["voices"]["preset"] == 9
    assert payload["voices"]["custom"] >= 0


# -----------------------------------------------------------------------
# Voice listing endpoint
# -----------------------------------------------------------------------

def test_list_voices_returns_all_presets():
    from fastapi.testclient import TestClient
    server = _import_server()
    original = server._API_KEY
    server._API_KEY = ""
    try:
        client = TestClient(server.app)
        r = client.get("/v1/voices")
        assert r.status_code == 200
        body = r.json()
        assert "data" in body
        ids = {v["id"] for v in body["data"]}
        assert {"aiden", "ryan", "vivian"}.issubset(ids)
        kinds = {v["kind"] for v in body["data"]}
        assert "preset" in kinds
        assert "openai_voice_aliases" in body
    finally:
        server._API_KEY = original


def test_legacy_voices_endpoint_still_works():
    """Back-compat: GET /voices must still return a data-shaped payload."""
    from fastapi.testclient import TestClient
    server = _import_server()
    original = server._API_KEY
    server._API_KEY = ""
    try:
        client = TestClient(server.app)
        r = client.get("/voices")
        assert r.status_code == 200
        assert "data" in r.json()
    finally:
        server._API_KEY = original


def test_delete_preset_is_forbidden():
    """Presets must not be deletable via the public API."""
    from fastapi.testclient import TestClient
    server = _import_server()
    original = server._API_KEY
    server._API_KEY = ""
    try:
        client = TestClient(server.app)
        r = client.delete("/v1/voices/ryan")
        assert r.status_code == 403
    finally:
        server._API_KEY = original


def test_delete_unknown_voice_returns_404():
    from fastapi.testclient import TestClient
    server = _import_server()
    original = server._API_KEY
    server._API_KEY = ""
    try:
        client = TestClient(server.app)
        r = client.delete("/v1/voices/vc_doesnotexist")
        assert r.status_code == 404
    finally:
        server._API_KEY = original


# -----------------------------------------------------------------------
# Voice registration (custom clones) — model is mocked
# -----------------------------------------------------------------------

def test_register_voice_round_trip(tmp_path):
    """POST /v1/voices with a fake audio file registers a custom voice.

    The actual model call is mocked (we inject a fake model onto the
    registry) so no GPU or qwen_tts install is required.  We verify:
      - new voice appears in the registry with a vc_* ID
      - sidecar files land in the configured custom dir
      - GET /v1/voices lists it
      - DELETE removes it and the files
    """
    from fastapi.testclient import TestClient
    import torch

    server = _import_server()
    reg = server._voice_registry

    # Redirect the registry to a temp dir so we don't touch /data/voices.
    original_dir = reg.custom_dir
    reg.custom_dir = tmp_path
    original_api_key = server._API_KEY
    server._API_KEY = ""

    # Stub model that returns a fabricated VoiceClonePromptItem-like object.
    class _FakeItem:
        ref_spk_embedding = torch.zeros(2048, dtype=torch.float32)
        ref_code = None
        x_vector_only_mode = True
        icl_mode = False
        ref_text = None

    class _FakeModel:
        def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
            # Ensure the temp file was written as expected.
            assert Path(ref_audio).exists()
            return [_FakeItem()]

    reg.bind_model(_FakeModel())
    # Tests run without a model-loaded lifespan, so force ready=True for the
    # registration endpoint's 503 gate.
    original_ready = server._model_ready
    server._model_ready = True
    # The registration handler acquires the inference semaphore; make sure
    # it exists (normally created in lifespan).
    if not hasattr(server, "_INFER_SEM") or not isinstance(getattr(server, "_INFER_SEM", None), asyncio.Semaphore):
        server._INFER_SEM = asyncio.Semaphore(1)

    try:
        client = TestClient(server.app)
        # 1) Register
        audio_bytes = b"RIFF0000WAVEfmt "  # fake bytes — our fake model ignores them
        r = client.post(
            "/v1/voices",
            files={"audio": ("sample.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data={"name": "Alice", "language": "English"},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        voice_id = body["id"]
        assert voice_id.startswith("vc_")
        assert body["kind"] == "custom"
        assert body["name"] == "Alice"

        # Sidecar files on disk
        assert (tmp_path / f"{voice_id}.safetensors").is_file()
        sidecar = json.loads((tmp_path / f"{voice_id}.json").read_text())
        assert sidecar["name"] == "Alice"
        assert sidecar["id"] == voice_id

        # 2) List includes it
        r = client.get("/v1/voices")
        assert r.status_code == 200
        ids = {v["id"] for v in r.json()["data"]}
        assert voice_id in ids

        # 3) Registered voice is resolvable and usable as a voice parameter
        assert server._resolve_voice(voice_id) == voice_id

        # 4) Delete
        r = client.delete(f"/v1/voices/{voice_id}")
        assert r.status_code == 200
        assert r.json() == {"id": voice_id, "deleted": True}
        assert not (tmp_path / f"{voice_id}.safetensors").exists()
        assert not (tmp_path / f"{voice_id}.json").exists()

        # 5) Gone from listing
        r = client.get("/v1/voices")
        ids = {v["id"] for v in r.json()["data"]}
        assert voice_id not in ids
    finally:
        reg.custom_dir = original_dir
        server._API_KEY = original_api_key
        server._model_ready = original_ready
        reg.bind_model(None)


def test_register_voice_requires_model_ready():
    """When _model_ready is False, POST /v1/voices must return 503."""
    from fastapi.testclient import TestClient
    server = _import_server()
    original_ready = server._model_ready
    original_api_key = server._API_KEY
    server._model_ready = False
    server._API_KEY = ""
    try:
        client = TestClient(server.app)
        r = client.post(
            "/v1/voices",
            files={"audio": ("s.wav", io.BytesIO(b"RIFF"), "audio/wav")},
            data={"name": "X"},
        )
        assert r.status_code == 503
    finally:
        server._model_ready = original_ready
        server._API_KEY = original_api_key


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
        # /v1/voices also requires auth
        r = client.get("/v1/voices")
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
