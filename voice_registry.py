"""Voice registry — preset speakers + user-registered clones.

Two sources of voices feed the same code path on the -Base checkpoint:

  1. **Presets** — nine named speakers (aiden, dylan, …) lifted from the
     CustomVoice checkpoint's ``talker.model.codec_embedding.weight`` and
     shipped as a ~37 KB sidecar bundle at ``assets/preset/bundle.safetensors``.
     Loaded once at startup; immutable.

  2. **Custom clones** — registered via ``POST /v1/voices`` with a reference
     audio clip (and optional transcript).  Persisted as one
     ``<uuid>.safetensors`` + ``<uuid>.json`` pair under ``$QWEN3_TTS_VOICES_DIR``
     (default ``/data/voices/custom``).  Survive container restarts.

Both kinds end up as a :class:`VoiceClonePromptItem` in an in-memory dict
keyed by voice ID, consumed by the generation path.

Preset voice IDs are bare names (``"vivian"``).  Custom voice IDs are
``"vc_" + 8 hex chars`` so they're unambiguous.

Provenance of the preset bundle is documented in
``assets/preset/bundle.json`` and in the HF dataset at
https://huggingface.co/datasets/malaiwah/qwen3-tts-preset-voices.
"""

from __future__ import annotations

import json
import logging
import secrets
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from safetensors.torch import load_file, save_file


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------

# Kinds exposed in API responses and used for deletion gating.
KIND_PRESET = "preset"
KIND_CUSTOM = "custom"


@dataclass
class VoiceRecord:
    """In-memory record for one voice (preset or custom).

    Two synthesis paths are supported depending on the active backend:

    **Tier 2/3 (Qwen3TTSModel)** — ``prompt_item`` holds a pre-extracted
    ``VoiceClonePromptItem`` (speaker embedding + optional prosody tokens).
    Built at registration time via ``Qwen3TTSModel.create_voice_clone_prompt()``.

    **Tier 1 (FasterQwen3TTS)** — ``prompt_item`` is ``None``; instead
    ``ref_audio_path`` points to the persisted reference audio file and
    ``ref_text_str`` holds the transcript.  Both are passed directly to
    ``FasterQwen3TTS.generate_voice_clone(ref_audio=…, ref_text=…)`` at
    synthesis time, which resolves the speaker embedding internally.

    Preset voices always use the Tier 2/3 path (pre-extracted embeddings
    loaded from the safetensors bundle) regardless of the active backend.
    """
    id: str
    kind: str                 # "preset" | "custom"
    name: str                 # Human-friendly label (may equal id for presets)
    prompt_item: Any          # qwen_tts.VoiceClonePromptItem, or None (Tier 1 custom)
    created_at: Optional[float] = None
    meta: Optional[dict] = None   # Freeform extra info for /v1/voices responses
    # Tier-1 path (FasterQwen3TTS): reference audio kept on disk, resolved per-call.
    ref_audio_path: Optional[Path] = None
    ref_text_str: Optional[str] = None

    def summary(self) -> dict:
        """Serializable summary for /v1/voices listing."""
        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
        }
        if self.created_at is not None:
            out["created_at"] = int(self.created_at)
        if self.meta:
            out.update(self.meta)
        if self.ref_audio_path is not None:
            # Signal to callers that this is a Tier-1 (ref-audio) clone.
            out["clone_mode"] = "ref_audio"
        return out


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------

class VoiceRegistry:
    """Thread-safe registry of voices.

    Construct once per process and call :meth:`load` during app startup.
    The model is injected lazily (via :meth:`bind_model`) because the
    registry is constructed before model weights are loaded.
    """

    def __init__(
        self,
        preset_bundle_path: Path,
        preset_meta_path: Path,
        custom_dir: Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.preset_bundle_path = preset_bundle_path
        self.preset_meta_path = preset_meta_path
        self.custom_dir = custom_dir
        self.device = device
        self.dtype = dtype
        self._voices: dict[str, VoiceRecord] = {}
        self._lock = threading.RLock()
        self._model = None  # bound later

    # -- lifecycle ------------------------------------------------------

    def bind_model(self, model: Any) -> None:
        """Attach the loaded TTS model (needed for ``register()``)."""
        self._model = model

    def load(self) -> None:
        """Load presets from the bundle + scan ``custom_dir`` for user clones.

        Safe to call once at startup.  Logs — doesn't raise — on individual
        custom-voice load failures so one corrupt sidecar doesn't kill the
        whole server.

        If the configured device is ``cuda`` but CUDA is unavailable at load
        time (e.g. CPU-only test environments), falls back to CPU so the
        registry still populates — generation would be slow but the listing
        and routing surface stays functional.
        """
        with self._lock:
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.info("CUDA unavailable — loading voice registry on CPU.")
                self.device = "cpu"
            self._load_presets()
            self._load_custom()

    # -- preset loading -------------------------------------------------

    def _load_presets(self) -> None:
        if not self.preset_bundle_path.is_file():
            logger.warning(
                "Preset bundle not found at %s — preset voices disabled. "
                "Expected a 9-row safetensors lifted from the CustomVoice checkpoint.",
                self.preset_bundle_path,
            )
            return

        rows = load_file(str(self.preset_bundle_path))
        meta: dict = {}
        if self.preset_meta_path.is_file():
            try:
                meta = json.loads(self.preset_meta_path.read_text())
            except Exception as exc:  # noqa: BLE001
                logger.warning("preset bundle.json unreadable: %s", exc)

        speakers_meta = (meta.get("speakers") or {}) if meta else {}
        count = 0
        for name, row in rows.items():
            # Bundle was saved on CPU; move to target device/dtype lazily.
            embedding = row.to(device=self.device, dtype=self.dtype).contiguous()
            prompt_item = _build_prompt_item(
                ref_code=None,
                ref_spk_embedding=embedding,
                x_vector_only_mode=True,
                icl_mode=False,
                ref_text=None,
            )
            spk_meta = speakers_meta.get(name, {})
            self._voices[name] = VoiceRecord(
                id=name,
                kind=KIND_PRESET,
                name=name,
                prompt_item=prompt_item,
                meta={k: v for k, v in spk_meta.items() if k in ("gender", "lang")},
            )
            count += 1

        logger.info("Loaded %d preset voices from %s", count, self.preset_bundle_path.name)

    # -- custom loading -------------------------------------------------

    def _load_custom(self) -> None:
        if not self.custom_dir.is_dir():
            logger.info("Custom voices dir %s does not exist — skipping.", self.custom_dir)
            return

        loaded = 0
        failed = 0
        for sidecar in sorted(self.custom_dir.glob("*.json")):
            voice_id = sidecar.stem
            try:
                self._load_one_custom(voice_id)
                loaded += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.error("Failed to load custom voice %s: %s", voice_id, exc)

        if loaded or failed:
            logger.info("Loaded %d custom voices from %s (failed: %d)",
                        loaded, self.custom_dir, failed)

    def _load_one_custom(self, voice_id: str) -> VoiceRecord:
        meta_path = self.custom_dir / f"{voice_id}.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"missing sidecar JSON for {voice_id}")

        meta = json.loads(meta_path.read_text())

        # ----------------------------------------------------------------
        # Tier-1 (FasterQwen3TTS) clone — sidecar has a ref_audio file
        # instead of pre-extracted safetensor embeddings.
        # ----------------------------------------------------------------
        if meta.get("clone_mode") == "ref_audio":
            # Find the reference audio file (any supported suffix).
            ref_audio_path: Optional[Path] = None
            for suf in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"):
                candidate = self.custom_dir / f"{voice_id}{suf}"
                if candidate.is_file():
                    ref_audio_path = candidate
                    break
            if ref_audio_path is None:
                raise FileNotFoundError(
                    f"ref_audio file not found for Tier-1 clone {voice_id}"
                )
            record = VoiceRecord(
                id=voice_id,
                kind=KIND_CUSTOM,
                name=meta.get("name") or voice_id,
                prompt_item=None,
                created_at=meta.get("created_at"),
                meta={k: meta[k] for k in ("language_hint", "ref_text", "clone_mode") if k in meta},
                ref_audio_path=ref_audio_path,
                ref_text_str=meta.get("ref_text") or "",
            )
            self._voices[voice_id] = record
            return record

        # ----------------------------------------------------------------
        # Tier-2/3 (Qwen3TTSModel) clone — safetensors sidecar.
        # ----------------------------------------------------------------
        tensors_path = self.custom_dir / f"{voice_id}.safetensors"
        if not tensors_path.is_file():
            raise FileNotFoundError(f"missing safetensors sidecar for {voice_id}")

        tensors = load_file(str(tensors_path))
        ref_spk_embedding = tensors.get("ref_spk_embedding")
        if ref_spk_embedding is None:
            raise ValueError("ref_spk_embedding tensor missing in safetensors")
        ref_code = tensors.get("ref_code")  # may be absent for x_vector_only clones

        prompt_item = _build_prompt_item(
            ref_code=ref_code.to(self.device) if ref_code is not None else None,
            ref_spk_embedding=ref_spk_embedding.to(device=self.device, dtype=self.dtype).contiguous(),
            x_vector_only_mode=bool(meta.get("x_vector_only_mode", ref_code is None)),
            icl_mode=bool(meta.get("icl_mode", ref_code is not None)),
            ref_text=meta.get("ref_text"),
        )

        record = VoiceRecord(
            id=voice_id,
            kind=KIND_CUSTOM,
            name=meta.get("name") or voice_id,
            prompt_item=prompt_item,
            created_at=meta.get("created_at"),
            meta={
                k: meta[k]
                for k in ("language_hint", "ref_text", "x_vector_only_mode", "icl_mode")
                if k in meta
            },
        )
        self._voices[voice_id] = record
        return record

    # -- lookup / listing ----------------------------------------------

    def get(self, voice_id: str) -> Optional[VoiceRecord]:
        with self._lock:
            return self._voices.get(voice_id)

    def list(self, *, kind: Optional[str] = None) -> list[dict]:
        with self._lock:
            values: Iterable[VoiceRecord] = self._voices.values()
            if kind is not None:
                values = [v for v in values if v.kind == kind]
            else:
                values = list(values)
        return [v.summary() for v in values]

    def __contains__(self, voice_id: str) -> bool:
        with self._lock:
            return voice_id in self._voices

    # -- registration / deletion ---------------------------------------

    def register(
        self,
        *,
        name: str,
        ref_audio_bytes: bytes,
        ref_audio_suffix: str = ".wav",
        ref_text: Optional[str] = None,
        language: Optional[str] = None,
    ) -> VoiceRecord:
        """Create a new custom voice from a reference audio clip.

        Two registration paths depending on the active backend:

        **Tier 2/3 (Qwen3TTSModel)** — calls ``create_voice_clone_prompt()`` to
        pre-extract the speaker embedding and optional prosody tokens, then
        persists them as a safetensors sidecar.  If ``ref_text`` is provided
        (or auto-transcribed), uses full ICL mode; otherwise x-vector-only.

        **Tier 1 (FasterQwen3TTS)** — the model has no
        ``create_voice_clone_prompt`` method; instead it accepts ``ref_audio``
        + ``ref_text`` directly at generation time via ``generate_voice_clone``.
        The reference audio is saved verbatim in the custom voices directory and
        re-used on every synthesis call.  Quality is equivalent to Tier 2/3 ICL
        mode; the trade-off is a small per-call overhead for embedding
        extraction (vs. a one-time cost at registration).
        """
        if self._model is None:
            raise RuntimeError("VoiceRegistry.register called before model bound")
        if not name or not name.strip():
            raise ValueError("name is required")

        self.custom_dir.mkdir(parents=True, exist_ok=True)
        voice_id = self._next_voice_id()

        # ----------------------------------------------------------------
        # Tier-1 path (FasterQwen3TTS): no create_voice_clone_prompt API.
        # Save the reference audio on disk as WAV (soundfile-compatible) so
        # FasterQwen3TTS can load it via sf.read() at generation time.
        # M4A/AAC and other container formats are transcoded to WAV via ffmpeg.
        # ----------------------------------------------------------------
        if not hasattr(self._model, "create_voice_clone_prompt"):
            # soundfile supports WAV, FLAC, OGG, AIFF but not M4A/AAC/MP4.
            _SOUNDFILE_NATIVE = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
            if ref_audio_suffix.lower() in _SOUNDFILE_NATIVE:
                audio_file = self.custom_dir / f"{voice_id}{ref_audio_suffix}"
                audio_file.write_bytes(ref_audio_bytes)
            else:
                # Transcode to 16-bit 24 kHz mono WAV via ffmpeg (matches TTS sample rate).
                audio_file = self.custom_dir / f"{voice_id}.wav"
                with tempfile.NamedTemporaryFile(
                    suffix=ref_audio_suffix, dir=str(self.custom_dir), delete=False,
                ) as tmp:
                    tmp.write(ref_audio_bytes)
                    tmp_path = tmp.name
                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-loglevel", "error",
                            "-i", tmp_path,
                            "-ar", "24000", "-ac", "1", "-sample_fmt", "s16",
                            str(audio_file),
                        ],
                        check=True, timeout=60,
                    )
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"ffmpeg transcoding of {ref_audio_suffix} to WAV failed: {exc}"
                    ) from exc
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                logger.info(
                    "Transcoded %s → WAV (%d KB → %d KB) for Tier-1 clone %s",
                    ref_audio_suffix, len(ref_audio_bytes) // 1024,
                    audio_file.stat().st_size // 1024, voice_id,
                )
            meta = {
                "id": voice_id,
                "name": name.strip(),
                "created_at": int(time.time()),
                "ref_text": ref_text.strip() if ref_text and ref_text.strip() else None,
                "language_hint": language,
                "clone_mode": "ref_audio",
                "source": "faster-qwen3-tts (ref_audio path)",
            }
            (self.custom_dir / f"{voice_id}.json").write_text(json.dumps(meta, indent=2))
            with self._lock:
                record = VoiceRecord(
                    id=voice_id,
                    kind=KIND_CUSTOM,
                    name=meta["name"],
                    prompt_item=None,
                    created_at=meta["created_at"],
                    meta={"language_hint": language, "ref_text": meta["ref_text"],
                          "clone_mode": "ref_audio"},
                    ref_audio_path=audio_file,
                    ref_text_str=meta["ref_text"] or "",
                )
                self._voices[voice_id] = record
            logger.info("Registered custom voice %s (%s) — Tier-1 ref_audio path (%d bytes)",
                        voice_id, name, len(ref_audio_bytes))
            return record

        # ----------------------------------------------------------------
        # Tier-2/3 path (Qwen3TTSModel): pre-extract speaker embedding.
        # ----------------------------------------------------------------
        x_vector_only = ref_text is None or not ref_text.strip()

        # The model API takes a file path.  Write to a temp file so we can
        # decode unusual formats via soundfile/ffmpeg.
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=ref_audio_suffix, dir=str(self.custom_dir), delete=False,
        ) as tmp:
            tmp.write(ref_audio_bytes)
            tmp_path = tmp.name

        try:
            items = self._model.create_voice_clone_prompt(
                ref_audio=tmp_path,
                ref_text=ref_text or "",
                x_vector_only_mode=x_vector_only,
            )
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

        if not items:
            raise RuntimeError("create_voice_clone_prompt returned no items")
        item = items[0]

        # Persist as safetensors + json sidecars.
        tensors: dict[str, torch.Tensor] = {}
        spk_emb = getattr(item, "ref_spk_embedding", None)
        if spk_emb is None:
            raise RuntimeError("clone prompt missing ref_spk_embedding")
        tensors["ref_spk_embedding"] = spk_emb.detach().cpu().contiguous()
        ref_code = getattr(item, "ref_code", None)
        if ref_code is not None:
            tensors["ref_code"] = ref_code.detach().cpu().contiguous()

        meta = {
            "id": voice_id,
            "name": name.strip(),
            "created_at": int(time.time()),
            "ref_text": ref_text if not x_vector_only else None,
            "language_hint": language,
            "x_vector_only_mode": bool(getattr(item, "x_vector_only_mode", x_vector_only)),
            "icl_mode": bool(getattr(item, "icl_mode", not x_vector_only)),
            "source": "create_voice_clone_prompt",
        }

        save_file(tensors, str(self.custom_dir / f"{voice_id}.safetensors"))
        (self.custom_dir / f"{voice_id}.json").write_text(json.dumps(meta, indent=2))

        # Insert the just-loaded item into the registry (re-use the one we
        # just produced rather than re-reading from disk — saves a round-trip
        # and keeps the tensors on the GPU).
        with self._lock:
            record = VoiceRecord(
                id=voice_id,
                kind=KIND_CUSTOM,
                name=meta["name"],
                prompt_item=item,
                created_at=meta["created_at"],
                meta={
                    "language_hint": language,
                    "ref_text": meta["ref_text"],
                    "x_vector_only_mode": meta["x_vector_only_mode"],
                    "icl_mode": meta["icl_mode"],
                },
            )
            self._voices[voice_id] = record
        logger.info("Registered custom voice %s (%s) — Tier-2/3 x_vector_only=%s",
                    voice_id, name, x_vector_only)
        return record

    def delete(self, voice_id: str) -> bool:
        """Delete a custom voice.  Presets are immutable — returns False."""
        with self._lock:
            record = self._voices.get(voice_id)
            if record is None:
                return False
            if record.kind == KIND_PRESET:
                raise PermissionError("preset voices cannot be deleted")
            self._voices.pop(voice_id, None)

        # Remove files on disk best-effort — lock is already released so we
        # don't hold it during I/O.
        for suffix in (".safetensors", ".json"):
            try:
                (self.custom_dir / f"{voice_id}{suffix}").unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to remove %s%s: %s", voice_id, suffix, exc)
        logger.info("Deleted custom voice %s", voice_id)
        return True

    # -- internals ------------------------------------------------------

    def _next_voice_id(self) -> str:
        # 8 hex chars of entropy → 2^32 keys; collision chance at 10k voices is
        # ~1 in 400k.  Retry on the vanishingly rare clash.
        for _ in range(8):
            candidate = "vc_" + secrets.token_hex(4)
            if candidate not in self._voices:
                return candidate
        raise RuntimeError("could not allocate unique voice_id after 8 tries")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

_QWEN_TTS_MISSING_WARNED = False


def _build_prompt_item(
    *, ref_code, ref_spk_embedding, x_vector_only_mode, icl_mode, ref_text,
):
    """Construct a ``qwen_tts.VoiceClonePromptItem`` when the library is present.

    The registry is imported by the FastAPI module at startup — including in
    test environments that only have CPU-only torch and no ``qwen_tts``
    installed.  When qwen_tts is unavailable we register the voice under its
    ID (so ``/v1/voices`` listings and voice-resolution work in tests) but
    set ``prompt_item=None``.  Generation endpoints then fail loudly at call
    time, which is the correct behaviour for a misconfigured deploy.
    """
    global _QWEN_TTS_MISSING_WARNED
    try:
        from qwen_tts import VoiceClonePromptItem  # type: ignore
    except ImportError:
        if not _QWEN_TTS_MISSING_WARNED:
            logger.warning(
                "qwen_tts not installed — registry will list voice IDs but "
                "prompt_items are unavailable. Generation endpoints will fail "
                "at request time.  Install qwen_tts to enable synthesis."
            )
            _QWEN_TTS_MISSING_WARNED = True
        return None
    return VoiceClonePromptItem(
        ref_code=ref_code,
        ref_spk_embedding=ref_spk_embedding,
        x_vector_only_mode=x_vector_only_mode,
        icl_mode=icl_mode,
        ref_text=ref_text,
    )
