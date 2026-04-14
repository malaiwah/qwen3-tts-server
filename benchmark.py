#!/usr/bin/env python3
"""Benchmark a running qwen3-tts-server.

Measures steady-state wall-clock latency and real-time factor (RTF) across
short, medium, and long inputs.  First run is treated as warmup and excluded.

Usage:
    ./benchmark.py
    ./benchmark.py --url http://my-gpu-host:8001 --runs 5
    ./benchmark.py --voice aiden --language French --runs 3

The script reads the ``X-Gen-Time-Ms`` header for server-side generation time
and computes wall-clock RTF from the audio duration reported by soundfile.
"""
from __future__ import annotations

import argparse
import io
import os
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import urllib.error
import urllib.request


# Default workloads (short / medium / long speech) — kept English for consistency
# across languages.  Override with --text to benchmark your own content.
_WORKLOADS: dict[str, str] = {
    "short":  "The quick brown fox jumps over the lazy dog.",
    "medium": (
        "Text-to-speech synthesis converts written language into spoken audio. "
        "Modern neural systems generate natural, expressive speech at rates "
        "faster than real time. This sentence is about nine seconds long when "
        "spoken at a moderate pace."
    ),
    "long": (
        "The history of speech synthesis stretches back more than two hundred years. "
        "Wolfgang von Kempelen built a mechanical speaking machine in 1791, using "
        "bellows to force air through a reed and hand-articulated resonators shaped "
        "like a human vocal tract. Electronic synthesis began in the 1930s with "
        "Homer Dudley's Voder at Bell Labs. Formant synthesis dominated the late "
        "twentieth century before concatenative approaches and, more recently, deep "
        "neural networks produced the natural-sounding voices we hear today."
    ),
}


def _audio_duration_seconds(data: bytes, fmt: str) -> float:
    """Return audio duration in seconds.  Uses soundfile if available, falls
    back to a ffprobe subprocess, and finally raises if neither is present."""
    try:
        import soundfile as sf
        with sf.SoundFile(io.BytesIO(data)) as f:
            return len(f) / f.samplerate
    except Exception:
        pass
    # Fallback: ffprobe
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}") as tmp:
        tmp.write(data)
        tmp.flush()
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", tmp.name,
        ], text=True)
    return float(out.strip())


def _run_one(url: str, params: dict, api_key: str | None) -> tuple[float, float, int, int]:
    """Run one request.  Returns (wall_seconds, audio_seconds, gen_ms, bytes)."""
    full_url = f"{url}?{urlencode(params)}"
    req = urllib.request.Request(full_url, method="POST")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
        gen_ms = int(resp.headers.get("X-Gen-Time-Ms", "0") or 0)
    wall = time.perf_counter() - t0
    audio = _audio_duration_seconds(data, params["response_format"])
    return wall, audio, gen_ms, len(data)


def _format_row(label: str, runs: list[tuple[float, float, int, int]]) -> str:
    walls = [r[0] for r in runs]
    audios = [r[1] for r in runs]
    gens = [r[2] for r in runs]
    avg_wall = statistics.mean(walls)
    avg_audio = statistics.mean(audios)
    avg_gen = statistics.mean(gens) / 1000.0 if gens[0] else avg_wall
    rtf = avg_audio / avg_wall if avg_wall > 0 else 0.0
    return (
        f"  {label:<7}  audio={avg_audio:6.2f}s  "
        f"wall={avg_wall*1000:7.1f}ms  gen={avg_gen*1000:7.1f}ms  "
        f"RTF={rtf:5.2f}x   (n={len(runs)})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8001", help="Base URL of the TTS server.")
    parser.add_argument("--voice", default="ryan", help="Speaker voice (default: ryan).")
    parser.add_argument("--language", default="English")
    parser.add_argument("--format", default="wav", choices=["mp3", "wav", "opus"],
                        help="Response format (default: wav — avoids mp3 encode time skewing the server-side budget).")
    parser.add_argument("--runs", type=int, default=5, help="Runs per workload (first is warmup, excluded).")
    parser.add_argument("--workloads", default="short,medium,long",
                        help="Comma-separated workload names. Default: short,medium,long.")
    parser.add_argument("--text", default=None, help="Custom text (overrides --workloads; runs only 'custom').")
    parser.add_argument("--api-key", default=os.getenv("QWEN_API_KEY") or None,
                        help="Bearer token (reads QWEN_API_KEY env if unset).")
    args = parser.parse_args()

    base = args.url.rstrip("/") + "/v1/audio/speech"
    workloads: dict[str, str]
    if args.text:
        workloads = {"custom": args.text}
    else:
        selected = [w.strip() for w in args.workloads.split(",") if w.strip()]
        unknown = [w for w in selected if w not in _WORKLOADS]
        if unknown:
            print(f"Unknown workloads: {unknown}.  Available: {list(_WORKLOADS)}", file=sys.stderr)
            return 2
        workloads = {w: _WORKLOADS[w] for w in selected}

    print(f"qwen3-tts-server benchmark — {args.url}  voice={args.voice}  lang={args.language}  fmt={args.format}  runs={args.runs}\n")

    for name, text in workloads.items():
        print(f"[{name}]  len={len(text)} chars")
        params = {
            "text": text,
            "voice": args.voice,
            "response_format": args.format,
            "language": args.language,
        }
        kept: list[tuple[float, float, int, int]] = []
        for i in range(args.runs):
            try:
                wall, audio, gen_ms, nbytes = _run_one(base, params, args.api_key)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:300]
                print(f"  run {i+1}: HTTP {exc.code} — {body}", file=sys.stderr)
                return 1
            except urllib.error.URLError as exc:
                print(f"  run {i+1}: network error: {exc.reason}", file=sys.stderr)
                return 2
            rtf = audio / wall if wall > 0 else 0.0
            tag = " (warmup)" if i == 0 else ""
            print(f"  run {i+1}{tag}: wall={wall*1000:7.1f}ms  gen={gen_ms:>5}ms  audio={audio:5.2f}s  RTF={rtf:5.2f}x  bytes={nbytes}")
            if i > 0:
                kept.append((wall, audio, gen_ms, nbytes))
        if kept:
            print(_format_row("→ avg", kept))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
