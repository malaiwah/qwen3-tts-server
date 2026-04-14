#!/usr/bin/env python3
"""Quick CLI tester for a running qwen3-tts-server.

Usage:
    ./test-tts.py "Hello, world!"
    ./test-tts.py "Bonjour" --language French --voice aiden --output bonjour.mp3
    ./test-tts.py --url http://my-gpu-host:8001 "Speak slowly" --instruct "Calm and slow"

Reads the prompt from the positional argument, sends it to the server's
``/v1/audio/speech`` endpoint, and writes the resulting audio to the
specified file (default: out.mp3 in the current directory).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", help="Text to synthesise. Use quotes around the prompt.")
    parser.add_argument("--url", default="http://localhost:8001", help="Base URL of the TTS server.")
    parser.add_argument("--voice", default="ryan", help="Speaker voice (see GET /voices).")
    parser.add_argument("--language", default="English")
    parser.add_argument("--instruct", default=None, help='Style/emotion hint, e.g. "Excited and fast".')
    parser.add_argument("--format", default="mp3", choices=["mp3", "wav", "opus"])
    parser.add_argument("--output", "-o", default=None, help="Output file (default: out.<format>).")
    args = parser.parse_args()

    output = Path(args.output or f"out.{args.format}")

    params = {
        "text": args.text,
        "voice": args.voice,
        "response_format": args.format,
        "language": args.language,
    }
    if args.instruct:
        params["instruct"] = args.instruct

    url = f"{args.url.rstrip('/')}/v1/audio/speech?{urlencode(params)}"
    print(f"POST {url}")
    start = time.time()
    req = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
            gen = resp.headers.get("X-Gen-Time-Ms", "?")
            total = resp.headers.get("X-Latency-Ms", "?")
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Could not reach server at {args.url}: {exc.reason}", file=sys.stderr)
        return 2

    output.write_bytes(data)
    wall = (time.time() - start) * 1000
    print(f"OK: wrote {len(data)} bytes to {output} (gen={gen}ms total={total}ms wall={wall:.0f}ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
