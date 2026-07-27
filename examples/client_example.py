"""Minimal client for the Bangla ASR /predict endpoint.

Usage:
    python examples/client_example.py path/to/audio.wav [--url http://localhost:8000]
"""

import argparse
import base64
import sys

import requests


def transcribe(audio_path: str, url: str) -> dict:
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "config": {"language": {"sourceLanguage": "bn"}},
        "audio": [{"audioContent": audio_b64}],
    }

    resp = requests.post(f"{url.rstrip('/')}/predict", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", help="Path to a wav/flac/ogg/mp3 file")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the ASR server")
    args = parser.parse_args()

    result = transcribe(args.audio_path, args.url)
    for item in result["output"]:
        print(item["source"])
    print(f"(took {result['time_taken']:.2f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
