"""Compare the legacy Wav2Vec2 ASR service against the new FastConformer
(NeMo + LitServe) service on every audio file in data/.

Both services share the same request/response contract (config.language.sourceLanguage,
audio[].audioContent base64, output[].source), so the same payload is POSTed to each.

There is no ground-truth transcript for these files, so this is NOT a WER/accuracy
benchmark — it reports latency and prints both transcripts side by side so you can
eyeball quality yourself.

Usage:
    python benchmark.py [--data-dir data] [--timeout 600] [--output benchmark_results.json]
"""

import argparse
import base64
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

WAV2VEC2_URL = "http://172.16.16.100:8000/asr"
CONFORMER_URL = "http://localhost:8000/predict"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


@dataclass
class EndpointResult:
    ok: bool
    transcript: str = ""
    reported_time_taken: float | None = None
    wall_clock_seconds: float = 0.0
    error: str = ""


def build_payload(audio_bytes: bytes) -> dict:
    return {
        "config": {"language": {"sourceLanguage": "bn"}},
        "audio": [{"audioContent": base64.b64encode(audio_bytes).decode("utf-8")}],
    }


def call_endpoint(url: str, payload: dict, timeout: float) -> EndpointResult:
    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        wall_clock = time.time() - start
        resp.raise_for_status()
        data = resp.json()
        transcript = " ".join(item.get("source", "") for item in data.get("output", []))
        return EndpointResult(
            ok=True,
            transcript=transcript,
            reported_time_taken=data.get("time_taken"),
            wall_clock_seconds=wall_clock,
        )
    except (requests.RequestException, ValueError) as exc:
        # RequestException: network/timeout/HTTP errors. ValueError: resp.json() on a
        # non-JSON body. Caught broadly-but-specifically so one flaky endpoint doesn't
        # abort the whole comparison run.
        return EndpointResult(ok=False, wall_clock_seconds=time.time() - start, error=str(exc))


def iter_audio_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", help="Directory of audio files to benchmark")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout in seconds")
    parser.add_argument("--output", default="benchmark_results.json", help="Where to save detailed JSON results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = iter_audio_files(data_dir)
    if not files:
        print(f"No audio files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    for path in files:
        print(f"\n=== {path.name} ===")
        payload = build_payload(path.read_bytes())

        print("  wav2vec2 (legacy)   ...", end="", flush=True)
        wav2vec2_result = call_endpoint(WAV2VEC2_URL, payload, args.timeout)
        print(f" {'ok' if wav2vec2_result.ok else 'FAILED'} ({wav2vec2_result.wall_clock_seconds:.2f}s)")

        print("  fastconformer (new) ...", end="", flush=True)
        conformer_result = call_endpoint(CONFORMER_URL, payload, args.timeout)
        print(f" {'ok' if conformer_result.ok else 'FAILED'} ({conformer_result.wall_clock_seconds:.2f}s)")

        print(f"  wav2vec2     : {wav2vec2_result.transcript if wav2vec2_result.ok else wav2vec2_result.error}")
        print(f"  fastconformer: {conformer_result.transcript if conformer_result.ok else conformer_result.error}")

        results.append(
            {
                "file": path.name,
                "wav2vec2": asdict(wav2vec2_result),
                "fastconformer": asdict(conformer_result),
            }
        )

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to {out_path}")

    print("\n=== Latency summary (wall clock) ===")
    print(f"{'file':<20} {'wav2vec2':>12} {'fastconformer':>16}")
    for r in results:
        w = f"{r['wav2vec2']['wall_clock_seconds']:.2f}s" if r["wav2vec2"]["ok"] else "FAILED"
        c = f"{r['fastconformer']['wall_clock_seconds']:.2f}s" if r["fastconformer"]["ok"] else "FAILED"
        print(f"{r['file']:<20} {w:>12} {c:>16}")


if __name__ == "__main__":
    main()
