"""Evaluate an ASR HTTP endpoint against ground-truth transcripts, computing
WER/CER and writing predictions.json + report.txt to --output-dir.

Requests run concurrently (--concurrency, default 10), so the report captures
response time and throughput under concurrent load: wall_clock_seconds is
per-request, the run's wall-clock time gives requests/sec at that concurrency,
and p50/p95/p99 show how much slower requests get with several in flight.

--data-dir must hold a data.tar plus a ground_truth.json in the format
scripts/download_eval_data.py produces; audio is read out of the tar on the
fly. Without ground truth this errors out rather than falling back to a
latency-only report.

--api-url takes any endpoint sharing this pipeline's request/response contract
(config.language.sourceLanguage, audio[].audioContent, output[].source).

Requires the `eval` extra: pip install ".[eval]" (jiwer, tqdm, soundfile;
httpx is already a base dependency).

Usage (from the repo root):
    python scripts/benchmark.py \\
        --api-url http://<host>:<port>/<path> \\
        [--data-dir data/eval_fleurs_bn] [--concurrency 10] \\
        [--output-dir benchmark_output] [--timeout 600]
"""

import argparse
import asyncio
import base64
import io
import json
import re
import statistics
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import jiwer
import numpy as np
import soundfile as sf
from pydantic import BaseModel, ValidationError
from tqdm.asyncio import tqdm as tqdm_asyncio

PUNCTUATION_RE = re.compile(r"[\"“”‘’'।,.!?;:()\[\]{}—–\-]")


def normalize_for_wer(text: str) -> str:
    """Strip punctuation before WER/CER comparison.

    FLEURS references carry quotes and the Bengali daŗi (।) that models never
    transcribe, which otherwise inflates error rates with word-boundary
    artifacts rather than real transcription mistakes.
    """
    return re.sub(r"\s+", " ", PUNCTUATION_RE.sub(" ", text)).strip()


class ResponseOutput(BaseModel):
    source: str


class AsrEndpointResponse(BaseModel):
    """Expected shape of an ASR endpoint's response, validated before use.

    A self-contained copy of AsrResponse rather than an import, since scripts/
    runs standalone. time_taken is optional: --api-url may point at another
    service that doesn't report it.
    """

    output: list[ResponseOutput]
    time_taken: float | None = None


@dataclass
class EndpointResult:
    ok: bool
    transcript: str = ""
    reported_time_taken: float | None = None
    wall_clock_seconds: float = 0.0
    error: str = ""


def ensure_pcm16_wav(audio_bytes: bytes) -> bytes:
    """Re-encode audio as 16-bit PCM WAV, the universally-supported baseline.

    FLEURS ships 32-bit float WAV, which the stdlib `wave` module can't read
    ("unknown format: 3"). Falls back to the original bytes when soundfile
    can't read the source rather than failing the request.

    Scales float32 to int16 by hand: sf.read(dtype="int16") skips the
    float->int16 scale factor for a float-source WAV in this libsndfile
    version, silently producing near-silent audio instead of raising.
    """
    try:
        data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        return audio_bytes
    data_int16 = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, data_int16, samplerate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def build_payload(audio_bytes: bytes) -> dict:
    audio_bytes = ensure_pcm16_wav(audio_bytes)
    return {
        "config": {"language": {"sourceLanguage": "bn"}},
        "audio": [{"audioContent": base64.b64encode(audio_bytes).decode("utf-8")}],
    }


async def call_endpoint(
    client: httpx.AsyncClient, url: str, payload: dict, timeout: float
) -> EndpointResult:
    """POST payload to an ASR endpoint, time it, and validate the response.

    Catches network, non-JSON and schema-mismatch failures separately so one
    bad response doesn't abort the run, and so a schema mismatch is reported
    distinctly rather than as an empty transcript.
    """
    start = time.monotonic()
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        wall_clock = time.monotonic() - start
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = resp.text.strip()
            detail = f" — body: {body[:500]}" if body else ""
            return EndpointResult(
                ok=False,
                wall_clock_seconds=wall_clock,
                error=f"{exc}{detail}",
            )
        data = resp.json()
        try:
            validated = AsrEndpointResponse.model_validate(data)
        except ValidationError as exc:
            return EndpointResult(
                ok=False,
                wall_clock_seconds=wall_clock,
                error=f"Response did not match expected schema: {exc}",
            )
        transcript = " ".join(item.source for item in validated.output)
        return EndpointResult(
            ok=True,
            transcript=transcript,
            reported_time_taken=validated.time_taken,
            wall_clock_seconds=wall_clock,
        )
    except (httpx.HTTPError, ValueError) as exc:
        return EndpointResult(
            ok=False, wall_clock_seconds=time.monotonic() - start, error=str(exc)
        )


def percentile(data: list[float], pct: float) -> float:
    idx = min(len(data) - 1, int(len(data) * pct))
    return data[idx]


def format_report(
    api_url: str,
    data_dir: Path,
    n_records: int,
    n_failed: int,
    words,
    chars,
    concurrency: int,
    batch_wall_clock: float,
    ok_latencies: list[float],
) -> str:
    width = 64
    rule, thin = "=" * width, "-" * width
    throughput = n_records / batch_wall_clock if batch_wall_clock else 0.0

    lines = [
        rule,
        " ASR BENCHMARK REPORT",
        rule,
        "",
        f" Endpoint    {api_url}",
        f" Data dir    {data_dir}",
        f" Utterances  {n_records}  ({n_failed} failed)",
        "",
        thin,
        " ACCURACY  (corpus-level, aggregated -- not an average",
        "           of per-file rates)",
        thin,
        f" WER {words.wer:>6.4f}    MER {words.mer:>6.4f}    " f"WIL {words.wil:>6.4f}",
        f" CER {chars.cer:>6.4f}",
        "",
        f" Word  sub {words.substitutions:<6} ins {words.insertions:<6} "
        f"del {words.deletions:<6} hits {words.hits}",
        f" Char  sub {chars.substitutions:<6} ins {chars.insertions:<6} "
        f"del {chars.deletions:<6} hits {chars.hits}",
        "",
        thin,
        " CONCURRENCY",
        thin,
        f" Concurrency level   {concurrency}",
        f" Batch wall clock    {batch_wall_clock:.2f}s",
        f" Throughput          {throughput:.2f} req/s",
        "",
    ]

    if ok_latencies:
        lines += [
            thin,
            f" LATENCY  (wall clock, successful requests, "
            f"{concurrency} concurrent)",
            thin,
            f" mean {statistics.mean(ok_latencies):.2f}s   "
            f"median {statistics.median(ok_latencies):.2f}s   "
            f"min {min(ok_latencies):.2f}s   "
            f"max {max(ok_latencies):.2f}s",
            f" p50  {percentile(ok_latencies, 0.50):.2f}s   "
            f"p95    {percentile(ok_latencies, 0.95):.2f}s   "
            f"p99 {percentile(ok_latencies, 0.99):.2f}s",
            "",
        ]

    lines.append(rule)
    return "\n".join(lines)


def load_ground_truth(data_dir: Path) -> dict:
    gt_path = data_dir / "ground_truth.json"
    if not gt_path.exists():
        print(
            f"No ground_truth.json found in {data_dir} — this script "
            "requires labeled data.\n"
            "Fetch a small labeled Bengali benchmark with: "
            f"python scripts/download_eval_data.py --data-dir {data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    return json.loads(gt_path.read_text(encoding="utf-8"))


async def run_benchmark(args, ground_truth: dict, tar: tarfile.TarFile) -> list[dict]:
    semaphore = asyncio.Semaphore(args.concurrency)

    async def worker(fname: str, gt: dict, client: httpx.AsyncClient) -> dict | None:
        try:
            member = tar.getmember(fname)
        except KeyError:
            tqdm_asyncio.write(f"skipping {fname}: not found in {tar.name}")
            return None

        audio_bytes = tar.extractfile(member).read()
        payload = build_payload(audio_bytes)
        async with semaphore:
            result = await call_endpoint(client, args.api_url, payload, args.timeout)

        return {
            "file": fname,
            "reference": gt["transcription"],
            "raw_reference": gt.get("raw_transcription", gt["transcription"]),
            **asdict(result),
        }

    async with httpx.AsyncClient() as client:
        tasks = [worker(fname, gt, client) for fname, gt in ground_truth.items()]
        results = await tqdm_asyncio.gather(*tasks, desc="Benchmarking", unit="file")

    return [r for r in results if r is not None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--api-url",
        required=True,
        help="ASR endpoint to benchmark (this pipeline, or another service)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing data.tar + ground_truth.json",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_output",
        help="Where to write predictions.json/report.txt",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max requests in flight at once",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ground_truth = load_ground_truth(data_dir)

    tar_path = data_dir / "data.tar"
    if not tar_path.exists():
        print(
            f"No data.tar found in {data_dir} — this script requires "
            "labeled data.\nFetch a small labeled Bengali benchmark with: "
            f"python scripts/download_eval_data.py --data-dir {data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    batch_start = time.monotonic()
    with tarfile.open(tar_path, "r") as tar:
        records = asyncio.run(run_benchmark(args, ground_truth, tar))
    batch_wall_clock = time.monotonic() - batch_start

    if not records:
        print(
            "No audio files matched ground_truth.json — nothing to benchmark.",
            file=sys.stderr,
        )
        sys.exit(1)

    references = [normalize_for_wer(r["reference"]) for r in records]
    hypotheses = [
        normalize_for_wer(r["transcript"]) for r in records
    ]

    words = jiwer.process_words(references, hypotheses)
    chars = jiwer.process_characters(references, hypotheses)
    for r in records:
        ref_norm = normalize_for_wer(r["reference"])
        hyp_norm = normalize_for_wer(r["transcript"])
        r["wer"] = jiwer.wer(ref_norm, hyp_norm) if r["reference"] else None
        r["cer"] = jiwer.cer(ref_norm, hyp_norm) if r["reference"] else None

    ok_latencies = sorted(r["wall_clock_seconds"] for r in records if r["ok"])
    n_failed = sum(1 for r in records if not r["ok"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.json"
    predictions_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_text = format_report(
        api_url=args.api_url,
        data_dir=data_dir,
        n_records=len(records),
        n_failed=n_failed,
        words=words,
        chars=chars,
        concurrency=args.concurrency,
        batch_wall_clock=batch_wall_clock,
        ok_latencies=ok_latencies,
    )
    report_path = output_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(
        f"\nCorpus WER: {words.wer:.4f}  CER: {chars.cer:.4f}  "
        f"({n_failed} failed / {len(records)} total)"
    )
    print(
        f"Batch: {batch_wall_clock:.2f}s wall clock, "
        f"{len(records) / batch_wall_clock:.2f} req/s at "
        f"concurrency={args.concurrency}"
    )
    print(f"Saved {predictions_path} and {report_path}")


if __name__ == "__main__":
    main()
