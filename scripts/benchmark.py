"""Evaluate an ASR HTTP endpoint against ground-truth transcripts, computing
WER/CER and writing predictions.json + report.txt to --output-dir.

Requests run concurrently (--concurrency, default 10) via asyncio + httpx, so
the report captures response time and throughput *under concurrent load*, not
just one-at-a-time latency — the two can differ a lot (see "Concurrency"
below): request wall_clock_seconds is the per-request response time, while
the whole run's wall-clock time gives real throughput (requests/sec) at that
concurrency level, and the batch's latency percentiles (p50/p95/p99) show
how much slower requests get when several are in flight at once.

Requires --data-dir to contain a data.tar (one archive member per audio
file) plus a ground_truth.json (tar member name -> {transcription,
raw_transcription}) — the format scripts/download_eval_data.py produces.
Audio is read directly out of data.tar on the fly (no extraction step) via
Python's stdlib tarfile. There is no ground truth for benchmark quality
without it, so this script errors out rather than silently falling back to
a latency-only report.

--api-url takes any endpoint sharing this pipeline's request/response contract
(config.language.sourceLanguage, audio[].audioContent base64, output[].source)
— e.g. this service's internal /predict, or a legacy comparable service.

Audio is normalized to 16-bit PCM WAV before sending (see ensure_pcm16_wav):
FLEURS ships 32-bit float WAV, which Python's stdlib `wave` module — and
evidently some legacy ASR services built on it — can't read at all ("unknown
format: 3"). PCM16 is the universally-supported baseline and costs nothing
for services that already handle float WAV fine.

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


class ResponseOutput(BaseModel):
    source: str


class AsrEndpointResponse(BaseModel):
    """Expected shape of an ASR endpoint's response, validated before use.

    A self-contained copy of src/api/v1/asr/schema.py's AsrResponse rather
    than an import: scripts/ run standalone (python scripts/benchmark.py),
    and importing across into src/ from there needs a sys.path hack this
    tool doesn't otherwise need. time_taken is optional since --api-url may
    point at a legacy service (e.g. wav2vec2) that doesn't report it.
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

    FLEURS ships 32-bit float WAV; Python's stdlib `wave` module can't read
    that at all ("unknown format: 3"), and at least one legacy ASR service
    built on it hits the exact same error. Falls back to the original bytes
    if soundfile can't read the source (e.g. mp3, which libsndfile doesn't
    support) rather than failing the whole request.

    Reads as float32 and scales to int16 manually rather than
    sf.read(dtype="int16") directly: the latter does not apply the
    expected float->int16 scale factor for a float-source WAV in this
    soundfile/libsndfile version, silently truncating samples to near
    zero (confirmed: max amplitude dropped from 0.63 to 0.00003) instead
    of raising, which produced near-silent audio and made every request
    transcribe to the same short garbage output regardless of content.
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
        "audio": [
            {"audioContent": base64.b64encode(audio_bytes).decode("utf-8")}
        ],
    }


async def call_endpoint(
    client: httpx.AsyncClient, url: str, payload: dict, timeout: float
) -> EndpointResult:
    """POST payload to an ASR endpoint, time it, and validate the response
    against AsrEndpointResponse before trusting it.

    Catches HTTPError (network/timeout/HTTP-status errors), ValueError
    (resp.json() on a non-JSON body), and ValidationError (JSON body that
    doesn't match the expected {output: [{source}], time_taken} shape)
    broadly-but-specifically so one flaky or wrongly-shaped response doesn't
    abort the whole run — and so a schema mismatch is reported distinctly
    from a network failure instead of silently returning an empty transcript.
    """
    start = time.monotonic()
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        wall_clock = time.monotonic() - start
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Include the response body: it usually carries the server's
            # actual reason (stack trace, "unsupported format", etc.) that
            # the generic "500 Internal Server Error" status line discards.
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
        f" WER {words.wer:>6.4f}    MER {words.mer:>6.4f}    "
        f"WIL {words.wil:>6.4f}",
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


async def run_benchmark(
    args, ground_truth: dict, tar: tarfile.TarFile
) -> list[dict]:
    semaphore = asyncio.Semaphore(args.concurrency)

    async def worker(
        fname: str, gt: dict, client: httpx.AsyncClient
    ) -> dict | None:
        try:
            member = tar.getmember(fname)
        except KeyError:
            tqdm_asyncio.write(f"skipping {fname}: not found in {tar.name}")
            return None

        audio_bytes = tar.extractfile(member).read()
        payload = build_payload(audio_bytes)
        async with semaphore:
            result = await call_endpoint(
                client, args.api_url, payload, args.timeout
            )

        return {
            "file": fname,
            "reference": gt["transcription"],
            "raw_reference": gt.get("raw_transcription", gt["transcription"]),
            **asdict(result),
        }

    async with httpx.AsyncClient() as client:
        tasks = [worker(fname, gt, client) for fname, gt in ground_truth.items()]
        results = await tqdm_asyncio.gather(
            *tasks, desc="Benchmarking", unit="file"
        )

    return [r for r in results if r is not None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--api-url",
        required=True,
        help="ASR endpoint to benchmark (wav2vec2, this pipeline, etc.)",
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

    references = [r["reference"] for r in records]
    hypotheses = [
        r["transcript"] for r in records
    ]  # empty string for failed calls, counts as total miss

    words = jiwer.process_words(references, hypotheses)
    chars = jiwer.process_characters(references, hypotheses)
    for r in records:
        r["wer"] = (
            jiwer.wer(r["reference"], r["transcript"]) if r["reference"] else None
        )
        r["cer"] = (
            jiwer.cer(r["reference"], r["transcript"]) if r["reference"] else None
        )

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
