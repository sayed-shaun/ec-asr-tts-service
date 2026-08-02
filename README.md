# Bangla ASR Pipeline

Bengali speech-to-text service. Model: [hishab/titu_stt_bn_fastconformer](https://huggingface.co/hishab/titu_stt_bn_fastconformer) (NeMo FastConformer-CTC). Served with [LitServe](https://github.com/Lightning-AI/LitServe).

## Architecture

Two independent services, split by concern — separate processes, separate containers in Docker:

- **LitServe model server** (`run_litserve.py`, built from `src/litserver/`) — holds the model, GPU-bound. Always binds `0.0.0.0:8000` inside its own container; not meant to be reached by anything other than the gateway.
- **Gateway** (`main.py`) — pure FastAPI, no model loaded in-process. This is the public entrypoint, always binding `0.0.0.0:GATEWAY_PORT` (`8000` by default — the only network setting left configurable, since it's the one thing that genuinely varies per deployment). It reaches the LitServe server over a real HTTP call (`httpx.AsyncClient`) at a **fixed** address: `litserver:8000` (see `src/core/config.py`) — not env-configurable, since the two services' wiring to each other is an implementation detail, not something a deployment needs to vary.

`litserver:8000` assumes Docker Compose's network (the `litserver` hostname resolves via Docker's internal DNS — see `docker-compose.yml`'s shared `asr-net` network). Running the two processes bare on one host without Compose (two terminals) needs `litserver` to resolve to `127.0.0.1`, e.g. by adding this to `/etc/hosts`:

```
127.0.0.1 litserver
```

`LITSERVE_PORT` is also fixed at `8000` — the same as `GATEWAY_PORT`'s default. Two containers can both use `8000` fine (separate network namespaces), but two bare processes on **one host** can't both bind `8000` — set `GATEWAY_PORT` to something else (e.g. `8080`) for local two-terminal dev outside Docker.

Run them as two separate commands — there is no single entrypoint that starts both:

```bash
python run_litserve.py                    # terminal 1 — model server, binds 8000
GATEWAY_PORT=8080 python main.py       # terminal 2 — gateway, binds 8080 to avoid the clash
```

In Docker Compose, the `gateway` service waits on the `litserver` service's healthcheck before starting (model load can take a while on first run — see `docker-compose.yml`).

```mermaid
flowchart LR
    Client(["Client / static test page"])

    subgraph Gateway["Gateway process — public — 0.0.0.0:GATEWAY_PORT"]
        Info["GET /v1/asr/info"]
        File["POST /v1/asr/transcribe/file"]
        WS["WS /v1/live-cc/ws"]
        Static["/static (test GUI)"]
    end

    subgraph LitServe["LitServe service — internal — litserver:8000"]
        Predict["POST /predict"]
        LitAPI["ASRLitAPI: decode_request -> predict -> encode_response"]
        Engine["ASREngine (NeMo model, GPU)"]
        Health["GET /health"]
    end

    Client -->|HTTP / WebSocket| Info
    Client --> File
    Client --> WS
    Client --> Static

    File -->|real HTTP, httpx.AsyncClient| Predict
    WS -->|real HTTP per chunk, httpx.AsyncClient| Predict
    Predict --> LitAPI --> Engine
```

**Note:** `POST /predict` (the raw JSON inference endpoint LitServe itself exposes) only runs on the internal LitServe port — it is not proxied through the gateway. Externally, use `POST /v1/asr/transcribe/file` (multipart upload) instead; the gateway forwards that to the internal endpoint for you. If you need raw base64-JSON access from outside the container, either open the internal port at your own risk or add a passthrough route to the gateway.

## Structure

```
.
├── main.py                        # entrypoint — gateway service (pure FastAPI, no model)
├── run_litserve.py                # entrypoint — LitServe model server service
├── fastapi.Dockerfile              # lightweight image for the gateway (no torch/nemo)
├── litserver.Dockerfile            # heavy image for the LitServe model server (GPU, torch/nemo)
├── src/
│   ├── core/
│   │   ├── config.py              # env-driven settings (no prefix)
│   │   └── logging.py             # loguru sinks
│   ├── litserver/
│   │   ├── engine.py              # NeMo model load + transcribe
│   │   ├── litapi.py              # LitServe API: setup/decode/predict/encode
│   │   ├── speaker.py             # Resemblyzer voice-embedding wrapper
│   │   ├── speaker_router.py      # internal /internal/speaker/embed route
│   │   └── server.py              # builds the LitServer instance
│   ├── utils/
│   │   └── audio.py               # base64 -> waveform decode/resample
│   └── api/v1/
│       ├── asr/router.py          # /v1/asr/* routes (gateway; proxies to LitServe over HTTP)
│       ├── asr/schema.py          # request/response models
│       └── live_cc/router.py      # /v1/live-cc/ws (gateway; proxies to LitServe over HTTP)
├── static/index.html              # manual test page
├── tests/test_api.py              # unit tests (model mocked)
├── examples/client_example.py     # minimal Python client
└── scripts/
    ├── benchmark.py                # WER/CER + latency eval against ground truth (requires it)
    └── download_eval_data.py       # fetch a small labeled Bengali ASR benchmark (ground truth)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Model license: **CC-BY-NC-4.0** (non-commercial). `nemo_toolkit[asr]` is a heavy install — several minutes, several GB. `pip install -e .` alone gets you the gateway's dependencies only (fastapi/httpx/uvicorn — no torch/nemo). To also run the model server locally, install the `serve` extra: `pip install -e ".[serve]"`.

## Run

```bash
python run_litserve.py   # terminal 1
python main.py           # terminal 2
```

First request downloads the checkpoint (cached under `$HF_HOME` / `~/.cache/huggingface`) — slow once, fast after. The gateway will return connection errors for `/v1/asr/transcribe/file` and `/v1/live-cc/ws` until `run_litserve.py` has finished loading the model — there's no readiness wait outside of Docker Compose (see below), so give it a moment on first run.

Test page: `http://localhost:8000/static/index.html`

## Docker

```bash
docker compose up --build
```

Two separate images, built from two separate Dockerfiles:

- `litserver` — built from `litserver.Dockerfile` (CUDA/PyTorch base, `torch`/`nemo_toolkit`/`litserve` installed via the `serve` extra). GPU-bound, several GB.
- `gateway` — built from `fastapi.Dockerfile` (`python:3.12-slim`, base dependencies only — no ML deps at all). Small, fast to build/deploy/scale independently of the model image.

`gateway` won't start accepting traffic until `litserver`'s healthcheck passes (`GET /health`), which covers the model-load wait automatically — no manual readiness polling needed.

GPU is on by default (`ACCELERATOR=cuda`, requires `nvidia-container-toolkit`). For CPU-only, remove the `deploy.resources` block from the `litserver` service in `docker-compose.yml` and set `ACCELERATOR=cpu`.

## Test

```bash
pip install -e ".[dev,serve]"
pytest -q
```

`serve` is required alongside `dev` because the tests exercise `ASREngine`/`ASRLitAPI` directly (with the model mocked), which import `torch`/`numpy`.

## Endpoints

| | |
|---|---|
| `POST /v1/asr/transcribe/file` | multipart file upload (gateway, public) |
| `WS /v1/live-cc/ws` | streamed captions, raw 16-bit PCM (gateway, public) |
| `GET /v1/asr/info` | model metadata (gateway, public) |
| `GET /health` | LitServe healthcheck (internal) |
| `GET /docs` | Swagger UI (gateway) — WebSocket routes never appear here, OpenAPI has no way to describe them |
| `POST /predict` | raw JSON inference — **internal LitServe only**, not exposed on the gateway |

`/predict` is LitServe's own default route (not configurable — see `main.py`'s `PREDICT_PATH` and `src/litserver/server.py`, which no longer overrides it) — a fixed implementation detail between the gateway and LitServe, not something a deployment needs to vary.

### Inference request (internal `/predict`, and what `/v1/asr/transcribe/file` builds internally)

```json
{
  "config": {
    "language": {"sourceLanguage": "bn"},
    "samplingRate": 16000
  },
  "audio": [
    {"audioContent": "<base64-encoded wav/flac/ogg/mp3>"}
  ]
}
```

### Response

```json
{
  "taskType": "asr",
  "output": [{"source": "transcribed bengali text"}],
  "time_taken": 0.42
}
```

### live-cc messages

```json
{"text": "in-progress guess so far", "is_final": false}
{"text": "committed bengali text", "is_final": true}
```

### Speaker gate (optional, off by default)

For live-cc calls where background talkers near the caller's mic shouldn't be
transcribed (single mic, no hardware control — e.g. a voice call bot). When
`SPEAKER_GATE_ENABLED=true`:

1. The first `SPEAKER_ENROLL_SECONDS` of the call are used to enroll a
   reference voice embedding (via LitServe's internal `/internal/speaker/embed`,
   backed by [Resemblyzer](https://github.com/resemble-ai/Resemblyzer)) —
   this assumes the target caller is the first person heard; a background
   voice speaking first enrolls the wrong speaker.
2. Every chunk after that is embedded and compared to the enrollment; chunks
   below `SPEAKER_SIMILARITY_THRESHOLD` cosine similarity are dropped —
   no caption emitted at all for that chunk, instead of being transcribed.

This is **segment-level gating only** — it decides whether a whole chunk
sounds like the enrolled speaker, it cannot separate two people talking
*simultaneously* within the same chunk (that would need real source
separation, not gating). `SPEAKER_SIMILARITY_THRESHOLD=0.75` is a generic
starting point from speaker-verification literature, not measured against
this project's own calls — tune it against real recordings before trusting
it in production.

Requires the `speaker` extra (`pip install ".[serve,speaker]"`; already
included in `litserver.Dockerfile`).

## Config

All settings are plain env vars (no prefix). See `.env.example` for the full list and defaults. Key ones beyond the model itself:

- `DEVICES` / `WORKERS_PER_DEVICE` — GPU worker process count; the actual concurrency lever for inference throughput (see `src/core/config.py` for the memory-tradeoff notes).
- `GATEWAY_PORT` — public gateway port (always binds `0.0.0.0`). The only network setting that's actually configurable — the gateway always reaches LitServe at the fixed `litserver:8000` (see "Architecture" above), not an env var.
- `SPEAKER_GATE_ENABLED` / `SPEAKER_ENROLL_SECONDS` / `SPEAKER_SIMILARITY_THRESHOLD` — optional live-cc speaker gate, see "Speaker gate" above.

## Client examples

```bash
python examples/client_example.py path/to/audio.wav
```

Currently POSTs to `/predict`, not a real route on this service (public or internal) — broken until updated to target `/v1/asr/transcribe/file` (or the internal `/predict` if run against the LitServe port directly).

## Evaluating against ground truth

`scripts/benchmark.py` computes WER/CER for any ASR HTTP endpoint sharing this pipeline's request/response contract (this service's internal `/predict`, a legacy comparable service, etc.) against labeled ground truth — it requires a `ground_truth.json` in `--data-dir` and errors out with a pointer to the command below if one isn't there (plain `data/` has none, it's just sample audio for eyeballing quality).

Fetch a small labeled Bengali benchmark, then run it:

```bash
pip install ".[eval]"
python scripts/download_eval_data.py --data-dir data/eval_fleurs_bn
python scripts/benchmark.py --api-url http://127.0.0.1:8000/predict \
    --data-dir data/eval_fleurs_bn --output-dir benchmark_output
```

`download_eval_data.py` pulls the full bn_in test + validation splits (a few hundred utterances each, not train) from [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0) via the `datasets` library, writing a single `data.tar` (one archive member per utterance, not hundreds of loose files) + `ground_truth.json` mapping tar member name -> transcript. Member names are `fleurs_bn_{split}_{position:04d}.wav` — deliberately not FLEURS' own `id` field, which is **not unique within a split** (confirmed: the validation split has 402 rows but only 150 distinct ids, which would silently overwrite 252 files if used). `benchmark.py` reads audio straight out of `data.tar` on the fly (via `tarfile`) — no extraction step.

`benchmark.py` writes two files to `--output-dir`:

- `predictions.json` — per-utterance reference, hypothesis, per-file WER/CER, latency, and any request error.
- `report.txt` — corpus-level WER/CER/MER/WIL (aggregated over the whole set, not an average of per-file rates) and latency stats. Per-file WER/CER lives in `predictions.json`, not here.
