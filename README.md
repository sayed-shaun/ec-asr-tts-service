# ASR Inference Pipeline

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/gateway-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LitServe](https://img.shields.io/badge/model%20server-LitServe-792EE5)](https://github.com/Lightning-AI/LitServe)
[![Docker](https://img.shields.io/badge/deploy-Docker%20Compose-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![sherpa-onnx](https://img.shields.io/badge/ASR-streaming%20Zipformer2-0A7E8C)](https://github.com/k2-fsa/sherpa-onnx)
[![Parler-TTS](https://img.shields.io/badge/TTS-indic--parler--tts-FF9933)](https://huggingface.co/ai4bharat/indic-parler-tts)
[![Hugging Face](https://img.shields.io/badge/models%20from-Hugging%20Face-yellow)](#current-models)

Bengali speech: streaming ASR (sherpa-onnx Zipformer2) and TTS (indic-parler-tts), served with [LitServe](https://github.com/Lightning-AI/LitServe) behind a FastAPI gateway.

- One-shot file transcription, streamed live captions, and text-to-speech over REST
- A separate WebSocket service for live calls: incremental ASR and cancellable, clause-by-clause TTS
- OpenAI-compatible `/v1/audio/transcriptions` and `/v1/audio/speech` for drop-in clients
- Benchmark tool for WER/CER against labeled ground truth

<p align="center">
  <a href="#current-models">Models</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#live-voicebot--callbot">Voicebot</a> ·
  <a href="#structure">Structure</a> ·
  <a href="#setup">Setup</a> ·
  <a href="#docker">Docker</a> ·
  <a href="#test">Test</a> ·
  <a href="#endpoints">Endpoints</a> ·
  <a href="#config">Config</a> ·
  <a href="#client-examples">Client examples</a> ·
  <a href="#evaluating-against-ground-truth">Evaluation</a>
</p>

---

## Current Models

### ASR

- **Checkpoint:** [`vosk-model-small-streaming-bn`](https://huggingface.co/alphacep/vosk-model-small-streaming-bn) — pulled from Hugging Face at load time, not hosted by this project
- **Architecture:** sherpa-onnx streaming Zipformer2 transducer (ONNX). Consumes audio as one growing stream, so there is no attention-memory blowup on long clips and no segmenting step
- **Input:** 16 kHz mono (`SAMPLE_RATE`)
- **License:** Apache-2.0 (commercial use OK)
- **Swap it:** set `ZIPFORMER_MODEL_NAME`. Checkpoints name their ONNX files differently, so a different repo also needs a `ZipformerLayout` — see [`zipformer/layouts.py`](src/litserver/zipformer/layouts.py), which ships `VOSK_BN` and `K2_FSA`. `GET /api/v1/asr/info` reports what is actually loaded

**Why this one, and only this one**

This is the only ASR engine in the repo. It decodes **incrementally**, which is
what a live call needs: one stream stays open across frames, and sherpa-onnx's
endpoint rules say when the caller stopped talking. Whole-utterance engines
cannot do either, so NeMo Conformer, Wav2Vec2 and Whisper were removed along
with the `ENGINE` switch and `MAX_SEGMENT_SECONDS` that existed to serve them.

**Inference notes**

- **Runs on CPU by default** (`ZIPFORMER_PROVIDER=cpu`). That is an onnxruntime execution provider, independent of `ACCELERATOR`: it is a property of the installed sherpa-onnx wheel. The PyPI wheel bundles a CPU-only `libonnxruntime`, so `cuda` there **silently falls back to CPU** rather than failing — using it needs a CUDA wheel from [k2-fsa's index](https://k2-fsa.github.io/sherpa/onnx/python/install.html) built against the image's CUDA/cuDNN pair

  **Measured** on an RTX 2050 (4GB), 24 FLEURS clips / 340s audio, GPU otherwise idle:

  | provider | batch | RTF | frame mean | frame p95 | VRAM |
  |---|---|---|---|---|---|
  | `cpu` | 7.59s | 0.0223 | 2.72 ms | 17.45 ms | — |
  | `cuda` | 3.88s | 0.0114 | 2.58 ms | 17.04 ms | 303 MiB |

  CUDA is ~1.95x on the batch path but only ~1.06x per streaming frame — noise. CPU already decodes a 100ms frame in 2.7ms (~37x real time) and runs batch at 45x real time, so there is no latency problem to solve, and the 303 MiB matters on a card that must also hold the ~2.6GB TTS model. Revisit CUDA only for bulk offline transcription on a box where the GPU is not shared with TTS.

  **To A/B CUDA:** two things must change together — the wheel supplies the provider, the setting selects it.

  ```bash
  SHERPA_ONNX_CUDA_VERSION=1.13.5+cuda12.cudnn9.onnxruntime1.27.1 docker compose build litserver
  # then set ZIPFORMER_PROVIDER=cuda in .env
  ```

  Pick the exact local version from [k2-fsa's index](https://k2-fsa.github.io/sherpa/onnx/cuda.html) matching the image's CUDA/cuDNN. The CUDA wheel is monolithic (no `sherpa-onnx-core` split) but does **not** bundle cuDNN — it needs `libcudnn.so.9` from the system, which the `cudnn9` base image supplies and a slimmer base would not. If the wheel is CPU-only while `ZIPFORMER_PROVIDER=cuda`, the engine logs a warning at load — onnxruntime itself would fall back to CPU without a word.

  **Where to run it:** keep `cpu` for live calls. The checkpoint is small and decodes a few hundred ms per forward pass, where kernel-launch overhead can exceed the compute, and the GPU is wanted for the autoregressive TTS model — whose latency a caller actually hears. CUDA is worth measuring on the batch `/predict` path instead, where `decode_streams()` runs one encoder pass across a whole batch
- Higher WER than the heavier alternatives — the trade for size, speed and streaming
- Hallucinated text on silence/noise is filtered by `BaseASREngine.is_non_speech` (thresholds measured on this corpus)
- Spelled-out numbers are rewritten as digits when `ITN_ENABLED` (the default) — see [`utils/itn.py`](src/utils/itn.py)

### TTS

- **Checkpoint:** [`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts) (`TTS_MODEL_NAME`) — ~2.6GB in bf16, GPU-resident, pulled from Hugging Face at load time
- **Output:** mono 44.1 kHz; the gateway hands it back as base64 WAV, a playable `audio/wav` body, or raw PCM frames
- **Voices:** named voices map to Parler style prompts in [`parler/voices.py`](src/litserver/parler/voices.py); `GET /api/v1/tts/voices` lists them, and a request's `description` overrides the prompt with free-form style text
- **Long text** is split on clause boundaries at `TTS_MAX_CHARS` ([`parler/chunking.py`](src/litserver/parler/chunking.py)) and rejoined with a short pause, so a long reply doesn't hit the model as one generation. On the voicebot socket the same chunker runs incrementally, speaking each clause as soon as it completes
- **Turn it off** with `TTS_ENABLED=false` — worth doing whenever the GPU can't hold both checkpoints

---

## Architecture

Three independent services, separate processes and containers:

| Service | File | Role |
|---|---|---|
| **LitServe model server** | `run_litserve.py` | Holds the models, GPU-bound. Binds `0.0.0.0:LITSERVE_PORT` (default `8000`), internal only — no host port is published. Two LitAPIs share the process: ASR on `/predict`, TTS on `/synthesize`. |
| **Gateway** | `main.py` | Pure FastAPI, no model loaded. Public entrypoint, binds `0.0.0.0:GATEWAY_PORT` (default `8000`). Reaches LitServe over real HTTP at `LITSERVE_BASE_URL` (default `http://litserver:8000`). |
| **Voicebot streaming** | `run_voicebot.py` | WebSocket service for live calls, binds `0.0.0.0:VOICEBOT_PORT` (default `8100`). Unlike the gateway it **owns the models in-process** — see below. |

`litserver` resolves via Docker Compose's internal DNS (`asr-net` network), and `gateway` waits on `litserver`'s healthcheck before starting. Split the two across hosts and the gateway only needs `LITSERVE_BASE_URL` pointed at the new address.

```mermaid
flowchart LR
    Client(["Client"])

    subgraph Gateway["Gateway process — public — 0.0.0.0:GATEWAY_PORT"]
        Info["GET /api/v1/asr/info"]
        Transcribe["POST /api/v1/asr/transcribe"]
        File["POST /api/v1/asr/transcribe/file"]
        Speak["POST /api/v1/tts/synthesize"]
        OpenAI["POST /v1/audio/transcriptions<br/>POST /v1/audio/speech"]
    end

    subgraph LitServe["LitServe service — internal — litserver:LITSERVE_PORT"]
        Predict["POST /predict"]
        LitAPI["ASRLitAPI: decode_request -> predict -> encode_response"]
        Engine["ZipformerEngine (settings.ZIPFORMER_MODEL_NAME)"]
        Synth["POST /synthesize"]
        TtsAPI["TTSLitAPI: decode_request -> predict -> encode_response"]
        TtsEngine["ParlerTTSEngine (settings.TTS_MODEL_NAME)"]
        Health["GET /health"]
    end

    Client -->|HTTP| Info
    Client --> Transcribe
    Client --> File
    Client --> Speak
    Client --> OpenAI

    Transcribe -->|real HTTP, httpx.AsyncClient| Predict
    File -->|real HTTP, httpx.AsyncClient| Predict
    Speak -->|real HTTP, httpx.AsyncClient| Synth
    OpenAI -->|real HTTP, httpx.AsyncClient| Predict
    OpenAI -->|real HTTP, httpx.AsyncClient| Synth
    Predict --> LitAPI --> Engine
    Synth --> TtsAPI --> TtsEngine
```

<details>
<summary>Why TTS shares the LitServe process instead of getting its own service</summary>

`ls.LitServer` accepts a list of `LitAPI`s, each with its own `api_path` and its
own worker processes — so ASR and TTS never contend for one request loop, and
there is no second container, second port or second healthcheck to operate.

The cost is that `ACCELERATOR` / `DEVICES` / `WORKERS_PER_DEVICE` are
server-level: both checkpoints are resident on the same GPU, and each is loaded
`WORKERS_PER_DEVICE` times. `indic-parler-tts` is ~2.6GB in bf16 (the ASR
checkpoint costs nothing there — it runs on CPU under onnxruntime by default). If that doesn't fit, set `TTS_ENABLED=false` (the TTS
routes stay mounted on the gateway and answer `503`) or split it into its own
service.

</details>

### Live voicebot / callbot

`run_voicebot.py` is a separate service for streaming use, with two sockets:

| | |
|---|---|
| `WS /api/v1/voicebot/asr` | caller streams 16-bit PCM in; server sends `{"type":"partial"\|"final","text":…}` |
| `WS /api/v1/voicebot/tts` | client sends `configure` / `text` / `flush` / `cancel` / `end`; server streams PCM frames with `audio_start` / `audio_end` around each clause |

**Using it**

```bash
python run_voicebot.py                                     # or: docker compose up voicebot
curl localhost:8100/health

python examples/voicebot_client.py asr path/to/audio.wav   # stream PCM in, get text out
python examples/voicebot_client.py tts "আমি ভালো আছি।"      # writes voicebot_tts.wav
```

`examples/voicebot_client.py` streams a file in 100ms frames the way telephony
would, printing partials as they arrive plus a `final` when the endpoint rules
fire. Set `TTS_ENABLED=false` to run ASR only — the TTS socket then closes with
`1013 TTS is disabled` rather than hanging.

Wire protocol, if you are writing your own client:

| socket | client sends | server sends |
|---|---|---|
| `/asr` | binary PCM frames; `{"type":"end"}` to flush | `ready`, `partial`, `final`, `done` |
| `/tts` | `configure` / `text` / `flush` / `cancel` / `end` | `ready`, `configured`, `audio_start`, binary PCM, `audio_end`, `cancelled`, `done`, `error` |

It holds the models in-process rather than proxying, which is forced rather
than chosen: a call needs **one ASR stream open across frames** and a TTS
generation that can be **cancelled mid-sentence**, and LitServe has no
WebSocket support and runs stateless workers in separate processes. The
LitServe path stays the right one for batch and REST.

- **ASR** is the same Zipformer engine the LitServe path loads, built directly rather than proxied — it decodes incrementally. Endpoint detection (`rule1_min_trailing_silence` etc.) is what tells you the caller stopped talking, so the orchestrator knows when to run the LLM.
- **TTS** accepts text as deltas, so an LLM's output can be fed in as generated; a clause is spoken as soon as it is complete rather than after the whole reply.
- **Barge-in** is the `cancel` message: it bumps a generation counter, dropping everything queued *and* stopping the frames of whatever is mid-send.
- The LLM that joins the two sockets is not part of this repo; the orchestrator owns turn-taking.

Concurrency: each ASR connection gets its own sherpa stream off the shared
recognizer, so calls decode independently. TTS synthesis is serialized behind
one lock — a single Parler model cannot be entered concurrently, so heavy
concurrent TTS load needs more replicas, not more sockets.

<details>
<summary><code>/predict</code> is internal-only — details</summary>

`POST /predict` (LitServe's own raw JSON inference route) only runs on the internal LitServe port; it is not proxied through the gateway. Externally, use `POST /api/v1/asr/transcribe` (raw base64 JSON) or `POST /api/v1/asr/transcribe/file` (multipart upload) — the gateway forwards either to `/predict` for you. Need raw base64-JSON access from outside the container? Either open the internal port at your own risk or add a passthrough route to the gateway.

</details>

---

## Structure

```
.
├── main.py                        # entrypoint — gateway service (pure FastAPI, no model)
├── run_litserve.py                # entrypoint — LitServe model server service
├── run_voicebot.py                # entrypoint — streaming WebSocket service (live calls)
├── examples/voicebot_client.py    # CLI client for both voicebot sockets
├── fastapi.Dockerfile             # lightweight image for the gateway (no ML deps)
├── litserver.Dockerfile           # heavy image for the model server and voicebot (GPU, torch)
├── src/
│   ├── core/
│   │   ├── config.py              # env-driven settings (no prefix)
│   │   └── logging.py             # loguru sinks
│   ├── litserver/                 # one package per model, plus the shared contracts
│   │   ├── base.py                # BaseEngine + BaseASREngine/BaseTTSEngine + Audio
│   │   ├── litapi.py              # ASRLitAPI (/predict) and TTSLitAPI (/synthesize)
│   │   ├── server.py              # builds the LitServer instance (both APIs)
│   │   ├── zipformer/             # the ASR engine
│   │   │   ├── engine.py          # streaming Zipformer2 + ZipformerSession + build()
│   │   │   └── layouts.py         # per-checkpoint Hub repo layouts
│   │   └── parler/
│   │       ├── engine.py          # indic-parler-tts load + synthesize + build()
│   │       ├── chunking.py        # split text into speakable clauses
│   │       └── voices.py          # voice -> style-prompt table (no heavy imports)
│   ├── voicebot/
│   │   └── app.py                 # streaming ASR + TTS WebSockets (models in-process)
│   ├── utils/
│   │   ├── audio.py               # base64 -> waveform decode/resample, WAV framing
│   │   └── itn.py                 # spelled-out Bengali numbers -> digits
│   └── api/
│       ├── client.py              # gateway -> LitServe HTTP client (shared by every router)
│       └── v1/
│           ├── asr/router.py      # /api/v1/asr/*, /v1/audio/transcriptions, /asr
│           ├── asr/schema.py      # request/response models
│           ├── tts/router.py      # /api/v1/tts/*, /v1/audio/speech
│           └── tts/schema.py      # request/response models
├── tests/test_api.py              # unit tests (models mocked)
└── scripts/
    ├── benchmark.py                # WER/CER + latency eval against ground truth (requires it)
    └── download_eval_data.py       # fetch a small labeled Bengali ASR benchmark (ground truth)
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
```

- `pip install -e .` alone installs only the gateway's dependencies (fastapi/httpx/uvicorn — no ML stack). That is enough to run `main.py` against a model server running elsewhere.
- To run the model server or the voicebot locally, install the `serve` extra: `pip install -e ".[serve]"` — it pulls torch, litserve, librosa, sherpa-onnx and `parler-tts` from git (several minutes, several GB).
- `examples/voicebot_client.py` additionally needs `websockets` (`pip install websockets`); `soundfile` comes with the `serve` extra.
- **Python version:** fixed at 3.12 (`requires-python = "==3.12.*"`) across both images — the gateway (`python:3.12-slim`) and the LitServe/GPU image (`pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`).

---

## Docker

```bash
docker compose up --build
```

Three services, from two Dockerfiles:

| Service | Dockerfile | Published port | Notes |
|---|---|---|---|
| `litserver` | `litserver.Dockerfile` | none (internal) | CUDA/PyTorch base, GPU-bound, several GB |
| `voicebot` | `litserver.Dockerfile` | `VOICEBOT_PORT` (8100) | Same image, `python run_voicebot.py` — holds its own copy of the models |
| `gateway` | `fastapi.Dockerfile` | `GATEWAY_PORT` (8000) | `python:3.12-slim`, no ML deps — small, fast to build/deploy/scale independently |

- `gateway` won't accept traffic until `litserver`'s healthcheck (`GET /health`) passes — no manual readiness polling needed. First request otherwise pays the checkpoint download; the `hf-cache` volume keeps it across restarts and shares it between `litserver` and `voicebot` (`HF_HOME=/opt/cache/huggingface`).
- `voicebot` has no `depends_on`: it loads its models itself and is up when its own `GET /health` answers.
- GPU is on by default (`ACCELERATOR=cuda`, requires `nvidia-container-toolkit`). For CPU-only: remove the `deploy.resources` blocks from `litserver` and `voicebot` in `docker-compose.yml` and set `ACCELERATOR=cpu`. ASR is unaffected either way — it runs under onnxruntime, on `ZIPFORMER_PROVIDER`.

Run one service on its own with `docker compose up gateway litserver`, or `docker compose up voicebot` for live calls only.

---

## Test

```bash
pip install -e ".[dev,serve]"
pytest -q
```

`serve` is required alongside `dev` because the tests exercise `ZipformerEngine`, `ASRLitAPI`/`TTSLitAPI` and the voicebot sockets directly (models mocked), and importing those pulls in `numpy`, `sherpa-onnx`, `litserve` and `parler-tts`. The whole file fails to collect if any of them is missing — `parler-tts` installs from git and is the one most often absent. No checkpoint is downloaded and no GPU is needed.

---

## Endpoints

**Gateway** (`main.py`, public on `GATEWAY_PORT`):

| | |
|---|---|
| `POST /api/v1/asr/transcribe` | raw base64 JSON, forwarded straight to `/predict` |
| `POST /api/v1/asr/transcribe/file` | multipart file upload |
| `GET /api/v1/asr/info` | ASR metadata: Hugging Face model name, language, sample rate |
| `POST /api/v1/tts/synthesize` | text in, base64 WAV out |
| `POST /api/v1/tts/synthesize/audio` | same synthesis, returned as a playable `audio/wav` body |
| `GET /api/v1/tts/voices` | available voices and the style prompt each maps to |
| `GET /api/v1/tts/info` | TTS metadata; `enabled: false` when `TTS_ENABLED=false` |
| `GET /health` | gateway liveness — loads no model, so it answers while the model server is still warming up |
| `GET /docs` | Swagger UI — WebSocket routes never appear here |

**Drop-in compatibility routes**, mounted at the root so a client reaches them by base URL alone:

| | |
|---|---|
| `POST /v1/audio/transcriptions` | OpenAI-compatible transcription: multipart audio in, `{"text": …}` out (segments joined into one utterance) |
| `POST /v1/audio/speech` | OpenAI-compatible speech: `{input, voice, response_format}` in, raw `wav`/`pcm` out (`pcm` strips the WAV header for telephony) |
| `POST /asr` | what the chatbot UI posts recorded audio to through Caddy's `/asr*` proxy. Identical to `/api/v1/asr/transcribe/file`; the `model_type` form field is accepted and ignored, since this service serves exactly one engine |

**Voicebot** (`run_voicebot.py`, public on `VOICEBOT_PORT`) — see [Live voicebot / callbot](#live-voicebot--callbot):

| | |
|---|---|
| `WS /api/v1/voicebot/asr` | 16-bit PCM in, `partial`/`final` transcripts out |
| `WS /api/v1/voicebot/tts` | text deltas in, cancellable PCM frames out |
| `GET /health` | liveness; reports both loaded checkpoints |

**LitServe** (`run_litserve.py`, internal only — no host port published):

| | |
|---|---|
| `POST /predict` | raw JSON inference |
| `POST /synthesize` | raw JSON synthesis |
| `GET /health` | LitServe healthcheck, what `gateway`'s `depends_on` waits for |

<details>
<summary>Inference request / response payloads</summary>

Request (internal `/predict`, and what `/api/v1/asr/transcribe/file` builds internally):

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

Response:

```json
{
  "taskType": "asr",
  "output": [{"source": "transcribed bengali text"}],
  "time_taken": 0.42
}
```

TTS request (`/api/v1/tts/synthesize`, and internal `/synthesize`):

```json
{
  "input": "আমি বাংলায় কথা বলি।",
  "voice": "Aditi",
  "description": null
}
```

`voice` picks a style prompt from `GET /api/v1/tts/voices`; `description` overrides
it with free-form style text. Both are optional — omitted, the request uses
`TTS_VOICE`. Response:

```json
{
  "taskType": "tts",
  "audioContent": "<base64-encoded mono 16-bit wav>",
  "sampleRate": 44100,
  "voice": "Aditi",
  "time_taken": 1.83
}
```

</details>

---

## Config

All settings are plain env vars (no prefix), read from `.env`. See [`.env.example`](.env.example) for the annotated list and [`src/core/config.py`](src/core/config.py) for the defaults.

**Models**

- `ZIPFORMER_MODEL_NAME` — the ASR checkpoint. A different repo layout also needs a `ZipformerLayout` (see [`zipformer/layouts.py`](src/litserver/zipformer/layouts.py)).
- `ZIPFORMER_PROVIDER` — onnxruntime execution provider, `cpu` or `cuda`. Independent of `ACCELERATOR`; needs a matching wheel (see [Current Models](#current-models)).
- `TTS_ENABLED` — mount the TTS LitAPI alongside ASR in the same LitServe process, and load Parler in the voicebot. Costs a second checkpoint's VRAM per worker; `false` serves ASR only, and the TTS routes answer `503` while the voicebot's TTS socket closes with `1013`.
- `TTS_MODEL_NAME` / `TTS_VOICE` / `TTS_MAX_CHARS` — the TTS checkpoint, the voice used when a request names none, and the clause length text is split at before synthesis.
- `ITN_ENABLED` — rewrite spelled-out Bengali numbers as digits. On by default; measured worth ~1.7 WER points on FLEURS.

**Serving**

- `ACCELERATOR` / `DEVICES` / `WORKERS_PER_DEVICE` — LitServe placement and worker count, the actual concurrency lever for inference throughput. Server-level, so ASR and TTS share them.
- `TRANSCRIBE_BATCH_SIZE` — utterances batched per engine call.
- `LITSERVE_TIMEOUT` — how long a request may sit in LitServe's queue before a `504`. Raised well above LitServe's own 30s default because Parler generation is slow enough to queue past it with nothing wrong.

**Networking**

- `GATEWAY_PORT` / `LITSERVE_PORT` / `VOICEBOT_PORT` — each service's own port; all bind `0.0.0.0`. Only the gateway and voicebot ports are published to the host.
- `LITSERVE_BASE_URL` — where the gateway reaches the model server. Defaults to the compose service name (`http://litserver:8000`), which resolves only inside that network; set it explicitly when the two run apart, and keep it in step with `LITSERVE_PORT`.
- `CORS_ALLOW_ORIGINS` — comma-separated origins allowed to call the gateway cross-origin, so a browser UI can hit `/asr` and `/v1/audio/speech` directly instead of proxying.
- `VOICEBOT_PCM_FRAME_MS` — size of each PCM frame the TTS socket emits; below ~20ms per-frame overhead dominates.

---

## Client examples

```bash
# transcribe a file through the gateway
curl -F file=@path/to/audio.wav http://localhost:8000/api/v1/asr/transcribe/file

# the same thing, OpenAI-style
curl -F file=@path/to/audio.wav http://localhost:8000/v1/audio/transcriptions

# synthesize speech to a playable file
curl -X POST http://localhost:8000/api/v1/tts/synthesize/audio \
    -H 'content-type: application/json' \
    -d '{"input": "আমি বাংলায় কথা বলি।", "voice": "Aditi"}' \
    -o out.wav

# the streaming sockets (voicebot service, port 8100)
python examples/voicebot_client.py asr path/to/audio.wav
python examples/voicebot_client.py tts "আমি ভালো আছি।"
```

---

## Evaluating against ground truth

`scripts/benchmark.py` computes WER/CER for any ASR HTTP endpoint sharing this pipeline's request/response contract, measured against labeled ground truth.

```bash
pip install ".[eval]"
python scripts/download_eval_data.py --data-dir data/eval_fleurs_bn
python scripts/benchmark.py --api-url http://127.0.0.1:8000/api/v1/asr/transcribe \
    --data-dir data/eval_fleurs_bn --output-dir benchmark_output
```

Point `--api-url` at any endpoint sharing the request/response contract below — the gateway route above, LitServe's `/predict` if you reach it directly, or another service entirely.

Requests run concurrently (`--concurrency`, default 10), so the report doubles as a throughput measurement rather than a pure accuracy run. Writes two files to `--output-dir`:

- `predictions.json` — per-utterance reference, hypothesis, WER/CER, latency, any request error.
- `report.txt` — corpus-level WER/CER/MER/WIL (aggregated, not averaged) plus latency p50/p95/p99 and requests/sec at that concurrency.

<details>
<summary>Evaluation dataset details</summary>

[FLEURS](https://huggingface.co/datasets/google/fleurs) (Google's multilingual speech benchmark, 100+ languages) — read-aloud, professionally recorded sentences from the FLoRes machine-translation dataset, so content is naturally-worded but not spontaneous/conversational. This project uses the `bn_in` (Bengali, India locale) config, license CC-BY-4.0.

`download_eval_data.py` pulls the `test` (920 utterances) and `validation` (402 utterances) splits via the `datasets` library — not `train`, which isn't held-out eval data. Audio is 16 kHz mono, 32-bit float WAV, a few seconds to ~15s long, each with a `transcription` (normalized) and `raw_transcription` (original casing/punctuation).

Output layout in `--data-dir`:

- `data.tar` — one archive member per utterance (`fleurs_bn_{split}_{position:04d}.wav`). `benchmark.py` reads audio straight out of it via `tarfile`, no extraction step.
- `ground_truth.json` — tar member name → `{transcription, raw_transcription}`, merged across both splits.

Member names are the loop position, not FLEURS' own `id` field — `id` is **not unique within a split** (`validation` has 402 rows but only 150 distinct ids, which would silently overwrite 252 files if used as the filename).

</details>
