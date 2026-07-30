# Bangla ASR Pipeline

Bengali speech-to-text service. Model: [hishab/titu_stt_bn_fastconformer](https://huggingface.co/hishab/titu_stt_bn_fastconformer) (NeMo FastConformer-CTC). Served with [LitServe](https://github.com/Lightning-AI/LitServe).

## Structure

```
.
├── main.py                        # entrypoint — builds & runs the LitServer
├── src/
│   ├── core/
│   │   ├── config.py              # env-driven settings (ASR_*)
│   │   └── logging.py             # loguru sinks
│   ├── models/conformer/
│   │   ├── engine.py              # NeMo model load + transcribe
│   │   └── litapi.py              # LitServe API: setup/decode/predict/encode
│   ├── utils/
│   │   └── audio.py               # base64 -> waveform decode/resample
│   └── api/v1/
│       ├── asr/router.py          # /v1/asr/* routes
│       ├── asr/schema.py          # request/response models
│       └── live_cc/router.py      # /v1/live-cc/ws
├── static/index.html              # manual test page
├── tests/test_api.py              # unit tests (model mocked)
├── examples/client_example.py     # minimal Python client
└── benchmark.py                   # latency/quality comparison script
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Model license: **CC-BY-NC-4.0** (non-commercial). `nemo_toolkit[asr]` is a heavy install — several minutes, several GB.

## Run

```bash
python main.py
```

First request downloads the checkpoint (cached under `$HF_HOME` / `~/.cache/huggingface`) — slow once, fast after.

Test page: `http://localhost:8000/static/index.html`

## Docker

```bash
docker compose up --build
```

GPU is on by default (`ASR_ACCELERATOR=cuda`, requires `nvidia-container-toolkit`). For CPU-only, remove the `deploy.resources` block in `docker-compose.yml` and set `ASR_ACCELERATOR=cpu`.

## Test

```bash
pip install -e ".[dev]"
pytest -q
```

## Endpoints

| | |
|---|---|
| `POST {ASR_API_PATH}` | main inference — JSON, base64 audio |
| `POST /v1/asr/transcribe/file` | multipart file upload |
| `WS /v1/live-cc/ws` | streamed captions, raw 16-bit PCM |
| `GET /v1/asr/info` | model metadata |
| `GET /health` | healthcheck |
| `GET /docs` | Swagger UI |

`ASR_API_PATH` defaults to `/api/v1/asr/transcribe` — check `.env` / `.env.example`, not this file, for the current value.

### Inference request

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

## Config

All settings are env vars prefixed `ASR_`. See `.env.example` for the full list and defaults.

## Client examples

```bash
python examples/client_example.py path/to/audio.wav
python benchmark.py --data-dir data
```

Both currently POST to `/predict`, not `ASR_API_PATH` — broken until updated to match.
