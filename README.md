# Bangla ASR Pipeline

Bengali speech-to-text service built on
[hishab/titu_stt_bn_fastconformer](https://huggingface.co/hishab/titu_stt_bn_fastconformer)
(a NeMo FastConformer-CTC model trained on ~18k hours of Bengali speech),
served with [LitServe](https://github.com/Lightning-AI/LitServe).

## Architecture

```
main.py                        # entrypoint: builds & runs the LitServer (imports ONE model's litapi — see below)
src/
  core/
    config.py                  # pydantic-settings, env-driven (ASR_* vars)
    logging.py                  # loguru sinks (console + rotating file)
  models/
    conformer/                 # one directory per model architecture — self-contained
      engine.py                  # NeMo ASRModel load + transcribe wrapper (this architecture's own)
      litapi.py                   # ASRLitAPI: setup/decode_request/predict/encode_response
  utils/
    audio.py                    # base64 -> mono float32 @ target sample rate (shared across all models)
  api/v1/asr/
    schema.py                   # request/response contract (pydantic) — only thing this dir holds, besides router
    router.py                    # extra FastAPI routes mounted on the LitServer app
tests/
  test_api.py                  # unit tests (engine mocked, no GPU/model needed)
examples/
  client_example.py            # minimal Python client
```

`ASRLitAPI.setup()` loads the NeMo model once per worker process; `predict()`
never touches the network or disk. Everything else (`router.py`) is plain
FastAPI mounted on `server.app` and must **not** call the engine directly,
since worker processes owning the loaded model are separate from the process
serving these extra routes.

### Adding a new model architecture

`src/models/<name>/` is meant to be self-contained: `engine.py` (the
architecture-specific inference wrapper) and `litapi.py` (imports its own
sibling `engine.py`, never another model's). `schema.py`, `router.py`,
`config.py`, and `utils/audio.py` are shared across every model — only touch
those if the new architecture needs a genuinely different request/response
shape. To actually serve the new model, point `main.py`'s
`from src.models.<name>.litapi import ASRLitAPI` at it — only one model
runs per `main.py` process at a time; there's no runtime model switch.

## Request / response contract

`POST /predict`

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

```json
{
  "taskType": "asr",
  "output": [{"source": "transcribed bengali text"}],
  "time_taken": 0.42
}
```

Multiple items in `audio` are transcribed as a batch (order preserved in
`output`). Audio is decoded and resampled to 16 kHz mono automatically
(`librosa`), so wav/flac/ogg/mp3 all work as long as `ffmpeg`/`libsndfile`
are available (see Docker image).

Other endpoints:
- `GET /health` — LitServe healthcheck (also checks the model is loaded)
- `GET /info` — LitServe worker/build info
- `GET /v1/asr/info` — static model metadata
- `POST /v1/asr/transcribe/file` — multipart file upload (`file=@audio.wav`), for
  testing from Swagger UI's "Choose File" button; base64-encodes the upload and
  re-dispatches to `POST /predict` internally — same model path, no duplicated logic
- `GET /docs` — interactive Swagger UI

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
```

> The model is licensed **CC-BY-NC-4.0** (non-commercial). `nemo_toolkit[asr]`
> is a heavy install (torch, pytorch-lightning, hydra, etc.) — expect several
> minutes and a few GB of disk/download.

## Run

```bash
python main.py
```

The first request triggers the Hugging Face/NeMo download of the model
checkpoint (cached under `~/.cache/huggingface` or `$HF_HOME`) — startup will
be slow the first time.

## Docker

```bash
docker compose up --build
```

Uncomment the `deploy.resources` block in `docker-compose.yml` for GPU
acceleration (requires `nvidia-container-toolkit` on the host).

## Configuration

All settings are env vars prefixed `ASR_` (see `.env.example`), e.g.
`ASR_ACCELERATOR`, `ASR_TRANSCRIBE_BATCH_SIZE`, `ASR_PORT`, `ASR_MODEL_NAME`.

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

Tests mock `ASREngine` — they don't download or run the actual NeMo model.

## Example client

```bash
python examples/client_example.py path/to/audio.wav
```
