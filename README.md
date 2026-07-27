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
  api/v1/live_cc/
    router.py                    # WebSocket live-captioning endpoint (see below)
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

## live-cc: live closed captioning

`WS /v1/live-cc/ws` — same model as batch requests, streamed. Client sends
raw 16-bit PCM mono audio at `ASR_LIVE_CC_INPUT_SAMPLE_RATE` over the socket
in any chunk size. Two kinds of caption come back:

```json
{"text": "in-progress guess so far", "is_final": false}
{"text": "committed bengali text", "is_final": true}
```

Every `ASR_LIVE_CC_INTERIM_INTERVAL_SECONDS` of in-progress audio, the whole
in-progress chunk is re-transcribed and pushed as an interim caption
(`is_final: false`) — replace the previous interim with each new one, don't
append. Every `ASR_LIVE_CC_CHUNK_SECONDS` the chunk is finalized
(`is_final: true`), appended permanently, and the buffer resets. This is what
makes captions feel like they're live-updating rather than arriving in one
lump every `ASR_LIVE_CC_CHUNK_SECONDS`.

Implementation: like `transcribe_file`, this route never touches the model
directly (it runs in the main process, not a LitServe worker) — each
transcription (interim or final) is wrapped as a WAV and re-dispatched
internally to `POST /predict`, reusing the exact same model path. This means
interim updates cost real, redundant GPU work: each one re-transcribes from
the start of the current chunk, so total compute per chunk scales up
(roughly `chunk_seconds / interim_seconds`) — tune
`ASR_LIVE_CC_INTERIM_INTERVAL_SECONDS` down if you're running multiple
concurrent live-cc connections on a memory/compute-constrained GPU, or set it
to `0` to disable interim updates entirely. `ASR_LIVE_CC_CHUNK_SECONDS` is
still a latency/word-splitting tradeoff for the *final* text: chunks are hard
cuts with no overlap, so shorter means final captions commit faster but words
at chunk boundaries can get cut, longer is the reverse. This is buffered
pseudo-streaming, not true incremental decoding — there's no cache-aware
streaming state carried between chunks, so each transcription is independent
and starts from scratch.

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

Open `http://localhost:8000/static/index.html` for a manual test page:
upload a file to hit `/v1/asr/transcribe/file`, or click Start on the live
captioning section to stream your mic to `/v1/live-cc/ws` (mounted at
`/static`, not `/` — LitServe already owns that route).

## Docker

```bash
docker compose up --build
```

GPU acceleration is on by default (`deploy.resources` in `docker-compose.yml`,
`ASR_ACCELERATOR=cuda`) and requires `nvidia-container-toolkit` on the host;
remove that block and set `ASR_ACCELERATOR=cpu` for a CPU-only host.

## Configuration

All settings are env vars prefixed `ASR_` (see `.env.example`), e.g.
`ASR_ACCELERATOR`, `ASR_TRANSCRIBE_BATCH_SIZE`, `ASR_PORT`, `ASR_MODEL_NAME`.

## Performance

Numbers below are measured on an RTX 2050 with a 3s clip, `WORKERS_PER_DEVICE=2`.
Re-measure on your own hardware before treating any of them as a target.

| | |
|---|---|
| Steady-state latency | ~30 ms (RTF ~0.007, ≈140x realtime) |
| First request (cold) | ~69 ms — was 744 ms before warmup |
| Throughput | 28 req/s at concurrency 1, ~46 req/s at 2+ |

**Startup warmup.** The first request into a fresh worker used to pay two
one-off initialization costs: ~920 ms for librosa/soundfile/soxr lazy imports
on the first `decode_base64_audio()`, and ~279 ms of cuDNN/CUDA warmup on the
first `model.transcribe()`. Both are now burned during `setup()`
(`ASREngine.load()` warms the model, `warm_audio_decoder()` warms the decode
path), best-effort so a failed warmup can't stop a worker coming up. Don't
remove them — the cost doesn't disappear, it just moves onto your first user.

**Throughput plateaus at ~46 req/s** because both workers saturate;
`ASR_WORKERS_PER_DEVICE` is the lever, bounded by GPU memory since each worker
holds a full model copy.

**Measured, deliberately not done:**

- *FP16* — no win here (18.6 ms vs 18.1 ms at 1s audio). The model is small
  enough to be latency-bound rather than compute-bound, so tensor cores don't
  buy anything. Might change on a bigger GPU with longer audio.
- *Avoiding the temp file in `decode_base64_audio`* — costs 0.11 ms, versus
  ~20 ms of inference. Not worth reintroducing the BytesIO path that broke
  WebM/Opus decoding.
- *Avoiding the internal ASGI re-dispatch* (`transcribe/file`, live-cc) —
  costs 5.8 ms, cheap for keeping the model in exactly one place.

**Audio format matters on hot paths.** WebM/Opus decodes in ~31 ms because it
shells out to ffmpeg; WAV decodes in ~0.3 ms via soundfile. Fine for the
record-voice button (once per recording), but send raw PCM or WAV through
live-cc's high-frequency path, not WebM.

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
