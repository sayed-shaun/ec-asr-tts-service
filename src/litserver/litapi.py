import base64
import json
import time
from collections.abc import Iterator

import litserve as ls
from fastapi import HTTPException, status
from loguru import logger

from src.api.v1.asr.schema import AsrRequest, AsrResponse, Output
from src.api.v1.tts.schema import TtsRequest
from src.core.config import settings
from src.litserver.base import BaseASREngine, BaseTTSEngine
from src.litserver.parler import engine as parler
from src.litserver.zipformer import engine as zipformer
from src.utils.audio import decode_base64_audio, warm_audio_decoder
from src.utils.itn import bengali_numerals_to_digits

TTS_API_PATH = "/synthesize"
ITN_MIN_VALUE = 10

class ASRLitAPI(ls.LitAPI):
    """Serves Bengali ASR over HTTP via the Zipformer engine."""

    def setup(self, device: str) -> None:
        """Load the model engine and warm the audio decoder."""
        self.engine = self.build_engine(device)
        self.engine.load(sample_rate=settings.SAMPLE_RATE)

        try:
            warm_audio_decoder(settings.SAMPLE_RATE)
            logger.info("Audio decoder warmed up")
        except Exception as exc:
            logger.warning(f"Audio decoder warmup failed (continuing anyway): {exc}")

    @staticmethod
    def build_engine(device: str) -> BaseASREngine:
        return zipformer.build(device)

    def decode_request(self, request: AsrRequest) -> dict:
        sample_rate = settings.SAMPLE_RATE
        try:
            audios = [
                decode_base64_audio(item.audioContent, target_sr=sample_rate)
                for item in request.audio
            ]
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        return {
            "audios": audios,
            "sample_rate": sample_rate,
            "received_at": time.time(),
        }

    def predict(self, x: dict) -> dict:
        transcriptions = self.engine.transcribe(
            x["audios"],
            batch_size=settings.TRANSCRIBE_BATCH_SIZE,
            sample_rate=x["sample_rate"],
        )
        return {"transcriptions": transcriptions, "received_at": x["received_at"]}

    def encode_response(self, output: dict) -> AsrResponse:
        """Serialize transcripts, rewriting spelled-out numbers as digits.

        ITN is a property of Bengali transcripts rather than of the model,
        so it stays here; a quirk of the checkpoint belongs to its package.
        """
        time_taken = time.time() - output["received_at"]
        texts = output["transcriptions"]
        if settings.ITN_ENABLED:
            texts = [
                bengali_numerals_to_digits(text, min_value=ITN_MIN_VALUE)
                for text in texts
            ]
        return AsrResponse(
            taskType="asr",
            output=[Output(source=text) for text in texts],
            time_taken=time_taken,
        )


class TTSLitAPI(ls.LitAPI):
    """Serves Bengali text-to-speech over HTTP, one clause at a time.

    Runs in the same LitServe process as ASRLitAPI, on its own api_path and
    workers but sharing accelerator/devices/workers_per_device, so both
    checkpoints are resident on the same GPU. TTS_ENABLED=false drops it.

    Always streams, even for callers that want one buffer. Parler is
    autoregressive, so a whole reply is only finished when its last clause is;
    emitting each clause as it lands is the difference between first audio
    after the reply and first audio after the first clause. The gateway
    reassembles the stream for /v1/audio/speech without `stream`, so the
    whole-buffer contract is unchanged and only one worker pool exists.

    Chunks travel as newline-delimited JSON carrying raw PCM, not WAV: WAV
    headers cannot be concatenated, and NDJSON keeps the per-chunk metadata
    (index, sample rate) that a reassembling caller needs.
    """

    def setup(self, device: str) -> None:
        self.engine = self.build_engine(device)
        self.engine.load()

    @staticmethod
    def build_engine(device: str) -> BaseTTSEngine:
        return parler.build(device)

    def decode_request(self, request: TtsRequest) -> dict:
        return {
            "text": request.input,
            "voice": request.voice or settings.TTS_VOICE,
            "description": request.description,
            "received_at": time.time(),
        }

    def predict(self, x: dict) -> Iterator[dict]:
        """Validate, then hand the whole request to the engine as a stream.

        Whether that needs splitting into clauses is the engine's business
        (see BaseTTSEngine.stream_playable), so nothing here is model-specific.

        Must contain `yield` rather than return a generator built elsewhere:
        LitServe refuses to start a stream=True LitAPI whose predict is not
        itself a generator function. The voice is therefore checked on first
        consumption instead of at call time -- still before any chunk is
        emitted, so a bad one cannot truncate a stream already carrying a 200,
        and the gateway rejects it earlier anyway.
        """
        voice = x["voice"]
        if voice not in self.engine.voices:
            logger.warning(f"TTS request rejected: unknown voice: {voice}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"unknown voice: {voice}",
            )
        try:
            stream = self.engine.stream_playable(
                x["text"], x["voice"], x["description"]
            )
            for index, audio in enumerate(stream):
                yield {
                    "audio": audio,
                    "index": index,
                    "voice": x["voice"],
                    "received_at": x["received_at"],
                }
        except ValueError as exc:
            logger.warning(f"TTS request rejected: {exc}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc

    def encode_response(self, outputs: Iterator[dict]):
        """Serialize each chunk as one NDJSON line.

        Deliberately unannotated: LitServe derives the route's FastAPI
        response model from this return type, and FastAPI rejects
        Iterator[str] as a Pydantic field ("Invalid args for response
        field!"), refusing to register the endpoint at all.

        time_taken is elapsed-so-far rather than a total, so the first line
        reports time to first audio -- the number this streaming path exists
        to reduce -- and the last still reports the whole request.
        """
        for output in outputs:
            audio = output["audio"]
            yield json.dumps(
                {
                    "index": output["index"],
                    "audioContent": base64.b64encode(audio.pcm_s16le()).decode(
                        "utf-8"
                    ),
                    "sampleRate": audio.sample_rate,
                    "voice": output["voice"],
                    "time_taken": time.time() - output["received_at"],
                }
            ) + "\n"
