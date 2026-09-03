import base64
import time

import litserve as ls
from fastapi import HTTPException, status
from loguru import logger

from src.api.v1.asr.schema import AsrRequest, AsrResponse, Output
from src.api.v1.tts.schema import TtsRequest, TtsResponse
from src.core.config import settings
from src.litserver.base import BaseASREngine, BaseTTSEngine
from src.litserver.parler import engine as parler
from src.litserver.zipformer import engine as zipformer
from src.utils.audio import decode_base64_audio, warm_audio_decoder, wav_bytes
from src.utils.itn import bengali_numerals_to_digits

TTS_API_PATH = "/synthesize"

ITN_MIN_VALUE = 10
"""Bare numerals below this stay spelled out. Measured corpus-WER optimum over
the 1322-clip FLEURS benchmark (0.1541 -> 0.1370; 0.2811 -> 0.1898 on
digit-bearing references)."""


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
        sample_rate = request.config.samplingRate or settings.SAMPLE_RATE
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
            output=[Output(source=text) for text in texts],
            time_taken=time_taken,
        )


class TTSLitAPI(ls.LitAPI):
    """Serves Bengali text-to-speech over HTTP.

    Runs in the same LitServe process as ASRLitAPI, on its own api_path and
    workers but sharing accelerator/devices/workers_per_device, so both
    checkpoints are resident on the same GPU. TTS_ENABLED=false drops it.
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

    def predict(self, x: dict) -> dict:
        """Hand the whole request to the engine.

        Whether that needs splitting into clauses is the engine's business
        (see BaseTTSEngine.speak), so nothing here is model-specific.
        """
        try:
            audio = self.engine.speak(x["text"], x["voice"], x["description"])
        except ValueError as exc:
            logger.warning(f"TTS request rejected: {exc}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        return {"audio": audio, "voice": x["voice"], "received_at": x["received_at"]}

    def encode_response(self, output: dict) -> TtsResponse:
        audio = output["audio"]
        wav = wav_bytes(audio.samples, audio.sample_rate)
        return TtsResponse(
            audioContent=base64.b64encode(wav).decode("utf-8"),
            sampleRate=audio.sample_rate,
            voice=output["voice"],
            time_taken=time.time() - output["received_at"],
        )
