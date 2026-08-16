import time

import litserve as ls
from fastapi import HTTPException, status
from loguru import logger

from src.api.v1.asr.schema import AsrRequest, AsrResponse, Output
from src.core.config import settings
from src.litserver.engine.base import BaseEngine
from src.litserver.engine.conformer import ConformerEngine
from src.litserver.engine.wav2vec2 import Wav2Vec2Engine
from src.litserver.engine.whisper import WhisperEngine
from src.litserver.engine.zipformer import ZipformerEngine
from src.utils.audio import decode_base64_audio, warm_audio_decoder
from src.utils.bn_text_repair import repair_stray_vowel_signs
from src.utils.itn import bengali_numerals_to_digits

# Bare numerals below this stay spelled out; measured sweep over the 1322-clip
# FLEURS benchmark put the corpus-WER optimum here (0.1541 -> 0.1370, and
# 0.2811 -> 0.1898 on digit-bearing references). See src/utils/itn.py.
ITN_MIN_VALUE = 10


class ASRLitAPI(ls.LitAPI):
    """Serves Bengali ASR over HTTP via a pluggable engine (settings.ENGINE
    selects ConformerEngine, Wav2Vec2Engine, WhisperEngine, or
    ZipformerEngine).
    """

    def setup(self, device: str) -> None:
        """Load the model engine and warm the audio decoder.

        The engine warms the model itself; warm_audio_decoder warms the
        audio-decode half of the request path, which is the larger of the two
        one-off costs.
        """
        self.engine = self.build_engine(device)
        self.engine.load(sample_rate=settings.SAMPLE_RATE)

        try:
            warm_audio_decoder(settings.SAMPLE_RATE)
            logger.info("Audio decoder warmed up")
        except Exception as exc:
            logger.warning(f"Audio decoder warmup failed (continuing anyway): {exc}")

    @staticmethod
    def build_engine(device: str) -> BaseEngine:
        if settings.ENGINE == "wav2vec2":
            return Wav2Vec2Engine(
                model_name=settings.WAV2VEC2_MODEL_NAME,
                device=device,
                max_segment_seconds=settings.MAX_SEGMENT_SECONDS,
            )
        if settings.ENGINE == "whisper":
            return WhisperEngine(
                model_name=settings.WHISPER_MODEL_NAME,
                device=device,
                max_segment_seconds=settings.MAX_SEGMENT_SECONDS,
            )
        if settings.ENGINE == "zipformer":
            return ZipformerEngine(model_name=settings.ZIPFORMER_MODEL_NAME)
        return ConformerEngine(
            model_name=settings.CONFORMER_MODEL_NAME,
            device=device,
            max_segment_seconds=settings.MAX_SEGMENT_SECONDS,
        )

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

        Applied here rather than in the engine so the engine stays purely about
        recognition, and so it lands after per-segment texts are joined — a
        number split across a segment boundary would otherwise never be seen
        as one numeral run.
        """
        time_taken = time.time() - output["received_at"]
        texts = output["transcriptions"]
        if settings.ENGINE == "wav2vec2":
            texts = [repair_stray_vowel_signs(text) for text in texts]
        if settings.ITN_ENABLED:
            texts = [
                bengali_numerals_to_digits(text, min_value=ITN_MIN_VALUE)
                for text in texts
            ]
        return AsrResponse(
            output=[Output(source=text) for text in texts],
            time_taken=time_taken,
        )
