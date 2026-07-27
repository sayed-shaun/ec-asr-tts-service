import time

import litserve as ls
from fastapi import HTTPException

from src.api.v1.asr.schema import AsrRequest, AsrResponse, Output
from src.core.config import settings
from src.models.conformer.engine import ASREngine
from src.utils.audio import decode_base64_audio


class ASRLitAPI(ls.LitAPI):
    """Serves hishab/titu_stt_bn_fastconformer (NeMo FastConformer CTC) over HTTP."""

    def setup(self, device: str) -> None:
        self.engine = ASREngine(
            model_name=settings.MODEL_NAME,
            device=device,
            max_segment_seconds=settings.MAX_SEGMENT_SECONDS,
        )
        self.engine.load()

    def decode_request(self, request: AsrRequest) -> dict:
        sample_rate = request.config.samplingRate or settings.SAMPLE_RATE
        try:
            audios = [
                decode_base64_audio(item.audioContent, target_sr=sample_rate)
                for item in request.audio
            ]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
        time_taken = time.time() - output["received_at"]
        return AsrResponse(
            output=[Output(source=text) for text in output["transcriptions"]],
            time_taken=time_taken,
        )
