import time

import litserve as ls
from fastapi import HTTPException

from src.core.config import settings
from src.engine.asr_engine import ASREngine
from src.engine.audio_utils import decode_base64_audio

from .schema import AsrRequest, AsrResponse, Output


class ASRLitAPI(ls.LitAPI):
    """Serves hishab/titu_stt_bn_fastconformer (NeMo FastConformer CTC) over HTTP."""

    def setup(self, device: str) -> None:
        self.engine = ASREngine(
            model_name=settings.model_name,
            device=device,
            max_segment_seconds=settings.max_segment_seconds,
        )
        self.engine.load()

    def decode_request(self, request: AsrRequest) -> dict:
        sample_rate = request.config.samplingRate or settings.sample_rate
        try:
            audios = [decode_base64_audio(item.audioContent, target_sr=sample_rate) for item in request.audio]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"audios": audios, "sample_rate": sample_rate, "received_at": time.time()}

    def predict(self, x: dict) -> dict:
        transcriptions = self.engine.transcribe(
            x["audios"], batch_size=settings.transcribe_batch_size, sample_rate=x["sample_rate"]
        )
        return {"transcriptions": transcriptions, "received_at": x["received_at"]}

    def encode_response(self, output: dict) -> AsrResponse:
        time_taken = time.time() - output["received_at"]
        return AsrResponse(
            output=[Output(source=text) for text in output["transcriptions"]],
            time_taken=time_taken,
        )


def build_lit_api() -> ASRLitAPI:
    # LitServe request-level batching (max_batch_size>1) merges multiple concurrent
    # HTTP requests into one `predict()` call and requires batch()/unbatch() to split
    # them back apart; we don't implement that. Batching instead happens one level
    # down, inside a single request's `audio` list, via NeMo's own
    # transcribe(batch_size=...) — see settings.transcribe_batch_size.
    return ASRLitAPI(max_batch_size=1, api_path=settings.api_path)
