from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TtsRequest(BaseModel):
    """Text-to-speech request, shaped like the OpenAI speech endpoint."""

    input: str = Field(..., min_length=1, max_length=20_000)
    voice: str = ""
    description: str | None = Field(default=None, max_length=1_000)
    response_format: Literal["wav", "pcm"] = "wav"

    @field_validator("input")
    @classmethod
    def input_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("input must not be blank")
        return value


class TtsResponse(BaseModel):
    taskType: str = "tts"
    audioContent: str = Field(..., description="Base64-encoded mono 16-bit WAV")
    sampleRate: int
    voice: str
    time_taken: float
