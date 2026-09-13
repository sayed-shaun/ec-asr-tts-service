from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TtsRequest(BaseModel):
    """Text-to-speech request, shaped like the OpenAI speech endpoint."""

    input: str = Field(..., min_length=1, max_length=20_000)
    voice: str = ""
    description: str | None = Field(default=None, max_length=1_000)
    tag: str | None = Field(default=None, max_length=200)
    """Mark this reply as one worth keeping: send the knowledge-base tag it
    came from, and omit it for anything generated per request.

    Only tagged replies are cached. A generated answer is new wording every
    time, so storing it fills a bounded cache with entries nothing will ever
    ask for again, evicting the canned answers that are asked constantly.

    The tag is not the cache key -- the text is -- so it is safe to send even
    when the answer behind a tag is edited upstream: the text changes, the
    key changes, and the old audio is simply never served again. Sending it
    is a hint about reuse, never an assertion about wording.
    """

    response_format: Literal["wav", "pcm"] = "wav"
    stream: bool = False
    """Send each clause as soon as it is synthesized instead of the whole
    reply at once. Requires response_format="pcm": a WAV header states the
    total length, which is not known until the last clause is done."""

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
