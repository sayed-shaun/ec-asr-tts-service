from pydantic import BaseModel, Field


class Language(BaseModel):
    sourceLanguage: str = "bn"


class AsrConfig(BaseModel):
    language: Language = Field(default_factory=Language)
    audioFormat: str | None = "wav"
    samplingRate: int | None = None


class AudioContent(BaseModel):
    audioContent: str = Field(..., description="Base64-encoded audio (wav/flac/ogg/mp3)")


class AsrRequest(BaseModel):
    config: AsrConfig = Field(default_factory=AsrConfig)
    audio: list[AudioContent] = Field(..., min_length=1)


class Output(BaseModel):
    source: str


class AsrResponse(BaseModel):
    taskType: str = "asr"
    output: list[Output]
    time_taken: float


class ErrorResponse(BaseModel):
    detail: str
