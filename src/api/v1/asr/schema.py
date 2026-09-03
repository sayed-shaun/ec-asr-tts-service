from typing import List

from pydantic import BaseModel


class Language(BaseModel):
    sourceLanguage: str


class Config(BaseModel):
    language: Language


class AudioContent(BaseModel):
    """A single base64-encoded audio clip."""

    audioContent: str


class AsrRequest(BaseModel):
    """Mirrors the Java service's request structure — nested `config.language`
    instead of a flat field is part of that contract, not a design choice made here.
    """

    config: Config
    audio: List[AudioContent]


class Output(BaseModel):
    """A single transcription result."""

    source: str


class AsrResponse(BaseModel):
    """Mirrors the Java service's response structure."""

    taskType: str
    output: List[Output]
    time_taken: float
