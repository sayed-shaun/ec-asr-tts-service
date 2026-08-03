from pydantic import BaseModel


class EmbedRequest(BaseModel):
    audio_content: str
