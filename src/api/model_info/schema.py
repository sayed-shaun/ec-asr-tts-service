from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    model_name: str
    num_parameters: int
