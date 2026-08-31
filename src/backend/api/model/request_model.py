from pydantic import BaseModel


class EchoRequest(BaseModel):
    text: str


class LLMRequest(BaseModel):
    prompt: str
    output_structure: dict | None = None
