from pydantic import BaseModel


class EchoResponse(BaseModel):
    echo: str


class LLMResponse(BaseModel):
    output: str
