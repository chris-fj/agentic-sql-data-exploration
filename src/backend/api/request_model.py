from typing import Literal

from pydantic import BaseModel


class EchoRequest(BaseModel):
    text: str


class LLMRequest(BaseModel):
    llm: Literal["local", "cloud"] = "cloud"
    prompt: str
    structure_output: dict | None = None
