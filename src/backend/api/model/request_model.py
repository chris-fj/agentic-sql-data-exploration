from typing import Literal

from pydantic import BaseModel


class EchoRequest(BaseModel):
    text: str


class LLMRequest(BaseModel):
    llm: Literal["local", "cloud"] = "cloud"
    prompt: str
    output_structure: dict | None = None
