from pydantic import (
    conlist,
    BaseModel,
    Field,
)
from typing import (
    List,
    Optional,
)

from enum import Enum


class UserIntent(str, Enum):
    QUESTION = "question"
    EXPLAIN = "explain"
    HOWTO = "how-to"
    SUMMARIZE = "summarize"
    TRANSLATE = "translate"
    COMPARE = "compare"
    LIST = "list"
    GENERATE = "generate"
    CODE = "code"
    OTHER = "other"


class ContextBlock(BaseModel):
    block_type: str = Field(
        default=...,
        description="The type of block: 'code', 'text', 'table', 'quote', etc.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Programming language if block_type is 'code' (e.g., 'python', 'javascript'). Otherwise null.",
    )
    content: str = Field(
        default=...,
        description="The exact verbatim content of the block, preserving all whitespace, newlines, and special characters.",
    )


class ClarifyingUserIntent(BaseModel):
    """
    Model for clarifying user's intent

    Extracts the relevant information in a structured way without making up details
    or adding unsolicited information/guesses.
    """

    type_of_request: UserIntent = Field(
        default=...,
        description="The type of request provided by the user, using the fine-grained UserIntent enum.",
    )
    core_intent: str = Field(
        default=...,
        description="The user's core intent in imperative form. The main idea of the request.",
    )
    keywords: List[str] = Field(
        default_factory=list,
        min_length=0,
        max_length=15,
        description="High-impact terms directly related to the core intent. Up to 15 terms.",
    )
    style: str = Field(
        default="prose",
        description=(
            "The way in which the output is presented (e.g., prose, bullet points, tables, verses, "
            "step-by-step instructions). Infer from the task if not explicit (e.g., 'list' → bullet points, "
            "'poem' → verse). Default to prose."
        ),
    )
    output_format: str = Field(
        default="markdown",
        description=(
            "The format in which the information is presented (e.g., markdown, plain text, html, json, "
            "python code). If not explicit, infer from the task (e.g., 'write a script' → python, "
            "'create a table' → markdown). Default to markdown."
        ),
    )
    tone: str = Field(
        default="neutral",
        description=(
            "The desired tone of the output (e.g., formal, casual, humorous, sarcastic, technical, simple, "
            "ELI5). Extract from the prompt if mentioned; otherwise neutral."
        ),
    )
    additional_instructions: str = Field(
        default="",
        description=(
            "Any remaining constraints, audience, length, do/don't instructions, role, or specific details "
            "from the original prompt that do not fit elsewhere. Preserve verbatim."
        ),
    )
    context_blocks: List[ContextBlock] = Field(
        default_factory=list,
        description=(
            "All verbatim fenced code blocks, quoted passages, or data snippets provided by the user. "
            "Copy each block exactly, including language markers if present."
        ),
    )
