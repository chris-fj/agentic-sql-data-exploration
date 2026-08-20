"""Pydantic models for the SQL agent API endpoint."""

from typing import Literal

from pydantic import BaseModel, Field


class SQLAgentRequest(BaseModel):
    """Request body for POST /api/sql-agent."""

    query: str
    llm: Literal["local", "cloud"] = "cloud"


class KeyFindingItem(BaseModel):
    """A single key finding from the data analysis."""

    insight: str
    value: str | None = None


class SQLAgentResponse(BaseModel):
    """Response from the SQL agent (now backed by the StateGraph)."""

    explanation: str = Field(
        description="Full markdown narrative (data questions) or chat response text."
    )
    key_findings: list[KeyFindingItem] = Field(
        default_factory=list,
        description="Key data insights (3-7 items for data questions; empty for chat).",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations (2-5 items for data questions).",
    )
    chart: str | None = Field(
        default=None,
        description="Base64 PNG data URI if a chart was generated, otherwise None.",
    )
