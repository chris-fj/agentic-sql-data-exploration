"""Pydantic models for the SQL agent API endpoint."""

from typing import Literal

from pydantic import BaseModel


class SQLAgentRequest(BaseModel):
    """Request body for POST /api/sql-agent."""

    query: str
    llm: Literal["local", "cloud"] = "cloud"


class SQLAgentResponse(BaseModel):
    """Response from the SQL agent."""

    explanation: str
    chart: str | None = None
