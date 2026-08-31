"""FastAPI router for the SQL exploration agent."""

import logging

from fastapi import APIRouter

from backend.agents.sql_agent import run_sql_agent
from backend.api.model.sql_agent_models import (
    SQLAgentRequest,
    SQLAgentResponse,
)

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/sql-agent", response_model=SQLAgentResponse)
async def call_sql_agent(request: SQLAgentRequest) -> SQLAgentResponse:
    """Answer a natural-language question using the SQL agent StateGraph.

    The graph classifies the intent, routes data questions through a
    schema-aware SQL pipeline (generate → execute → optional chart →
    structured report), and handles general chat with a plain LLM response.
    """
    logger.info(
        "SQL agent request — query=%s",
        request.query[:200],
    )
    result = await run_sql_agent(request.query)
    return SQLAgentResponse(**result)
