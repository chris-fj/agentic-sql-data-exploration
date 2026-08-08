"""FastAPI router for the SQL exploration agent."""
import logging
from fastapi import APIRouter

from backend.agents.sql_agent import run_sql_agent
from backend.api.model.sql_agent_models import SQLAgentRequest, SQLAgentResponse

router = APIRouter()

logger = logging.getLogger(__name__)

@router.post("/sql-agent", response_model=SQLAgentResponse)
async def call_sql_agent(request: SQLAgentRequest) -> SQLAgentResponse:
    """Answer a natural-language data question using the SQL agent.

    The agent writes and runs SELECT queries against the DuckDB database,
    optionally generates charts, and produces a markdown explanation.
    """
    logger.info("Inside SQL Agent with parameters: %s", str(request.model_dump_json()))
    result = await run_sql_agent(request.query, request.llm)
    return SQLAgentResponse(**result)
