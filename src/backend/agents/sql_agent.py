"""LangGraph StateGraph agent that answers data questions using SQL.

Replaces the previous ``langchain.agents.create_agent`` approach with a
custom ``StateGraph`` that encodes the workflow as deterministic nodes
with structured LLM output at each LLM-call step:

    clarify_intent → route → [analyze_schema → generate_sql → execute_sql
                             → (optional) generate_chart → generate_report]
                          → [chat_response]

The graph is compiled once at import time and reused across requests.
"""

import logging

from backend.agents.graph import build_graph
from backend.agents.state import (
    AgentState,
    Report,
)

logger = logging.getLogger(__name__)

# Compile once — the graph is stateless (no checkpointer), so the same
# compiled instance handles every request safely.
_graph = build_graph()


async def run_sql_agent(user_query: str) -> dict:
    """Run the SQL exploration agent and return structured results.

    Parameters
    ----------
    user_query : str
        Natural-language question about the data (or any chat prompt).

    Returns
    -------
    dict
        A dict with keys matching the structured output:

        - For data questions: ``explanation`` (markdown narrative),
          ``key_findings``, ``recommendations``, ``chart`` (base64 PNG or None).
        - For general chat: ``explanation`` (chat response text),
          ``key_findings`` (empty), ``recommendations`` (empty), ``chart`` (None).
        - For errors: ``explanation`` contains the error message.
    """
    initial: AgentState = {
        "user_query": user_query,
        "messages": [],
    }

    logger.info("Running SQL agent graph for query: %s", user_query[:120])
    result = await _graph.ainvoke(initial)

    # -- Extract structured output --------------------------------------------
    report: Report | None = result.get("report")
    final_response: str | None = result.get("final_response")
    final_error: str | None = result.get("final_error")

    if final_error and not report:
        return {
            "explanation": f"**Error:** {final_error}",
            "key_findings": [],
            "recommendations": [],
            "chart": None,
        }

    if report:
        logger.info(
            "Returning report: summary=%d chars, %d findings",
            len(report.executive_summary),
            len(report.key_findings),
        )
        return {
            "explanation": report.narrative,
            "key_findings": [
                {"insight": f.insight, "value": f.value} for f in report.key_findings
            ],
            "recommendations": report.recommendations,
            "chart": result.get("chart_image"),
        }

    # Chat response (non-data question)
    chat_text = final_response or "No response was generated."
    return {
        "explanation": chat_text,
        "key_findings": [],
        "recommendations": [],
        "chart": None,
    }
