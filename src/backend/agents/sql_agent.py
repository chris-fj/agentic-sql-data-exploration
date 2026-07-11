"""LangChain agent that answers data questions using SQL tools.

Uses ``langchain.agents.create_agent`` (langgraph-based) to build a
tool-calling agent that iterates between the LLM and three tools:
``run_sql_query``, ``generate_chart``, and ``generate_explanation``.
"""

import base64
import json
import re
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.tools import TOOLS
from backend.utils.llm import get_llm

SYSTEM_PROMPT = """\
You are a data analyst assistant with access to a DuckDB database.

## Database schema

Three tables in the database:

- **transactions** (transaction_id, transaction_date, seller_id, amount_eur)
- **seller** (seller_id, territory_id, seller_name)
- **territory** (territory_id, territory_name)

Relationships:
- transactions.seller_id → seller.seller_id  (one transaction belongs to one seller)
- seller.territory_id → territory.territory_id (each seller is in one territory)

## Your tools

1. **run_sql_query** — Run a SELECT SQL query. Returns JSON rows.
2. **generate_chart** — Run a SQL query and create a chart (bar, line, or pie).
   Returns a base64 PNG data URI.
3. **generate_explanation** — Generate a markdown explanation from results.
   This is your FINAL step.

## Workflow

For every user question:
1. Write and run the appropriate SELECT query using **run_sql_query**.
2. If the user asked for a chart or visualization, call **generate_chart**
   with the same (or a well-suited) SQL query, a descriptive title, and the
   appropriate chart type.
3. Always finish by calling **generate_explanation** with:
   - user_question: the user's original question verbatim
   - query_results: the output from run_sql_query
   - has_chart: true if you generated a chart, false otherwise

## Rules

- ONLY use SELECT statements. Never attempt INSERT, UPDATE, DELETE, DROP, etc.
- Use appropriate JOINs when the question spans multiple tables.
- For date filtering use: transaction_date BETWEEN 'start' AND 'end'.
- Use aggregation (SUM, AVG, COUNT, GROUP BY) for summary questions.
- Limit results to a reasonable number of rows.
- This is a tool aimed to be presented at the leadership team. Keep a formal tone (no emojis, etc...)
"""

# Regex to extract base64 data URIs from tool output
_DATA_URI_RE = re.compile(r"data:image/png;base64,[A-Za-z0-9+/=]+")


def _extract_chart(messages: list) -> str | None:
    """Walk agent messages and extract the first base64 PNG data URI found."""
    for msg in messages:
        content = ""
        if isinstance(msg, ToolMessage):
            content = msg.content or ""
        elif isinstance(msg, AIMessage):
            text = msg.content if isinstance(msg.content, str) else ""
            content = text

        match = _DATA_URI_RE.search(content)
        if match:
            return match.group(0)
    return None


def _extract_explanation(messages: list) -> str:
    """Extract the last AI message as the explanation text.

    Skips tool-call-only messages (where content is empty / a list of
    tool-call dicts).  Returns the last meaningful AI text.
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content.strip()
    return "No explanation was generated."


async def run_sql_agent(
    user_query: str, llm_type: Literal["local", "cloud"] = "cloud"
) -> dict:
    """Run the SQL exploration agent and return results.

    Args:
        user_query: Natural-language question about the data.
        llm_type: ``"cloud"`` (DeepSeek) or ``"local"`` (Ollama).

    Returns:
        A dict with keys ``explanation`` (markdown str) and ``chart``
        (base64 PNG data URI, or ``None`` if no chart was generated).
    """
    llm = get_llm(llm_type)
    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_query)]}
    )

    messages = result.get("messages", [])

    return {
        "explanation": _extract_explanation(messages),
        "chart": _extract_chart(messages),
    }
