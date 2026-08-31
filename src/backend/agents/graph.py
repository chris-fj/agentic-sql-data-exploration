"""LangGraph StateGraph for the SQL exploration agent.

Replaces the single ``create_agent`` call with a deterministic pipeline of
typed nodes, each using structured LLM output where applicable.
"""

import json
import logging

import duckdb
from jinja2 import Template
from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from backend.agents.state import (
    _DATA_QUESTION_INTENTS,
    AgentState,
    ClarifyingUserIntent,
    Report,
    SQLQuery,
    user_wants_chart,
)
from backend.middleware.timing import timed_node
from backend.tools import (
    build_chart_image,
    estimate_query_cost,
    execute_query,
    get_schema_info,
    validate_sql_ast,
    validate_sql_query,
)
from backend.tools.sql_tool import DB_PATH
from backend.utils.llm import get_llm

logger = logging.getLogger(__name__)

MAX_VALIDATION_RETRIES = 3

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLARIFY_PROMPT = """\
You are about to receive a prompt that the user provided to send into an LLM.
Your task is to analyze it and improve it, so that we keep all details provided
by the user, but in a way that's more structured, so it's easier to understand
by the LLM.

These are the rules guiding your operation procedure:

- You are emphatically forbidden to add additional details not explicitly backed
  by the user's prompt.
- Any source of ambiguity must be left as is, don't guess user's intent.
- Simplify the user's request, eliminating redundancies but keep all relevant
  details.
- Classify the request into exactly one of these intents:
    * data_question: the user wants a fact, value, or result found in the database
      (e.g. "What were sales last month?", "How many transactions?").
    * compare: the user wants a comparison between groups or time periods
      (e.g. "Compare revenue by territory", "Q1 vs Q2").
    * list: the user wants a ranked or enumerated set of rows
      (e.g. "List the top 10 customers").
    * summarize: the user wants an aggregate summary or KPIs
      (e.g. "Summarize transaction trends", "Total sales by region").
    * trend: the user wants a time-series view
      (e.g. "Show monthly sales trend").
    * other: anything that is not a data question answerable by querying the
      database (e.g. greetings, explanations, how-to, out-of-scope requests).
  Only choose a data intent when the answer must come from a SQL query.
- If the user provides a fenced code block (triple backticks), indented code, or
  any long verbatim passage (quote, table, etc.), copy it exactly into
  context_blocks with the correct block_type and, for code, the language. Do not
  alter the content. The core_intent should reference the block (e.g., "Review
  the following code"), but the block itself must be stored verbatim.
- Extract the core intent of the prompt in the imperative form ("Do X, Research
  Y, Generate Z, Explain W")
- Extract the keywords, high impact terms that are directly related to the core
  intent.
- If the user does not explicitly state a style, output format, or tone, infer
  the most natural one from the task. For example: "list the top 10" → bullet
  points (style) and markdown; "write a haiku" → verse; "give me a JSON" → json
  format; "explain like I'm 5" → simple, ELI5 tone. Only default to prose /
  markdown / neutral when no natural fit exists.
- Extract the output format. This is the format in which the information is
  presented, for example, markdown, code block, plain text, html, etc...). It
  must correspond to a file format or have an extension. If possible, infer the
  format according to the task, and if unclear, default to markdown.
- Any constraint that does not fit into the other fields (e.g., length,
  audience, "do not mention X", "act as a historian") must be placed verbatim in
  additional_instructions. Do not paraphrase or omit any detail.

Produce the results without adding additional details not backed by the prompt
itself. Your ultimate goal is to parse the request so that it's interpreted by
another LLM in the best possible way, to yield the most relevant results.

The user prompt can be found below:
"""

CHAT_TEMPLATE = Template("""\
{{ core_intent }}\
{%- if additional_instructions %}

Additional instructions: {{ additional_instructions }}
{%- endif %}
{%- if tone != 'neutral' %}

Tone: {{ tone }}
{%- endif %}
{%- if style != 'prose' %}

Presentation style: {{ style }}
{%- endif %}

Respond in {{ output_format }} format.
{%- for block in context_blocks %}

{% if block.block_type == 'code' -%}
```{{ block.language or '' }}
{{ block.content }}
```
{%- else -%}
{{ block.content }}
{%- endif %}
{%- endfor %}""")

SQL_GENERATION_PROMPT = """\
You are a data analyst with access to a DuckDB database. Write a SQL query
to answer the user's question.

## Database schema

{schema_info}

## User question

{user_query}

## Instructions

- Write a SELECT-only query. Never use INSERT, UPDATE, DELETE, DROP, etc.
- Never produce multi-statement queries. A single SELECT or WITH … SELECT only.
- Never name a CTE after a SQL command keyword (DELETE, DROP, INSERT, UPDATE,
  ALTER, CREATE, TRUNCATE, GRANT, REVOKE). Use descriptive names like
  `filtered_orders` or `recent_transactions`.
- Never query system tables or metadata functions (information_schema,
  duckdb_tables(), PRAGMA, SHOW, DESCRIBE, etc.). The schema is already
  provided above.
- Always include a LIMIT clause unless using aggregation (GROUP BY, COUNT,
  SUM, AVG). Limit to at most 100 rows for raw result sets.
- Use appropriate JOINs when the question spans multiple tables.
- For date filtering use: transaction_date BETWEEN 'start' AND 'end'.
- Use aggregation (SUM, AVG, COUNT, GROUP BY) for summary questions.
- Return the query and a brief explanation of what it does.
"""

REPORT_PROMPT = """\
You are a data analyst preparing a report for the leadership team.
Write a structured report based on the SQL query results below.

## Original user question

{user_query}

## Query results

Columns: {columns}
Rows (JSON):
{rows}
Total rows returned: {row_count}

## Chart

{chart_status}

## Instructions

1. Write an executive summary (1 paragraph) that directly answers the user's
   question.
2. Extract 3-7 key findings from the data. Each finding should have an
   "insight" and optionally a "value".
3. Provide 2-5 actionable recommendations based on the findings.
4. Write a full markdown narrative suitable for the leadership team. Use
   markdown formatting (headings, bold, bullet points, tables) where it
   improves readability. Include context, analysis, and implications.

Keep a formal, professional tone (no emojis). Do not make up numbers that
are not in the query results.
"""

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


@timed_node("clarify_intent")
async def clarify_intent_node(state: AgentState) -> dict:
    """Analyze the user prompt and produce a structured ClarifyingUserIntent."""
    llm = get_llm()
    structured = llm.with_structured_output(ClarifyingUserIntent)

    prompt = f"{CLARIFY_PROMPT}\n\n{state['user_query']}"
    result = await structured.ainvoke(prompt)

    logger.info("Intent classified as: %s", result.type_of_request.value)
    return {"intent": result}


def route_after_clarify(state: AgentState) -> str:
    """Decide whether to route to the SQL pipeline or the chat response."""
    intent = state.get("intent")
    if intent is None:
        return "chat_response"
    if intent.type_of_request in _DATA_QUESTION_INTENTS:
        logger.info("Routing to SQL pipeline (intent=%s)", intent.type_of_request.value)
        return "analyze_schema"
    logger.info("Routing to chat (intent=%s)", intent.type_of_request.value)
    return "chat_response"


@timed_node("chat_response")
async def chat_response_node(state: AgentState) -> dict:
    """Generate a plain chat response for non-data questions."""
    intent = state.get("intent")
    if intent is None:
        llm = get_llm()
        result = await llm.ainvoke(state["user_query"])
        return {"final_response": result.content}

    enhanced = CHAT_TEMPLATE.render(
        core_intent=intent.core_intent,
        additional_instructions=intent.additional_instructions,
        tone=intent.tone,
        style=intent.style,
        output_format=intent.output_format,
        context_blocks=[block.model_dump() for block in intent.context_blocks],
    )

    llm = get_llm()
    result = await llm.ainvoke(enhanced)
    return {"final_response": result.content}


@timed_node("analyze_schema")
def analyze_schema_node(state: AgentState) -> dict:
    """Read the DuckDB database schema and write it to state."""
    try:
        schema = get_schema_info(DB_PATH)
        logger.info("Schema loaded: %d chars", len(schema))
        return {"schema_info": schema}
    except Exception as exc:
        logger.error("Failed to read schema: %s", exc)
        raise


@timed_node("generate_sql")
async def generate_sql_node(state: AgentState) -> dict:
    """Generate a SQL query from the user question and schema info."""
    prompt = SQL_GENERATION_PROMPT.format(
        schema_info=state.get("schema_info", "Unknown schema"),
        user_query=state["user_query"],
    )

    # Feed back the previous validation error so the LLM can correct it.
    prev_error = state.get("sql_validation_error")
    if prev_error:
        prompt += (
            f"\n\n## Important — your previous query was rejected\n\n"
            f"{prev_error}\n\n"
            f"Fix the issue described above and produce a corrected query."
        )

    llm = get_llm()
    structured = llm.with_structured_output(SQLQuery)
    result = await structured.ainvoke(prompt)

    logger.info("Generated SQL: %s", result.query[:120])
    return {"sql_query": result}


@timed_node("validate_sql")
def validate_sql_node(state: AgentState) -> dict:
    """Validate the generated SQL via AST analysis + EXPLAIN cost estimation."""
    sql_query = state.get("sql_query")
    if sql_query is None:
        return {"sql_validation_error": "No SQL query was generated."}

    attempts = state.get("sql_validation_attempts", 0)

    # Phase 1: static AST analysis (sqlglot)
    error = validate_sql_ast(sql_query.query)
    if not error:
        # Phase 2: EXPLAIN cost analysis (DuckDB)
        error = estimate_query_cost(sql_query.query, db_path=DB_PATH)

    if error:
        attempts += 1
        logger.warning(
            "SQL validation failed (attempt %d/%d): %s",
            attempts,
            MAX_VALIDATION_RETRIES,
            error,
        )
        return {
            "sql_validation_error": error,
            "sql_validation_attempts": attempts,
        }

    logger.info("SQL validation passed (attempt %d)", attempts + 1)
    return {
        "sql_validation_error": None,
        "sql_validation_attempts": attempts + 1,
    }


def route_after_validation(state: AgentState) -> str:
    """Route after validation: on pass → execute; on retryable fail → regenerate;
    on exhausted retries → error report."""
    error = state.get("sql_validation_error")
    attempts = state.get("sql_validation_attempts", 0)

    if not error:
        return "execute_sql"

    if attempts < MAX_VALIDATION_RETRIES:
        logger.info(
            "Retrying SQL generation (attempt %d/%d)", attempts, MAX_VALIDATION_RETRIES
        )
        return "generate_sql"

    logger.error(
        "SQL validation exhausted %d retries — routing to error report", attempts
    )
    return "generate_report"


@timed_node("execute_sql")
def execute_sql_node(state: AgentState) -> dict:
    """Execute the generated SQL query against DuckDB and store results."""
    sql_query = state.get("sql_query")
    if sql_query is None:
        return {"execution_error": "No SQL query was generated."}

    # Final belt-and-suspenders: ensure query starts with SELECT or WITH.
    # (AST analysis ran in validate_sql_node; this catches any edge case.)
    error = validate_sql_query(sql_query.query)
    if error:
        logger.warning("Query validation failed: %s", error)
        return {"execution_error": error}

    try:
        columns, rows = execute_query(sql_query.query, db_path=DB_PATH)
    except duckdb.Error as exc:
        logger.error("Query execution failed: %s", exc)
        return {"execution_error": f"Query execution failed: {exc}"}

    return {"columns": columns, "rows": rows, "execution_error": None}


def route_chart(state: AgentState) -> str:
    """Decide whether to generate a chart or skip to the report."""
    # Only generate a chart if the user explicitly asked for one
    # AND we have data to chart (no execution error, at least 2 rows).
    intent = state.get("intent")
    if intent is None:
        return "generate_report"

    wants = user_wants_chart(state["user_query"])
    has_data = (
        state.get("execution_error") is None
        and state.get("rows") is not None
        and len(state.get("rows", [])) >= 1  # at least 1 row for a chart
    )

    if wants and has_data:
        logger.info("Routing to chart generation")
        return "generate_chart"
    logger.info("Skipping chart (wants=%s, has_data=%s)", wants, has_data)
    return "generate_report"


@timed_node("generate_chart")
def generate_chart_node(state: AgentState) -> dict:
    """Generate a matplotlib chart from the query result rows in state."""
    rows = state.get("rows")
    columns = state.get("columns")
    if not rows or not columns:
        return {"chart_error": "No data available to chart."}

    try:
        # Determine chart type from user query keywords
        query_lower = state["user_query"].lower()
        if "pie" in query_lower:
            chart_type = "pie"
        elif "line" in query_lower:
            chart_type = "line"
        else:
            chart_type = "bar"

        chart_image = build_chart_image(rows, columns, chart_type)
        logger.info("Chart generated (%s, %d rows)", chart_type, len(rows))
        return {"chart_image": chart_image, "chart_error": None}
    except (KeyError, IndexError, ValueError) as exc:
        logger.error("Chart generation failed: %s", exc)
        return {"chart_error": f"Chart generation failed: {exc}"}


@timed_node("generate_report")
async def generate_report_node(state: AgentState) -> dict:
    """Generate the final structured report from query results."""
    rows = state.get("rows")
    columns = state.get("columns")
    execution_error = state.get("execution_error")

    # Handle error cases gracefully — still produce a report.
    if execution_error:
        report = Report(
            executive_summary=f"The query could not be executed: {execution_error}",
            key_findings=[],
            recommendations=["Review the query and try again."],
            narrative=f"**Error:** {execution_error}\n\nPlease verify the question "
            "and database schema, then retry.",
        )
        return {"report": report}

    if not rows or not columns:
        report = Report(
            executive_summary="The query returned no results.",
            key_findings=[],
            recommendations=["Refine the query filters or verify data availability."],
            narrative="**No data returned.** The query executed successfully but "
            "returned zero rows. The requested data may not exist in the database.",
        )
        return {"report": report}

    chart_status = (
        "A chart was generated and is attached."
        if state.get("chart_image")
        else "No chart was requested."
    )

    prompt = REPORT_PROMPT.format(
        user_query=state["user_query"],
        columns=json.dumps(columns),
        rows=json.dumps(rows, default=str),
        row_count=len(rows),
        chart_status=chart_status,
    )

    llm = get_llm()
    structured = llm.with_structured_output(Report)

    result = await structured.ainvoke(prompt)
    logger.info(
        "Report generated: summary=%d chars, %d findings",
        len(result.executive_summary),
        len(result.key_findings),
    )
    return {"report": result}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Build and compile the SQL Agent StateGraph.

    Returns a compiled graph ready for ``.ainvoke(...)``.
    """
    workflow = StateGraph(AgentState)

    # -- Nodes ----------------------------------------------------------------
    workflow.add_node("clarify_intent", clarify_intent_node)
    workflow.add_node("chat_response", chat_response_node)
    workflow.add_node("analyze_schema", analyze_schema_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("validate_sql", validate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("generate_chart", generate_chart_node)
    workflow.add_node("generate_report", generate_report_node)

    # -- Edges ----------------------------------------------------------------
    workflow.add_edge(START, "clarify_intent")

    workflow.add_conditional_edges(
        "clarify_intent",
        route_after_clarify,
        {
            "analyze_schema": "analyze_schema",
            "chat_response": "chat_response",
        },
    )

    workflow.add_edge("chat_response", END)

    workflow.add_edge("analyze_schema", "generate_sql")
    workflow.add_edge("generate_sql", "validate_sql")

    workflow.add_conditional_edges(
        "validate_sql",
        route_after_validation,
        {
            "execute_sql": "execute_sql",
            "generate_sql": "generate_sql",
            "generate_report": "generate_report",
        },
    )

    workflow.add_conditional_edges(
        "execute_sql",
        route_chart,
        {
            "generate_chart": "generate_chart",
            "generate_report": "generate_report",
        },
    )

    workflow.add_edge("generate_chart", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()
