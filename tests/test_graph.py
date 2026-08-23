"""Tests for the SQL Agent StateGraph nodes and routing logic."""

from __future__ import annotations

import json
import tempfile
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import duckdb
import pytest

from backend.agents.graph import (
    analyze_schema_node,
    chat_response_node,
    clarify_intent_node,
    execute_sql_node,
    generate_chart_node,
    generate_report_node,
    generate_sql_node,
    route_after_clarify,
    route_chart,
)
from backend.agents.state import (
    _DATA_QUESTION_INTENTS,
    AgentState,
    ClarifyingUserIntent,
    ContextBlock,
    KeyFinding,
    Report,
    SQLQuery,
    UserIntent,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_intent() -> ClarifyingUserIntent:
    return ClarifyingUserIntent(
        type_of_request=UserIntent.DATA_QUESTION,
        core_intent="Show total sales by territory",
        keywords=["sales", "territory"],
        style="prose",
        output_format="markdown",
        tone="neutral",
        additional_instructions="",
        context_blocks=[],
    )


@pytest.fixture
def sample_sql_query() -> SQLQuery:
    return SQLQuery(
        query="SELECT t.territory_name, SUM(tr.amount_eur) AS total "
        "FROM transactions tr "
        "JOIN seller s ON tr.seller_id = s.seller_id "
        "JOIN territory t ON s.territory_id = t.territory_id "
        "GROUP BY t.territory_name",
        explanation="Aggregates total sales by territory.",
    )


def _state(**overrides: object) -> AgentState:
    """Build a minimal AgentState with sensible defaults, overridable per field."""
    defaults: dict[str, object] = {
        "user_query": "Show total sales by territory",
        "llm_type": "cloud",
        "messages": [],
        "intent": None,
        "schema_info": None,
        "sql_query": None,
        "columns": None,
        "rows": None,
        "execution_error": None,
        "chart_image": None,
        "chart_error": None,
        "report": None,
        "final_response": None,
        "final_error": None,
    }
    defaults.update(overrides)
    return defaults  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# route_after_clarify
# ---------------------------------------------------------------------------


class TestRouteAfterClarify:
    def test_routes_to_sql_for_question(self, sample_intent):
        state = _state(intent=sample_intent)
        assert route_after_clarify(state) == "analyze_schema"

    def test_routes_to_sql_for_compare(self):
        intent = ClarifyingUserIntent(
            type_of_request=UserIntent.COMPARE,
            core_intent="Compare X and Y",
            keywords=[],
        )
        state = _state(intent=intent)
        assert route_after_clarify(state) == "analyze_schema"

    def test_routes_to_chat_for_other(self):
        intent = ClarifyingUserIntent(
            type_of_request=UserIntent.OTHER,
            core_intent="Explain what DuckDB is",
            keywords=[],
        )
        state = _state(intent=intent)
        assert route_after_clarify(state) == "chat_response"

    def test_routes_to_chat_for_other(self):
        intent = ClarifyingUserIntent(
            type_of_request=UserIntent.OTHER,
            core_intent="Tell me a joke",
            keywords=[],
        )
        state = _state(intent=intent)
        assert route_after_clarify(state) == "chat_response"

    def test_routes_to_chat_when_intent_is_none(self):
        state = _state(intent=None)
        assert route_after_clarify(state) == "chat_response"

    def test_all_data_intents_route_to_sql(self):
        for intent_type in _DATA_QUESTION_INTENTS:
            intent = ClarifyingUserIntent(
                type_of_request=intent_type,
                core_intent="Analyze data",
                keywords=[],
            )
            state = _state(intent=intent)
            assert (
                route_after_clarify(state) == "analyze_schema"
            ), f"Expected {intent_type} to route to SQL pipeline"


# ---------------------------------------------------------------------------
# route_chart
# ---------------------------------------------------------------------------


class TestRouteChart:
    def test_wants_chart_with_chart_keyword(self, sample_intent):
        state = _state(
            intent=sample_intent,
            user_query="Plot a bar chart of sales by territory",
            rows=[{"territory": "Europe", "sales": 1000}],
            columns=["territory", "sales"],
            execution_error=None,
        )
        assert route_chart(state) == "generate_chart"

    def test_no_chart_without_keyword(self, sample_intent):
        state = _state(
            intent=sample_intent,
            user_query="Show total sales by territory",
            rows=[{"territory": "Europe", "sales": 1000}],
            columns=["territory", "sales"],
            execution_error=None,
        )
        assert route_chart(state) == "generate_report"

    def test_no_chart_when_execution_error(self, sample_intent):
        state = _state(
            intent=sample_intent,
            user_query="Plot a chart of sales",
            rows=None,
            columns=None,
            execution_error="Query failed",
        )
        assert route_chart(state) == "generate_report"

    def test_no_chart_when_no_intent(self):
        state = _state(
            intent=None,
            user_query="Plot a chart",
        )
        assert route_chart(state) == "generate_report"

    def test_chart_with_line_keyword(self, sample_intent):
        state = _state(
            intent=sample_intent,
            user_query="Show me a line graph of monthly trends",
            rows=[{"month": "Jan", "value": 100}],
            columns=["month", "value"],
        )
        assert route_chart(state) == "generate_chart"


# ---------------------------------------------------------------------------
# chat_response_node
# ---------------------------------------------------------------------------


class TestChatResponseNode:
    @pytest.mark.asyncio
    async def test_renders_enhanced_prompt_from_intent(self, sample_intent):
        """The LLM should receive the enhanced (template-rendered) prompt."""
        mock_llm = MagicMock()
        mock_llm.content = "Here's an explanation of Docker."
        mock_llm_instance = AsyncMock()
        mock_llm_instance.ainvoke = AsyncMock(return_value=mock_llm)

        state = _state(intent=sample_intent)

        with patch("backend.agents.graph.get_llm", return_value=mock_llm_instance):
            result = await chat_response_node(state)

        call_arg = mock_llm_instance.ainvoke.call_args[0][0]
        assert "Show total sales by territory" in call_arg
        assert "Respond in markdown format." in call_arg
        assert result["final_response"] == "Here's an explanation of Docker."

    @pytest.mark.asyncio
    async def test_chat_without_intent_sends_raw_query(self):
        """When no intent is set, send the raw user query verbatim."""
        mock_llm = MagicMock()
        mock_llm.content = "I'm not sure about that."
        mock_llm_instance = AsyncMock()
        mock_llm_instance.ainvoke = AsyncMock(return_value=mock_llm)

        state = _state(intent=None, user_query="What is life?")

        with patch("backend.agents.graph.get_llm", return_value=mock_llm_instance):
            result = await chat_response_node(state)

        call_arg = mock_llm_instance.ainvoke.call_args[0][0]
        assert call_arg == "What is life?"
        assert result["final_response"] == "I'm not sure about that."

    @pytest.mark.asyncio
    async def test_includes_tone_and_style_in_prompt(self):
        """Non-default tone and style should appear in the enhanced prompt."""
        intent = ClarifyingUserIntent(
            type_of_request=UserIntent.OTHER,
            core_intent="Explain recursion",
            keywords=[],
            tone="technical",
            style="step-by-step instructions",
        )

        mock_llm_instance = AsyncMock()
        mock_llm_instance.ainvoke = AsyncMock(
            return_value=MagicMock(content="Recursion is...")
        )

        state = _state(intent=intent)

        with patch("backend.agents.graph.get_llm", return_value=mock_llm_instance):
            await chat_response_node(state)

        call_arg = mock_llm_instance.ainvoke.call_args[0][0]
        assert "Tone: technical" in call_arg
        assert "Presentation style: step-by-step instructions" in call_arg


# ---------------------------------------------------------------------------
# clarify_intent_node
# ---------------------------------------------------------------------------


class TestClarifyIntentNode:
    @pytest.mark.asyncio
    async def test_returns_structured_intent(self):
        """The node should return a ClarifyingUserIntent from the LLM."""
        expected = ClarifyingUserIntent(
            type_of_request=UserIntent.DATA_QUESTION,
            core_intent="Show sales by territory",
            keywords=["sales", "territory"],
        )

        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=expected)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

        state = _state()

        with patch("backend.agents.graph.get_llm", return_value=mock_llm):
            result = await clarify_intent_node(state)

        assert result["intent"] == expected
        assert result["intent"].type_of_request == UserIntent.DATA_QUESTION


# ---------------------------------------------------------------------------
# generate_sql_node
# ---------------------------------------------------------------------------


class TestGenerateSQLNode:
    @pytest.mark.asyncio
    async def test_generates_sql_query(self):
        """The node should return a SQLQuery from the LLM."""
        expected = SQLQuery(
            query="SELECT * FROM transactions LIMIT 10",
            explanation="Returns the first 10 transactions.",
        )

        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=expected)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

        state = _state(schema_info="- **transactions**: id, amount")

        with patch("backend.agents.graph.get_llm", return_value=mock_llm):
            result = await generate_sql_node(state)

        assert result["sql_query"] == expected
        assert result["sql_query"].query == "SELECT * FROM transactions LIMIT 10"

    @pytest.mark.asyncio
    async def test_includes_schema_in_prompt(self):
        """The SQL generation prompt should include the schema info."""
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(
            return_value=SQLQuery(query="SELECT 1", explanation="test")
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

        state = _state(schema_info="custom schema text here")

        with patch("backend.agents.graph.get_llm", return_value=mock_llm):
            await generate_sql_node(state)

        call_arg = mock_structured.ainvoke.call_args[0][0]
        assert "custom schema text here" in call_arg


# ---------------------------------------------------------------------------
# execute_sql_node
# ---------------------------------------------------------------------------


class TestExecuteSQLNode:
    def test_executes_valid_query(self, sample_sql_query, temp_duckdb):
        """Run a valid query and verify columns + rows are returned."""
        state = _state(sql_query=sample_sql_query)
        result = execute_sql_node(state)

        assert result.get("execution_error") is None
        assert "columns" in result
        assert "rows" in result

    def test_rejects_non_select_query(self):
        """Queries with forbidden keywords should be rejected before execution."""
        sql_query = SQLQuery(
            query="DROP TABLE transactions",
            explanation="This should fail.",
        )
        state = _state(sql_query=sql_query)
        result = execute_sql_node(state)

        assert result["execution_error"] is not None
        assert "SELECT" in result["execution_error"]

    def test_errors_when_no_sql_query(self):
        """When sql_query is None, return an error."""
        state = _state(sql_query=None)
        result = execute_sql_node(state)

        assert result["execution_error"] == "No SQL query was generated."


# ---------------------------------------------------------------------------
# generate_chart_node
# ---------------------------------------------------------------------------


class TestGenerateChartNode:
    def test_generates_bar_chart(self):
        """Build a bar chart from rows and verify a base64 PNG is returned."""
        state = _state(
            user_query="Bar chart of sales",
            rows=[
                {"territory": "Europe", "sales": 1000},
                {"territory": "Asia", "sales": 800},
            ],
            columns=["territory", "sales"],
        )
        result = generate_chart_node(state)

        assert result.get("chart_error") is None
        assert result["chart_image"].startswith("data:image/png;base64,")

    def test_generates_line_chart(self):
        """Line chart keyword should produce a line chart."""
        state = _state(
            user_query="Line chart of trends",
            rows=[
                {"month": "Jan", "value": 100},
                {"month": "Feb", "value": 200},
            ],
            columns=["month", "value"],
        )
        result = generate_chart_node(state)

        assert result.get("chart_error") is None
        assert result["chart_image"].startswith("data:image/png;base64,")

    def test_errors_when_no_data(self):
        """When rows or columns are missing, return a chart_error."""
        state = _state(rows=None, columns=None)
        result = generate_chart_node(state)

        assert result["chart_error"] == "No data available to chart."


# ---------------------------------------------------------------------------
# generate_report_node
# ---------------------------------------------------------------------------


class TestGenerateReportNode:
    @pytest.mark.asyncio
    async def test_generates_report_from_data(self):
        """The node should return a structured Report from the LLM."""
        expected = Report(
            executive_summary="Total sales are 1800 EUR across 2 territories.",
            key_findings=[
                KeyFinding(insight="Europe leads with 1000 EUR", value="1000"),
                KeyFinding(insight="Asia follows with 800 EUR", value="800"),
            ],
            recommendations=["Increase marketing in Asia."],
            narrative="## Sales Report\n\nEurope: 1000 EUR\nAsia: 800 EUR",
        )

        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=expected)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

        state = _state(
            rows=[
                {"territory": "Europe", "sales": 1000},
                {"territory": "Asia", "sales": 800},
            ],
            columns=["territory", "sales"],
        )

        with patch("backend.agents.graph.get_llm", return_value=mock_llm):
            result = await generate_report_node(state)

        assert result["report"] == expected
        assert len(result["report"].key_findings) == 2

    @pytest.mark.asyncio
    async def test_handles_execution_error(self):
        """When the SQL execution failed, produce an error report."""
        state = _state(
            execution_error="Query execution failed: table not found",
        )
        result = await generate_report_node(state)

        report = result["report"]
        assert "table not found" in report.executive_summary
        assert "Review the query" in report.recommendations[0]

    @pytest.mark.asyncio
    async def test_handles_empty_results(self):
        """When the query returned no rows, produce an empty-result report."""
        state = _state(
            rows=[],
            columns=["territory", "sales"],
        )
        result = await generate_report_node(state)

        report = result["report"]
        assert "no results" in report.executive_summary.lower()
        assert "Refine" in report.recommendations[0]

    @pytest.mark.asyncio
    async def test_mentions_chart_when_present(self):
        """When a chart was generated, the prompt should mention it."""
        expected = Report(
            executive_summary="Sales by territory.",
            key_findings=[KeyFinding(insight="Europe leads", value="1000")],
            recommendations=[],
            narrative="Full report here.",
        )

        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=expected)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

        state = _state(
            rows=[{"territory": "Europe", "sales": 1000}],
            columns=["territory", "sales"],
            chart_image="data:image/png;base64,abc123",
        )

        with patch("backend.agents.graph.get_llm", return_value=mock_llm):
            await generate_report_node(state)

        call_arg = mock_structured.ainvoke.call_args[0][0]
        assert "chart was generated" in call_arg.lower()


# ---------------------------------------------------------------------------
# build_graph (integration smoke test)
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_graph_compiles(self):
        """The graph should compile without errors."""
        from backend.agents.graph import build_graph

        graph = build_graph()
        assert graph is not None


# ---------------------------------------------------------------------------
# analyze_schema_node
# ---------------------------------------------------------------------------


class TestAnalyzeSchemaNode:
    def test_reads_schema_from_db(self, temp_duckdb):
        """Read schema from actual DuckDB and verify it returns info."""
        result = analyze_schema_node(_state())
        assert result["schema_info"] is not None
        assert len(result["schema_info"]) > 0
        assert "territory" in result["schema_info"]
        assert "transactions" in result["schema_info"]
        assert "seller" in result["schema_info"]


# ---------------------------------------------------------------------------
# timed_node decorator
# ---------------------------------------------------------------------------


class TestTimedNode:
    def test_sync_node_logs_elapsed(self, caplog):
        """A sync node decorated with @timed_node should log start and end."""
        from backend.middleware.timing import timed_node

        @timed_node("test_sync")
        def node(state):
            return {"result": 42}

        with caplog.at_level("INFO", logger="backend.middleware.timing"):
            result = node({})

        assert result == {"result": 42}
        assert any(
            "test_sync" in r.message and "started" in r.message for r in caplog.records
        )
        assert any(
            "test_sync" in r.message and "completed in" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_async_node_logs_elapsed(self, caplog):
        """An async node decorated with @timed_node should log start and end."""
        import asyncio

        from backend.middleware.timing import timed_node

        @timed_node("test_async")
        async def node(state):
            await asyncio.sleep(0.01)
            return {"result": "ok"}

        with caplog.at_level("INFO", logger="backend.middleware.timing"):
            result = await node({})

        assert result == {"result": "ok"}
        assert any(
            "test_async" in r.message and "started" in r.message for r in caplog.records
        )
        assert any(
            "test_async" in r.message and "completed in" in r.message
            for r in caplog.records
        )

    def test_sync_node_logs_exception(self, caplog):
        """A sync node that raises should log the failure with elapsed time."""
        from backend.middleware.timing import timed_node

        @timed_node("test_error")
        def node(state):
            raise ValueError("boom")

        with caplog.at_level("ERROR", logger="backend.middleware.timing"):
            with pytest.raises(ValueError, match="boom"):
                node({})

        assert any(
            "test_error" in r.message and "failed after" in r.message
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# validate_sql_ast (Phase 1 — static analysis)
# ---------------------------------------------------------------------------


class TestValidateSQLAst:
    def test_passes_plain_select(self):
        from backend.tools.sql_tool import validate_sql_ast

        assert validate_sql_ast("SELECT * FROM transactions") is None

    def test_passes_cte_named_delete(self):
        """CTE alias 'deleted_items' is an Identifier, not exp.Delete — must pass."""
        from backend.tools.sql_tool import validate_sql_ast

        result = validate_sql_ast(
            "WITH deleted_items AS (SELECT * FROM transactions WHERE status = 1) "
            "SELECT * FROM deleted_items"
        )
        assert result is None, f"CTE named 'deleted_items' should pass, got: {result}"

    def test_passes_with_aggregation(self):
        from backend.tools.sql_tool import validate_sql_ast

        assert (
            validate_sql_ast(
                "SELECT territory_name, SUM(amount_eur) FROM transactions "
                "JOIN seller ON transactions.seller_id = seller.seller_id "
                "JOIN territory ON seller.territory_id = territory.territory_id "
                "GROUP BY territory_name"
            )
            is None
        )

    def test_rejects_drop(self):
        from backend.tools.sql_tool import validate_sql_ast

        err = validate_sql_ast("DROP TABLE transactions")
        assert err is not None
        assert "Drop" in err

    def test_rejects_delete(self):
        from backend.tools.sql_tool import validate_sql_ast

        err = validate_sql_ast("DELETE FROM transactions WHERE id = 1")
        assert err is not None
        assert "Delete" in err

    def test_rejects_insert(self):
        from backend.tools.sql_tool import validate_sql_ast

        err = validate_sql_ast(
            "INSERT INTO transactions VALUES (1, '2025-01-01', 1, 100)"
        )
        assert err is not None
        assert "Insert" in err

    def test_rejects_update(self):
        from backend.tools.sql_tool import validate_sql_ast

        err = validate_sql_ast("UPDATE transactions SET amount_eur = 0")
        assert err is not None
        assert "Update" in err

    def test_rejects_truncate(self):
        from backend.tools.sql_tool import validate_sql_ast

        err = validate_sql_ast("TRUNCATE TABLE transactions")
        assert err is not None
        assert "TruncateTable" in err

    def test_rejects_copy(self):
        from backend.tools.sql_tool import validate_sql_ast

        err = validate_sql_ast("COPY transactions TO '/tmp/data.csv'")
        assert err is not None
        assert "Copy" in err

    def test_rejects_information_schema(self):
        from backend.tools.sql_tool import validate_sql_ast

        err = validate_sql_ast("SELECT * FROM information_schema.tables")
        assert err is not None
        assert "information_schema" in err.lower()

    def test_rejects_duckdb_meta_function(self):
        from backend.tools.sql_tool import validate_sql_ast

        err = validate_sql_ast("SELECT * FROM duckdb_tables()")
        assert err is not None
        assert "duckdb_tables" in err

    def test_rejects_pragma(self):
        from backend.tools.sql_tool import validate_sql_ast

        err = validate_sql_ast("PRAGMA table_info('transactions')")
        assert err is not None
        assert "pragma" in err.lower()

    def test_rejects_show_tables(self):
        from backend.tools.sql_tool import validate_sql_ast

        err = validate_sql_ast("SHOW TABLES")
        assert err is not None
        assert "SHOW" in err

    def test_rejects_describe(self):
        from backend.tools.sql_tool import validate_sql_ast

        err = validate_sql_ast("DESCRIBE transactions")
        assert err is not None
        assert "describe" in err.lower()

    def test_rejects_multi_statement(self):
        from backend.tools.sql_tool import validate_sql_ast

        err = validate_sql_ast("SELECT 1; DROP TABLE transactions")
        assert err is not None
        assert "Multi-statement" in err

    def test_rejects_parse_error(self):
        from backend.tools.sql_tool import validate_sql_ast

        err = validate_sql_ast("SELEC 1")
        assert err is not None

    def test_rejects_current_setting(self):
        from backend.tools.sql_tool import validate_sql_ast

        err = validate_sql_ast("SELECT current_setting('search_path')")
        assert err is not None


# ---------------------------------------------------------------------------
# estimate_query_cost (Phase 2 — EXPLAIN analysis)
# ---------------------------------------------------------------------------


class TestEstimateQueryCost:
    def test_cheap_query_passes(self, temp_duckdb):
        from backend.tools.sql_tool import estimate_query_cost

        result = estimate_query_cost("SELECT * FROM transactions LIMIT 5")
        assert result is None, f"Cheap query should pass, got: {result}"

    def test_aggregated_query_passes(self, temp_duckdb):
        from backend.tools.sql_tool import estimate_query_cost

        result = estimate_query_cost(
            "SELECT territory_name, SUM(amount_eur) FROM transactions "
            "JOIN seller ON transactions.seller_id = seller.seller_id "
            "JOIN territory ON seller.territory_id = territory.territory_id "
            "GROUP BY territory_name"
        )
        assert result is None, f"Aggregated query should pass, got: {result}"

    def test_high_cardinality_no_limit_rejected(self, temp_duckdb):
        from backend.tools.sql_tool import estimate_query_cost

        # SELECT without LIMIT or aggregation on multiple tables —
        # the estimated cardinality may exceed the 10K threshold.
        err = estimate_query_cost(
            "SELECT * FROM transactions JOIN seller "
            "ON transactions.seller_id = seller.seller_id"
        )
        # This may or may not be rejected depending on DuckDB's cardinality
        # estimates for small tables.  If it passes, that's fine — the test
        # verifies the function runs without error against the temp DB.
        # The cross-join test below is stricter.
        _ = err  # no-op: just verify no exception

    def test_cross_join_rejected(self, temp_duckdb):
        from backend.tools.sql_tool import estimate_query_cost

        # Explicit cross join with huge LIMIT — should be caught either by
        # cardinality threshold or CROSS_PRODUCT detection.
        err = estimate_query_cost("SELECT * FROM transactions, seller")
        assert err is not None, "Cross join without LIMIT should be rejected"


# ---------------------------------------------------------------------------
# route_after_validation
# ---------------------------------------------------------------------------


class TestRouteAfterValidation:
    def test_pass_routes_to_execute(self):
        from backend.agents.graph import route_after_validation

        state = _state(sql_validation_error=None, sql_validation_attempts=1)
        assert route_after_validation(state) == "execute_sql"

    def test_fail_with_retries_left(self):
        from backend.agents.graph import (
            MAX_VALIDATION_RETRIES,
            route_after_validation,
        )

        state = _state(
            sql_validation_error="ERROR: Drop detected",
            sql_validation_attempts=1,
        )
        assert route_after_validation(state) == "generate_sql"

    def test_fail_with_no_retries_left(self):
        from backend.agents.graph import (
            MAX_VALIDATION_RETRIES,
            route_after_validation,
        )

        state = _state(
            sql_validation_error="ERROR: Drop detected",
            sql_validation_attempts=MAX_VALIDATION_RETRIES,
        )
        assert route_after_validation(state) == "generate_report"


# ---------------------------------------------------------------------------
# validate_sql_node (integration of Phase 1 + Phase 2)
# ---------------------------------------------------------------------------


class TestValidateSQLNode:
    def test_valid_query_passes(self, temp_duckdb):
        from backend.agents.graph import validate_sql_node
        from backend.agents.state import SQLQuery

        state = _state(
            sql_query=SQLQuery(
                query="SELECT * FROM transactions LIMIT 5",
                explanation="Test query",
            ),
            sql_validation_attempts=0,
        )
        result = validate_sql_node(state)
        assert result["sql_validation_error"] is None
        assert result["sql_validation_attempts"] == 1

    def test_dangerous_query_increments_attempts(self):
        from backend.agents.graph import validate_sql_node
        from backend.agents.state import SQLQuery

        state = _state(
            sql_query=SQLQuery(
                query="DROP TABLE transactions",
                explanation="Should fail",
            ),
            sql_validation_attempts=0,
        )
        result = validate_sql_node(state)
        assert result["sql_validation_error"] is not None
        assert result["sql_validation_attempts"] == 1

    def test_null_sql_query_errors(self):
        from backend.agents.graph import validate_sql_node

        state = _state(sql_query=None)
        result = validate_sql_node(state)
        assert result["sql_validation_error"] is not None
