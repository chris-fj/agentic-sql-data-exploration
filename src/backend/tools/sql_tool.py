"""DuckDB query helpers — plain functions, not LangChain tools.

The StateGraph nodes in ``backend.agents.graph`` import these directly
instead of going through the LLM tool-calling machinery.
"""

import json
import logging
import os
import re
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "/db/sql_agent.db")
MAX_ROWS = 1000

# ---------------------------------------------------------------------------
# sqlglot-based AST validation (Phase 1 — static analysis)
# ---------------------------------------------------------------------------

# Root-level AST types that represent destructive DML / DDL / DCL.
_DANGEROUS_STATEMENT_TYPES: frozenset[type[exp.Expression]] = frozenset(
    {
        exp.Delete,
        exp.Drop,
        exp.Insert,
        exp.Update,
        exp.Alter,         # ALTER TABLE / ALTER VIEW / …
        exp.TruncateTable, # TRUNCATE TABLE
        exp.Create,
        exp.Grant,
        exp.Revoke,
        exp.Merge,
        exp.Copy,          # COPY … TO (data exfiltration)
    }
)

# System schema prefixes that the LLM must never query.
_BLOCKED_SCHEMAS: frozenset[str] = frozenset(
    {"information_schema", "pg_catalog", "sqlite_master", "pragma"}
)

# Function-call names (exp.Anonymous) that expose DuckDB internals.
_BLOCKED_FUNCTIONS: frozenset[str] = frozenset(
    {
        "current_setting",
        "current_database",
        "current_schema",
        "version",
        "current_date",
        "current_time",
        "current_timestamp",
        "now",
        "gen_random_uuid",
        "uuid",
        "typeof",
        "txid_current",
    }
)

# duckdb_*() function prefix — any function starting with this exposes metadata.
_DUCKDB_META_PREFIX = "duckdb_"

# exp.Command text prefixes that indicate schema discovery (not EXPLAIN —
# EXPLAIN is used by our own cost-analysis pass so it's not blocked here).
_BLOCKED_COMMAND_PREFIXES: tuple[str, ...] = ("SHOW", "DESCRIBE", "LIST")


def validate_sql_ast(query: str) -> str | None:
    """Validate *query* via sqlglot AST analysis.

    Checks performed:

    1. Multi-statement rejection (sql injection via ``SELECT 1; DROP …``).
    2. Destructive root statements (DROP, DELETE, INSERT, UPDATE, ALTER,
       CREATE, TRUNCATE, GRANT, REVOKE, MERGE, COPY).
    3. System-catalog access (information_schema, pg_catalog,
       ``duckdb_*()`` functions, PRAGMA, SHOW, DESCRIBE).

    CTE aliases named after SQL keywords (e.g. ``deleted_items``) are
    **not** flagged — they are ``Identifier`` nodes inside a ``CTE``,
    not DML statements.

    Returns an error string if the query is rejected, or ``None`` if safe.
    """
    # -- Multi-statement check ------------------------------------------------
    statements = sqlglot.parse(query, error_level=sqlglot.ErrorLevel.IGNORE)
    # sqlglot.parse returns [None] when it can't parse anything
    non_null = [s for s in statements if s is not None]
    if len(non_null) > 1:
        return (
            "ERROR: Multi-statement queries are not allowed. "
            f"Found {len(non_null)} statements. "
            "Write a single SELECT or WITH … SELECT query."
        )
    if len(non_null) == 0:
        return "ERROR: Could not parse the SQL query — it appears to be invalid."

    try:
        tree = non_null[0]
    except Exception:
        return "ERROR: Could not parse the SQL query."

    # -- Root-type validation: only SELECT, WITH, or EXPLAIN are allowed ----
    # sqlglot parses typos like "SELEC 1" as Column, not Select — reject those.
    _ALLOWED_ROOTS = (exp.Select, exp.With, exp.Command)
    if not isinstance(tree, _ALLOWED_ROOTS):
        return (
            "ERROR: The query does not appear to be a valid SELECT statement. "
            f"Parsed as: {type(tree).__name__}. "
            "Write a valid SELECT or WITH … SELECT query."
        )

    # -- Walk the AST ---------------------------------------------------------
    for node in tree.walk():
        # Category A: destructive statement types
        if type(node) in _DANGEROUS_STATEMENT_TYPES:
            return (
                f"ERROR: Dangerous SQL statement detected: {type(node).__name__}. "
                "Only SELECT queries are allowed."
            )

        # Category A: PRAGMA
        if isinstance(node, exp.Pragma):
            return "ERROR: PRAGMA statements are not allowed."

        # Category B: DESCRIBE (its own AST type, not Command)
        if isinstance(node, exp.Describe):
            return (
                "ERROR: DESCRIBE statements are not allowed. "
                "The database schema is already provided."
            )

        # Category B: SHOW (parsed as Command)
        if isinstance(node, exp.Command):
            cmd = node.this.strip().upper() if node.this else ""
            if cmd.startswith(_BLOCKED_COMMAND_PREFIXES):
                return (
                    f"ERROR: {cmd.split()[0]} statements are not allowed. "
                    "The database schema is already provided."
                )

        # Category B: system-schema tables
        if isinstance(node, exp.Table):
            db = (node.db or "").lower()
            if db in _BLOCKED_SCHEMAS:
                return (
                    f"ERROR: Accessing system catalog '{db}' is not allowed. "
                    "The database schema is already provided."
                )

        # Category B: metadata / duckdb_*() functions
        if isinstance(node, exp.Anonymous):
            name = (node.name or "").lower()
            if name.startswith(_DUCKDB_META_PREFIX):
                return (
                    f"ERROR: The function '{node.name}' exposes database metadata "
                    "and is not allowed. The schema is already provided."
                )
            if name in _BLOCKED_FUNCTIONS:
                return (
                    f"ERROR: The function '{node.name}' is not allowed "
                    "in user queries."
                )

    return None


# ---------------------------------------------------------------------------
# EXPLAIN-based cost analysis (Phase 2 — dynamic analysis)
# ---------------------------------------------------------------------------

# Cardinality thresholds for EXPLAIN-based rejection.
_MAX_ROWS_ABSOLUTE = 1_000_000    # Reject anything over this regardless
_MAX_ROWS_WITHOUT_LIMIT = 10_000  # Reject without LIMIT or aggregation
# Full-table-scan threshold: scans over this cardinality without a FILTER
# are flagged as expensive.
_MAX_SCAN_CARDINALITY = 100_000


def _has_limit_or_aggregation(tree: exp.Expression) -> bool:
    """Return True if the query tree contains a LIMIT clause or aggregation."""
    for node in tree.walk():
        if isinstance(node, (exp.Limit, exp.Group, exp.AggFunc)):
            return True
    return False


def _cardinality(node: dict) -> int:
    """Extract the estimated cardinality from an EXPLAIN plan node."""
    extra = node.get("extra_info", {})
    if isinstance(extra, dict):
        raw = extra.get("Estimated Cardinality", "0")
        try:
            return int(raw)
        except (ValueError, TypeError):
            return 0
    return 0


def _walk_plan(node: dict, *, has_limit_or_agg: bool) -> str | None:
    """Recursively walk an EXPLAIN plan dict and return an error if unsafe."""
    name = node.get("name", "")
    cardinality = _cardinality(node)

    # Absolute ceiling
    if cardinality > _MAX_ROWS_ABSOLUTE:
        return (
            f"ERROR: Query estimated to return {cardinality:,} rows "
            f"(max allowed: {_MAX_ROWS_ABSOLUTE:,}). "
            "Add filters or a LIMIT clause."
        )

    # Without LIMIT/aggregation, enforce lower ceiling
    if not has_limit_or_agg and cardinality > _MAX_ROWS_WITHOUT_LIMIT:
        return (
            f"ERROR: Query estimated to return {cardinality:,} rows "
            f"without LIMIT or aggregation (max: {_MAX_ROWS_WITHOUT_LIMIT:,}). "
            "Add a LIMIT clause or use aggregation (GROUP BY, COUNT, SUM, AVG)."
        )

    # Detect cross-join / unqualified nested-loop join
    if name in ("CROSS_PRODUCT", "NESTED_LOOP_JOIN"):
        # NESTED_LOOP_JOIN is only problematic without a join condition —
        # but we can't easily detect that from the plan alone, so we flag it
        # as a warning only if cardinality is high.
        if name == "CROSS_PRODUCT" or cardinality > _MAX_SCAN_CARDINALITY:
            return (
                "ERROR: Cross join or unqualified join detected in query plan. "
                "Add a proper JOIN condition (ON …) between the tables."
            )

    # Large full table scans without filters
    if name in ("TABLE_SCAN", "SEQ_SCAN") and cardinality > _MAX_SCAN_CARDINALITY:
        children = node.get("children", [])
        has_filter = any(c.get("name", "") == "FILTER" for c in children)
        if not has_filter:
            return (
                f"ERROR: Full table scan with {cardinality:,} estimated rows "
                "and no filter. Add a WHERE clause or LIMIT."
            )

    # Recurse into children
    for child in node.get("children", []):
        if isinstance(child, dict):
            err = _walk_plan(child, has_limit_or_agg=has_limit_or_agg)
            if err:
                return err

    return None


def estimate_query_cost(
    query: str, db_path: str | None = None
) -> str | None:
    """Run EXPLAIN and estimate the cost of *query* against DuckDB.

    Returns an error string if the query is too expensive, or ``None`` if
    it passes the cost check.  If EXPLAIN itself fails, returns ``None``
    (skip — don't block a query just because planning failed).
    """
    db_path = db_path or DB_PATH
    try:
        conn = duckdb.connect(db_path, read_only=True)
        row = conn.execute(f"EXPLAIN (FORMAT JSON) {query}").fetchone()
        conn.close()
    except Exception:
        logger.warning("EXPLAIN failed for query — skipping cost check", exc_info=True)
        return None

    # DuckDB EXPLAIN FORMAT JSON returns (name, json_string) — the JSON is
    # in the second column.
    if row is None or len(row) < 2 or row[1] is None:
        return None

    try:
        plan_json = json.loads(row[1])
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse EXPLAIN output — skipping cost check")
        return None

    # Determine whether the query already has LIMIT or aggregation
    try:
        tree = sqlglot.parse_one(query)
        has_limit_or_agg = _has_limit_or_aggregation(tree)
    except Exception:
        has_limit_or_agg = False

    # Walk the plan tree
    if isinstance(plan_json, list):
        for item in plan_json:
            err = _walk_plan(item, has_limit_or_agg=has_limit_or_agg)
            if err:
                return err
    elif isinstance(plan_json, dict):
        err = _walk_plan(plan_json, has_limit_or_agg=has_limit_or_agg)
        if err:
            return err

    return None


# ---------------------------------------------------------------------------
# Legacy validation (belt-and-suspenders — runs in execute_sql after
# sqlglot validation has already passed)
# ---------------------------------------------------------------------------


def validate_sql_query(query: str) -> str | None:
    """Final sanity check: ensure the query starts with SELECT or WITH.

    The heavy lifting (AST analysis, cost estimation) is done by
    ``validate_sql_ast`` and ``estimate_query_cost`` in the
    ``validate_sql_node``.  This function only checks the prefix.
    """
    stripped = query.strip().rstrip(";")
    upper = stripped.upper()
    if not upper.startswith("SELECT") and not upper.startswith("WITH"):
        return (
            "ERROR: Only SELECT queries (or WITH … SELECT) are allowed. "
            f"Query starts with: {stripped[:60]}"
        )
    return None


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------


def execute_query(
    query: str,
    *,
    db_path: str = DB_PATH,
    max_rows: int = MAX_ROWS,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Run a SELECT query against DuckDB (read-only).

    Returns:
        ``(columns, rows)`` where *columns* is a list of column name strings
        and *rows* is a list of dicts.

    Raises:
        duckdb.Error: On query failure.
    """
    conn = duckdb.connect(db_path, read_only=True)
    try:
        result = conn.execute(query).fetchmany(max_rows)
        columns = [desc[0] for desc in conn.description]
    finally:
        conn.close()

    rows = [dict(zip(columns, row)) for row in result]
    logger.info("Query returned %d rows, %d columns", len(rows), len(columns))
    return columns, rows


def get_schema_info(db_path: str = DB_PATH) -> str:
    """Query DuckDB for table and column metadata.

    Returns a markdown-formatted string describing the database schema.
    """
    conn = duckdb.connect(db_path, read_only=True)
    try:
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()

        lines: list[str] = []
        for (table_name,) in tables:
            cols = conn.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table_name],
            ).fetchall()
            col_str = ", ".join(f"{name} ({dtype})" for name, dtype in cols)
            lines.append(f"  - **{table_name}**: {col_str}")
    finally:
        conn.close()

    return "\n".join(lines)
