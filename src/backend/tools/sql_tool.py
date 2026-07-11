"""Tool that executes a read-only SQL query against the DuckDB database."""

import json
import os
import re

import duckdb
from langchain_core.tools import tool

DB_PATH = os.getenv("DB_PATH", "/db/sql_agent.db")

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

MAX_ROWS = 1000


@tool
def run_sql_query(query: str) -> str:
    """Execute a SELECT-only SQL query against the DuckDB database.

    The database contains three tables:
      - transactions: transaction_id, transaction_date, seller_id, amount_eur
      - seller: seller_id, territory_id, seller_name
      - territory: territory_id, territory_name

    JOIN seller ON transactions.seller_id = seller.seller_id
    JOIN territory ON seller.territory_id = territory.territory_id

    Args:
        query: A SELECT SQL query to run. Only SELECT statements are allowed.

    Returns:
        A JSON-encoded list of result rows as dictionaries, or an error message.
    """
    stripped = query.strip().rstrip(";")

    if FORBIDDEN_KEYWORDS.search(stripped):
        return (
            "ERROR: Only SELECT queries are allowed. "
            "The query contains a forbidden keyword (DROP, DELETE, INSERT, "
            "UPDATE, ALTER, CREATE, TRUNCATE)."
        )

    upper = stripped.upper()
    if not upper.startswith("SELECT") and not upper.startswith("WITH"):
        return (
            "ERROR: Only SELECT queries (or WITH … SELECT) are allowed. "
            f"Query starts with: {stripped[:60]}"
        )

    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        result = conn.execute(stripped).fetchmany(MAX_ROWS)
        columns = [desc[0] for desc in conn.description]
        conn.close()
    except Exception as exc:
        return f"ERROR: Query failed: {exc}"

    rows = [dict(zip(columns, row)) for row in result]
    return json.dumps(rows, default=str)
