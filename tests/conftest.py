"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove LLM-related env vars from the test environment so that local
    ``.env`` files don't leak into tests."""
    for key in (
        "LLM_MODEL",
        "LLM_ENDPOINT",
        "LLM_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    for key in list(os.environ):
        if key.startswith("LLM_KWARGS__"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def temp_duckdb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Create a temporary DuckDB database with the expected schema.

    Patches the module-level ``DB_PATH`` constants in ``sql_tool`` and
    ``graph`` so all nodes that connect to DuckDB use the temp file.
    """
    db_path = str(tmp_path / "test_sql_agent.db")

    # Patch the module-level constants — os.getenv was evaluated at import
    # time so setenv alone won't change them.
    monkeypatch.setattr("backend.tools.sql_tool.DB_PATH", db_path)
    monkeypatch.setattr("backend.agents.graph.DB_PATH", db_path)

    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE territory (
            territory_id INTEGER,
            territory_name VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE seller (
            seller_id INTEGER,
            territory_id INTEGER,
            seller_name VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE transactions (
            transaction_id INTEGER,
            transaction_date DATE,
            seller_id INTEGER,
            amount_eur DOUBLE
        )
    """)
    conn.execute("""
        INSERT INTO territory VALUES (1, 'Europe'), (2, 'Asia')
    """)
    conn.execute("""
        INSERT INTO seller VALUES (1, 1, 'Alice'), (2, 2, 'Bob')
    """)
    conn.execute("""
        INSERT INTO transactions VALUES
            (1, '2025-01-15', 1, 1000.0),
            (2, '2025-01-20', 2, 800.0),
            (3, '2025-02-10', 1, 500.0)
    """)
    conn.close()

    return db_path
