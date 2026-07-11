"""Tool that generates a matplotlib chart from a SQL query result."""

import base64
import io
import os
import re

import duckdb
import matplotlib
import pandas as pd
from langchain_core.tools import tool

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DB_PATH = os.getenv("DB_PATH", "/db/sql_agent.db")

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

VALID_CHART_TYPES = {"bar", "line", "pie"}


@tool
def generate_chart(sql_query: str, chart_title: str, chart_type: str) -> str:
    """Run a SQL query and generate a chart from the results.

    The chart is returned as a base64-encoded PNG data URI that can be
    rendered directly in an <img> tag or with st.image().

    Args:
        sql_query: A SELECT SQL query whose results will be charted.
            The first column is used as the x-axis / labels; the second
            numeric column is used as the y-axis / values.
        chart_title: Title displayed above the chart.
        chart_type: One of "bar", "line", or "pie".

    Returns:
        A base64 data URI string (data:image/png;base64,…) on success,
        or an error message string on failure.
    """
    chart_type = chart_type.lower().strip()
    if chart_type not in VALID_CHART_TYPES:
        return (
            f"ERROR: Unsupported chart type '{chart_type}'. "
            f"Choose one of: {', '.join(sorted(VALID_CHART_TYPES))}."
        )

    stripped = sql_query.strip().rstrip(";")

    if FORBIDDEN_KEYWORDS.search(stripped):
        return (
            "ERROR: Only SELECT queries are allowed. "
            "The query contains a forbidden keyword."
        )

    upper = stripped.upper()
    if not upper.startswith("SELECT") and not upper.startswith("WITH"):
        return "ERROR: Only SELECT queries are allowed."

    # Run the query
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        df = conn.execute(stripped).fetchdf()
        conn.close()
    except Exception as exc:
        return f"ERROR: Query failed: {exc}"

    if df.empty:
        return "ERROR: Query returned no data — nothing to chart."

    # Build chart
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = df.iloc[:, 0].astype(str).tolist()

    if chart_type == "pie":
        values = df.iloc[:, 1]
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
    else:
        x = range(len(labels))
        values = df.iloc[:, 1]
        if chart_type == "bar":
            ax.bar(x, values, color="#4c72b0")
        elif chart_type == "line":
            ax.plot(x, values, marker="o", color="#4c72b0")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(chart_title)
        ax.set_ylabel(df.columns[1] if len(df.columns) > 1 else "")
        ax.set_xlabel(df.columns[0] if len(df.columns) > 0 else "")
        fig.tight_layout()

    # Encode to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
