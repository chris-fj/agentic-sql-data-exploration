"""Chart generation helpers — plain functions, not LangChain tools.

The ``generate_chart_node`` in ``backend.agents.graph`` imports these
directly instead of going through LLM tool-calling machinery.
"""

import base64
import io
import logging

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

VALID_CHART_TYPES = frozenset({"bar", "line", "pie"})


def build_chart_image(
    rows: list[dict],
    columns: list[str],
    chart_type: str = "bar",
    *,
    figsize: tuple[int, int] = (8, 5),
    dpi: int = 100,
) -> str:
    """Build a matplotlib chart from rows/columns and return a base64 PNG data URI.

    Args:
        rows: List of row dicts (keys are column names). First column = labels,
              second column (if present) = values.
        columns: Ordered column names.
        chart_type: ``"bar"``, ``"line"``, or ``"pie"``.
        figsize: Matplotlib figure size.
        dpi: Output resolution.

    Returns:
        A ``data:image/png;base64,...`` string.
    """
    chart_type = chart_type.lower().strip()
    if chart_type not in VALID_CHART_TYPES:
        chart_type = "bar"

    fig, ax = plt.subplots(figsize=figsize)
    labels = [str(row[columns[0]]) for row in rows]

    if chart_type == "pie":
        values = [float(row[columns[1]]) if len(columns) > 1 else 0 for row in rows]
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
    else:
        x = range(len(labels))
        values = [float(row[columns[1]]) if len(columns) > 1 else 0 for row in rows]
        if chart_type == "bar":
            ax.bar(x, values, color="#4c72b0")
        elif chart_type == "line":
            ax.plot(x, values, marker="o", color="#4c72b0")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(columns[1] if len(columns) > 1 else "")
        ax.set_xlabel(columns[0] if len(columns) > 0 else "")
        fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    logger.info("Chart built: type=%s, %d rows", chart_type, len(rows))
    return f"data:image/png;base64,{b64}"
