import asyncio
import base64
import datetime as dt
import json
import logging
import os
from zoneinfo import ZoneInfo

import httpx
import streamlit as st

from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "600"))


async def _post(url: str, payload: dict, timeout: int) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        return await client.post(url, json=payload, timeout=timeout)


st.set_page_config(page_title="SQL Agent", layout="wide")
st.title("SQL Agent — LLM Demo")
st.write(f"Current datetime is {dt.datetime.now(tz=ZoneInfo('Europe/Madrid'))}")

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
with st.form("llm_form"):
    user_text = st.text_area(
        "Enter your prompt",
        placeholder="Ask a question about transactions, sellers, or territories…",
    )
    submitted = st.form_submit_button("Submit")

# --------------------------------------------------------------------------
# Handle submission
# --------------------------------------------------------------------------
if submitted and not user_text.strip():
    st.warning("Please enter some text before submitting.")

elif submitted:
    try:
        logger.info("Processing user request: %s", user_text)
        with st.spinner("Processing…"):
            response = asyncio.run(
                _post(
                    f"{BACKEND_URL}/api/sql-agent",
                    {"query": user_text},
                    TIMEOUT,
                )
            )
        response.raise_for_status()
        data = response.json()

        # -- Key findings table ------------------------------------------------
        findings: list[dict] = data.get("key_findings", [])
        recommendations: list[str] = data.get("recommendations", [])

        if findings or recommendations:
            # Structured report layout
            if findings:
                st.subheader("Key Findings")
                rows = [
                    {
                        "Insight": f["insight"],
                        "Value": f.get("value") or "—",
                    }
                    for f in findings
                ]
                st.dataframe(rows, width="content", hide_index=True)

            if recommendations:
                st.subheader("Recommendations")
                for rec in recommendations:
                    st.markdown(f"- {rec}")

            st.divider()

        # -- Narrative markdown ------------------------------------------------
        st.markdown(data["explanation"])

        # -- Chart -------------------------------------------------------------
        chart_b64 = data.get("chart")
        if chart_b64:
            if chart_b64.startswith("data:"):
                header, encoded = chart_b64.split(",", 1)
            else:
                encoded = chart_b64
            chart_bytes = base64.b64decode(encoded)
            st.image(chart_bytes, caption="Chart", width="content")

    except httpx.ConnectError:
        st.error(
            "Could not connect to the backend. "
            "Make sure it's running with: `uv run uvicorn backend.app:app`"
        )
    except (httpx.TimeoutException, httpx.HTTPStatusError, json.JSONDecodeError) as e:
        st.error(f"An error occurred: {e}")
