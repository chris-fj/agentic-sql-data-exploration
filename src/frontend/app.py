import asyncio
import logging
import base64
import os

import httpx
import streamlit as st
from backend.utils.prompts import (
    build_enhanced_output,
    clarify_user_intent,
)

from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
TIMEOUT = int(os.getenv("CLOUD_TIMEOUT", 600))


async def _post_structured_llm(url: str, payload: dict, timeout: int) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        return await client.post(url, json=payload, timeout=timeout)


st.set_page_config(page_title="SQL Agent", layout="wide")
st.title("SQL Agent — LLM Demo")

# ------------------------------------------------------------------
# Mode selector
# ------------------------------------------------------------------
mode = st.radio(
    "Mode",
    options=["General Chat", "Data Question (SQL Agent)"],
    horizontal=True,
)

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
with st.form("llm_form"):
    user_text = st.text_area(
        "Enter your prompt",
        placeholder=(
            "Ask a question about transactions, sellers, or territories…"
            if "SQL Agent" in mode
            else "Type something here..."
        ),
    )
    submitted = st.form_submit_button("Submit")

# ------------------------------------------------------------------
# Handle submission
# ------------------------------------------------------------------
if submitted and not user_text.strip():
    st.warning("Please enter some text before submitting.")

elif submitted:
    # --- General Chat mode (existing pipeline) ---
    if mode == "General Chat":
        try:
            with st.spinner("Understanding your request"):
                user_intent = asyncio.run(clarify_user_intent(user_text, "cloud"))
                user_enhanced_prompt = build_enhanced_output(user_intent)
            with st.spinner("Processing your request"):
                response = asyncio.run(
                    _post_structured_llm(
                        f"{BACKEND_URL}/api/structured-llm",
                        {"prompt": user_enhanced_prompt},
                        TIMEOUT,
                    )
                )

            response.raise_for_status()
            data = response.json()
            st.info("**LLM response**")
            st.markdown(data["output"])
        except httpx.ConnectError as e:
            st.write(e)
            st.error(
                "Could not connect to the backend. "
                "Make sure it's running with: `uv run uvicorn backend.app:app`"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # --- SQL Agent mode ---
    else:
        try:
            with st.spinner("Analyzing your data question…"):
                response = asyncio.run(
                    _post_structured_llm(
                        f"{BACKEND_URL}/api/sql-agent",
                        {"query": user_text, "llm": "cloud"},
                        TIMEOUT,
                    )
                )
            response.raise_for_status()
            data = response.json()

            st.markdown(data["explanation"])

            if data.get("chart"):
                # The chart is a base64 data URI — decode and display
                chart_b64 = data["chart"]
                if chart_b64.startswith("data:"):
                    header, encoded = chart_b64.split(",", 1)
                else:
                    encoded = chart_b64
                chart_bytes = base64.b64decode(encoded)
                st.image(chart_bytes, caption="Chart", width="content")

        except httpx.ConnectError as e:
            st.write(e)
            st.error(
                "Could not connect to the backend. "
                "Make sure it's running with: `uv run uvicorn backend.app:app`"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
