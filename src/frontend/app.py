import asyncio
import logging
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


st.title("SQL Agent — LLM Demo")

with st.form("llm_form"):
    user_text = st.text_area("Enter your prompt", placeholder="Type something here...")
    submitted = st.form_submit_button("Submit")

if submitted:
    if not user_text.strip():
        st.warning("Please enter some text before submitting.")
    else:
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
