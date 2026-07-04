import os
import streamlit as st
import httpx
from backend.utils.prompts import (
    build_enhanced_output,
    clarify_user_intent,
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
TIMEOUT = int(os.getenv("CLOUD_TIMEOUT", 600))
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
                user_intent = clarify_user_intent(user_text, "cloud")
                user_enhanced_prompt = build_enhanced_output(user_intent)
            with st.spinner("Processing your request"):
                response = httpx.post(
                    f"{BACKEND_URL}/api/structured-llm",
                    json={"prompt": user_enhanced_prompt},
                    timeout=TIMEOUT,
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
