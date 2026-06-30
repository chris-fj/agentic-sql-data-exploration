import os
import streamlit as st
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.title("SQL Agent — LLM Demo")

with st.form("llm_form"):
    user_text = st.text_area("Enter your prompt", placeholder="Type something here...")
    submitted = st.form_submit_button("Submit")

if submitted:
    if not user_text.strip():
        st.warning("Please enter some text before submitting.")
    else:
        try:
            with st.spinner("Generating LLM response..."):
                response = httpx.post(
                    f"{BACKEND_URL}/api/call_structured_llm",
                    json={"prompt": user_text},
                    timeout=60,
                )
            response.raise_for_status()
            data = response.json()
            st.code(f"**LLM response:** {data['output']}", language=None)
        except httpx.ConnectError as e:
            st.write(e)
            st.error(
                "Could not connect to the backend. "
                "Make sure it's running with: `uv run uvicorn backend.app:app`"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
