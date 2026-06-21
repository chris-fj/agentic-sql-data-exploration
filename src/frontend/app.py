import streamlit as st
import httpx

BACKEND_URL = "http://localhost:8000"

st.title("SQL Agent — Echo Demo")

with st.form("echo_form"):
    user_text = st.text_area("Enter your text", placeholder="Type something here...")
    submitted = st.form_submit_button("Submit")

if submitted:
    if not user_text.strip():
        st.warning("Please enter some text before submitting.")
    else:
        try:
            response = httpx.post(
                f"{BACKEND_URL}/api/echo",
                json={"text": user_text},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            st.info(f"**Echo from backend:** {data['echo']}")
        except httpx.ConnectError:
            st.error(
                "Could not connect to the backend. "
                "Make sure it's running with: `uv run uvicorn backend.app:app`"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
