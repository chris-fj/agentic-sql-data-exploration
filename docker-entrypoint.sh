#!/bin/bash
# ------------------------------------------------------------------
# agentic-sql-data-exploration entrypoint
#
# Runs both the FastAPI backend (uvicorn) and Streamlit frontend
# in a single container. On SIGTERM/SIGINT both are shut down
# gracefully.
# ------------------------------------------------------------------
set -e

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-8501}"
FRONTEND_ADDRESS="${FRONTEND_ADDRESS:-0.0.0.0}"

echo "==> Starting backend on ${BACKEND_HOST}:${BACKEND_PORT}..."
uv run --no-project uvicorn backend.app:app \
    --host "${BACKEND_HOST}" \
    --port "${BACKEND_PORT}" &
UVICORN_PID=$!

echo "==> Starting frontend on ${FRONTEND_ADDRESS}:${FRONTEND_PORT}..."
streamlit run src/frontend/app.py \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.port "${FRONTEND_PORT}" \
    --server.address "${FRONTEND_ADDRESS}" &
STREAMLIT_PID=$!

# ------------------------------------------------------------------
# Signal handler: forward SIGTERM/SIGINT to both child processes
# ------------------------------------------------------------------
cleanup() {
    local exit_code=$?
    echo "==> Shutting down (exit=${exit_code})..."
    kill "${UVICORN_PID}" 2>/dev/null || true
    kill "${STREAMLIT_PID}" 2>/dev/null || true
    wait "${UVICORN_PID}" 2>/dev/null || true
    wait "${STREAMLIT_PID}" 2>/dev/null || true
    echo "==> All services stopped."
    exit "${exit_code}"
}

trap cleanup SIGTERM SIGINT

# ------------------------------------------------------------------
# Block until either process exits, then shut down the other.
# 'wait -n' returns the exit code of the first process that ends.
# ------------------------------------------------------------------
wait -n "${UVICORN_PID}" "${STREAMLIT_PID}"
cleanup
