# =============================================================================
# agentic-sql-data-exploration
#
# Multi-stage Dockerfile: builder installs deps, executor runs the app.
#
# Base image: python:3.14-slim (glibc-based Debian).
#
# If you switch to python:3.14-alpine to save ~30 MB (compressed), note:
#   - Native-extension packages (numpy, pandas, pyarrow — transitive deps of
#     streamlit and langchain) have no pre-built musllinux wheels.
#   - uv will compile them from source, requiring build deps in the builder
#     stage (gcc, musl-dev, libffi-dev, make) and adding 10-30 min to the build.
#   - Some packages may fail to compile on musl at all.
#   - For now, slim is the safe, fast choice.
# =============================================================================

# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# STAGE 1 — Builder: install runtime dependencies into .venv
# ---------------------------------------------------------------------------
FROM python:3.14-slim AS builder

# Install uv from the official image — single static binary, zero overhead
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only dependency manifests (leverages Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install runtime deps only:
#   --frozen          : exact versions from uv.lock (reproducible build)
#   --no-dev          : exclude pytest, black, jupyterlab, etc.
#   --no-install-project : skip building/installing the project wheel
RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project

# ---------------------------------------------------------------------------
# STAGE 2 — Executor: minimal runtime image
# ---------------------------------------------------------------------------
FROM python:3.14-slim

# Security: create a non-root user
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --home-dir /app --no-create-home appuser

WORKDIR /app

# Copy the complete virtualenv from builder (all runtime packages)
COPY --from=builder /app/.venv /app/.venv

# Copy uv binary (needed for uv run in the entrypoint)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy pyproject.toml (needed by uv run to resolve the project context)
COPY pyproject.toml ./

# Copy application source code
COPY src/ /app/src/

# Copy and set up the entrypoint
COPY ./scripts/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Environment:
#   VIRTUAL_ENV     — tells uv run which venv to use
#   PATH            — make venv binaries (uvicorn, streamlit) directly callable
#   PYTHONPATH      — make src/ modules importable without installing the wheel
#   PYTHONUNBUFFERED — stream logs to Docker without buffering
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1

# Give the non-root user ownership (streamlit needs write access for its cache)
RUN chown -R appuser:appgroup /app

USER appuser

# Document container ports:
#   8000 — FastAPI / uvicorn backend
#   8501 — Streamlit frontend
EXPOSE 8000 8501

ENTRYPOINT ["/app/docker-entrypoint.sh"]
