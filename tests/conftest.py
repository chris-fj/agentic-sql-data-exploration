"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove LLM_* env vars from the test environment so that local ``.env``
    files don't leak into tests."""
    for key in ("LLM_MODEL_NAME", "LLM_ENDPOINT", "LLM_API_KEY",
                 "LLM_TEMPERATURE", "LLM_MAX_TOKENS", "LLM_TIMEOUT",
                 "LLM_MAX_RETRIES"):
        monkeypatch.delenv(key, raising=False)
