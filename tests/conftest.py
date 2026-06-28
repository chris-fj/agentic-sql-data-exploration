"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove LLM-related env vars from the test environment so that local
    ``.env`` files don't leak into tests."""
    for key in (
        # Cloud backend
        "CLOUD_MODEL",
        "CLOUD_API_ENDPOINT",
        "CLOUD_API_KEY",
        "CLOUD_TEMPERATURE",
        "CLOUD_MAX_TOKENS",
        "CLOUD_TIMEOUT",
        "CLOUD_MAX_RETRIES",
        # Local backend
        "LOCAL_MODEL",
        "LOCAL_API_ENDPOINT",
        "LOCAL_API_KEY",
        "LOCAL_TEMPERATURE",
        "LOCAL_MAX_TOKENS",
        "LOCAL_TIMEOUT",
        "LOCAL_MAX_RETRIES",
    ):
        monkeypatch.delenv(key, raising=False)
