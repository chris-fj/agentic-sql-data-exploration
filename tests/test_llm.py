"""Tests for the LLM connection utility."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from backend.utils.llm import get_llm


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _set_env(monkeypatch: pytest.MonkeyPatch, **kwargs: str) -> None:
    """Set ``LLM_*`` environment variables for the duration of a test."""
    for key, value in kwargs.items():
        monkeypatch.setenv(key, value)


# ---------------------------------------------------------------------------
# Tests — get_llm construction
# ---------------------------------------------------------------------------

class TestGetLLM:
    """Verify that ``get_llm()`` returns a correctly-configured ChatOpenAI."""

    def test_returns_chat_openai_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = get_llm()
        assert isinstance(llm, ChatOpenAI)

    def test_reads_model_name_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, LLM_MODEL_NAME="custom-model")
        llm = get_llm()
        assert llm.model_name == "custom-model"

    def test_reads_endpoint_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, LLM_ENDPOINT="https://llm.example.com/v1")
        llm = get_llm()
        assert str(llm.openai_api_base) == "https://llm.example.com/v1"

    def test_reads_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, LLM_API_KEY="sk-test-key")
        llm = get_llm()
        assert llm.openai_api_key.get_secret_value() == "sk-test-key"

    def test_reads_temperature_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, LLM_TEMPERATURE="0.7")
        llm = get_llm()
        assert llm.temperature == 0.7

    def test_reads_max_tokens_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, LLM_MAX_TOKENS="2048")
        llm = get_llm()
        assert llm.max_tokens == 2048

    def test_reads_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, LLM_TIMEOUT="30.0")
        llm = get_llm()
        assert llm.request_timeout == 30.0

    def test_reads_max_retries_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, LLM_MAX_RETRIES="5")
        llm = get_llm()
        assert llm.max_retries == 5

    def test_falls_back_to_defaults_when_env_not_set(self) -> None:
        """All LLM_* vars were purged by the ``_isolate_env`` fixture."""
        llm = get_llm()
        assert llm.model_name == "gpt-4o"
        assert llm.temperature == 0.0
        assert llm.max_tokens == 4096


# ---------------------------------------------------------------------------
# Tests — mocked invocation
# ---------------------------------------------------------------------------

class TestLLMInvocation:
    """Verify that the LLM returned by ``get_llm()`` can be invoked and that
    its responses can be mocked."""

    def test_mocked_invoke_returns_ai_message(self) -> None:
        """Mock ``_generate`` to return a controlled response."""
        chat_result = ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="Hello from the mock!")),
            ],
        )

        llm = get_llm()
        with patch.object(llm, "_generate", return_value=chat_result):
            result = llm.invoke("Say hello")
            assert result.content == "Hello from the mock!"

    def test_mocked_stream_returns_chunks(self) -> None:
        """Mock ``_stream`` to verify streaming returns proper chunks."""
        chunks_in = [
            ChatGenerationChunk(message=AIMessageChunk(content="Hello")),
            ChatGenerationChunk(message=AIMessageChunk(content=" world!")),
        ]

        llm = get_llm()
        with patch.object(llm, "_stream", return_value=iter(chunks_in)):
            result = list(llm.stream("Say hello"))
            # langchain appends a trailing empty chunk with
            # chunk_position="last" and the final merge — at minimum we
            # should see the two content-bearing chunks we provided.
            assert len(result) >= 2
            assert result[0].content == "Hello"
            assert result[1].content == " world!"
