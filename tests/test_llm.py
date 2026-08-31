"""Tests for the LLM connection utilities."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_openai import ChatOpenAI

from backend.utils.llm import get_llm


class TestGetLLM:
    """Verify that ``get_llm()`` configures ``ChatOpenAI`` from env vars."""

    def test_returns_chat_openai_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The function should return a ``ChatOpenAI`` object."""
        # Arrange
        monkeypatch.setenv("LLM_MODEL", "gpt-4")
        monkeypatch.setenv("LLM_ENDPOINT", "http://example.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")

        # Act
        llm = get_llm()

        # Assert
        assert isinstance(llm, ChatOpenAI)

    def test_forwards_model_endpoint_and_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``LLM_MODEL``, ``LLM_ENDPOINT`` and ``LLM_API_KEY`` should be
        forwarded to ``ChatOpenAI``."""
        # Arrange
        monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
        monkeypatch.setenv("LLM_ENDPOINT", "https://api.deepseek.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")

        # Act
        llm = get_llm()

        # Assert
        assert llm.model_name == "deepseek-chat"
        assert llm.openai_api_base == "https://api.deepseek.com/v1"
        assert llm.openai_api_key.get_secret_value() == "sk-test"

    def test_parses_kwargs_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``LLM_KWARGS__*`` variables should be parsed and forwarded."""
        # Arrange
        monkeypatch.setenv("LLM_MODEL", "gpt-4")
        monkeypatch.setenv("LLM_ENDPOINT", "http://example.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_KWARGS__TEMPERATURE", "0.5")
        monkeypatch.setenv("LLM_KWARGS__MAX_TOKENS", "1024")
        monkeypatch.setenv("LLM_KWARGS__TIMEOUT", "30")
        monkeypatch.setenv("LLM_KWARGS__MAX_RETRIES", "5")
        monkeypatch.setenv(
            "LLM_KWARGS__EXTRA_BODY", '{"thinking": {"type": "disabled"}}'
        )

        # Act
        llm = get_llm()

        # Assert
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1024
        assert llm.request_timeout == 30.0
        assert llm.max_retries == 5
        assert llm.extra_body == {"thinking": {"type": "disabled"}}

    def test_missing_model_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing ``LLM_MODEL`` should raise ``OSError``."""
        # Arrange
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.setenv("LLM_ENDPOINT", "http://example.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")

        # Act & Assert
        with pytest.raises(OSError, match="LLM_MODEL"):
            get_llm()

    def test_missing_endpoint_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing ``LLM_ENDPOINT`` should raise ``OSError``."""
        # Arrange
        monkeypatch.setenv("LLM_MODEL", "gpt-4")
        monkeypatch.delenv("LLM_ENDPOINT", raising=False)
        monkeypatch.setenv("LLM_API_KEY", "sk-test")

        # Act & Assert
        with pytest.raises(OSError, match="LLM_ENDPOINT"):
            get_llm()

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing ``LLM_API_KEY`` should raise ``OSError``."""
        # Arrange
        monkeypatch.setenv("LLM_MODEL", "gpt-4")
        monkeypatch.setenv("LLM_ENDPOINT", "http://example.com/v1")
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        # Act & Assert
        with pytest.raises(OSError, match="LLM_API_KEY"):
            get_llm()

    def test_dummy_api_key_works_for_unauthenticated_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dummy non-empty key can be used when the endpoint needs no auth."""
        # Arrange
        monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
        monkeypatch.setenv("LLM_ENDPOINT", "http://localhost:11434/v1")
        monkeypatch.setenv("LLM_API_KEY", "not-needed")

        # Act
        llm = get_llm()

        # Assert
        assert llm.openai_api_key.get_secret_value() == "not-needed"

    def test_explicit_env_vars_override_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``LLM_MODEL``, ``LLM_ENDPOINT`` and ``LLM_API_KEY`` always win over
        matching ``LLM_KWARGS__*`` entries."""
        # Arrange
        monkeypatch.setenv("LLM_MODEL", "from-env-model")
        monkeypatch.setenv("LLM_ENDPOINT", "http://from-env.example.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "from-env-key")
        monkeypatch.setenv("LLM_KWARGS__MODEL", "from-kwargs-model")
        monkeypatch.setenv("LLM_KWARGS__BASE_URL", "http://from-kwargs.example.com/v1")
        monkeypatch.setenv("LLM_KWARGS__API_KEY", "from-kwargs-key")

        # Act
        llm = get_llm()

        # Assert
        assert llm.model_name == "from-env-model"
        assert llm.openai_api_base == "http://from-env.example.com/v1"
        assert llm.openai_api_key.get_secret_value() == "from-env-key"


class TestLLMInvocation:
    """Verify that the returned ``ChatOpenAI`` instance can be invoked and
    streamed without real network calls."""

    def test_mocked_invoke_returns_ai_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mock ``_generate`` to return a controlled ``AIMessage``."""
        # Arrange
        monkeypatch.setenv("LLM_MODEL", "gpt-4")
        monkeypatch.setenv("LLM_ENDPOINT", "http://example.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        chat_result = ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="Hello from mock!")),
            ],
        )
        llm = get_llm()

        # Act
        with patch.object(llm, "_generate", return_value=chat_result):
            result = llm.invoke("Say hello")

        # Assert
        assert isinstance(result, AIMessage)
        assert result.content == "Hello from mock!"

    def test_mocked_stream_returns_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mock ``_stream`` to yield controlled ``ChatGenerationChunk`` objects."""
        # Arrange
        monkeypatch.setenv("LLM_MODEL", "gpt-4")
        monkeypatch.setenv("LLM_ENDPOINT", "http://example.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        chunks_in = [
            ChatGenerationChunk(message=AIMessageChunk(content="Hello")),
            ChatGenerationChunk(message=AIMessageChunk(content=" world!")),
        ]
        llm = get_llm()

        # Act
        with patch.object(llm, "_stream", return_value=iter(chunks_in)):
            result = list(llm.stream("Say hello"))

        # Assert
        assert len(result) >= 2
        assert result[0].content == "Hello"
        assert result[1].content == " world!"
