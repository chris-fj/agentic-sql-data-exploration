"""Tests for the LLM connection utilities."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

from backend.utils.llm import get_cloud_llm, get_llm, get_local_llm

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _set_env(monkeypatch: pytest.MonkeyPatch, **kwargs: str) -> None:
    """Set environment variables for the duration of a test."""
    for key, value in kwargs.items():
        monkeypatch.setenv(key, value)


# ---------------------------------------------------------------------------
# Tests — get_cloud_llm
# ---------------------------------------------------------------------------


class TestGetCloudLLM:
    """Verify that ``get_cloud_llm()`` returns a correctly-configured
    ``ChatDeepSeek``."""

    def test_returns_chat_deepseek_instance(self) -> None:
        """The function should return a ``ChatDeepSeek`` object."""
        llm = get_cloud_llm(
            model_name="deepseek-chat",
            endpoint="https://api.deepseek.com/v1",
            api_key="sk-test",
        )
        assert isinstance(llm, ChatDeepSeek)

    def test_passes_parameters_to_constructor(self) -> None:
        """All parameters should be forwarded to the ``ChatDeepSeek``
        constructor and accessible as instance attributes."""
        # Arrange
        model_name = "deepseek-reasoner"
        endpoint = "https://custom.deepseek.example.com/v1"
        api_key = "sk-custom-key"

        # Act
        llm = get_cloud_llm(
            model_name=model_name,
            endpoint=endpoint,
            api_key=api_key,
            temperature=0.7,
            max_tokens=2048,
            timeout=30.0,
            max_retries=5,
        )

        # Assert
        assert llm.model_name == model_name
        assert llm.api_base == endpoint
        assert llm.api_key.get_secret_value() == api_key
        assert llm.temperature == 0.7
        assert llm.max_tokens == 2048
        assert llm.request_timeout == 30.0
        assert llm.max_retries == 5

    def test_uses_defaults_for_optional_params(self) -> None:
        """When only required arguments are supplied, sensible defaults should
        be used for temperature, max_tokens, timeout, and max_retries."""
        # Act
        llm = get_cloud_llm(
            model_name="deepseek-chat",
            endpoint="https://api.deepseek.com/v1",
            api_key="sk-test",
        )

        # Assert
        assert llm.temperature == 0.0
        assert llm.max_tokens == 4096
        assert llm.request_timeout == 60.0
        assert llm.max_retries == 2


# ---------------------------------------------------------------------------
# Tests — get_local_llm
# ---------------------------------------------------------------------------


class TestGetLocalLLM:
    """Verify that ``get_local_llm()`` returns a correctly-configured
    ``ChatOllama``."""

    def test_returns_chat_ollama_instance(self) -> None:
        """The function should return a ``ChatOllama`` object."""
        llm = get_local_llm(
            model_name="llama3.1:8b",
            endpoint="http://localhost:11434",
        )
        assert isinstance(llm, ChatOllama)

    def test_passes_parameters_to_constructor(self) -> None:
        """All parameters should be forwarded to the ``ChatOllama``
        constructor and accessible as instance attributes.

        Note: ``timeout`` is consumed internally by ``ChatOllama`` and is not
        exposed as a public attribute, so it is not asserted here.
        """
        # Arrange
        model_name = "mistral:7b"
        endpoint = "http://ollama.local:11434"

        # Act
        llm = get_local_llm(
            model_name=model_name,
            endpoint=endpoint,
            api_key="secret",
            temperature=0.3,
            max_tokens=1024,
            timeout=15.0,
        )

        # Assert
        assert llm.model == model_name
        assert llm.base_url == endpoint
        assert llm.temperature == 0.3
        assert llm.num_predict == 1024

    def test_uses_defaults_for_optional_params(self) -> None:
        """When only required arguments are supplied, sensible defaults should
        be used for temperature, max_tokens, and timeout."""
        # Act
        llm = get_local_llm(
            model_name="llama3.1:8b",
            endpoint="http://localhost:11434",
        )

        # Assert
        assert llm.temperature == 0.0
        assert llm.num_predict == 4096

    def test_omits_api_key_when_empty(self) -> None:
        """When *api_key* is empty (the default), it should not be forwarded
        to the ``ChatOllama`` constructor."""
        # Arrange / Act
        with patch("backend.utils.llm.ChatOllama", wraps=ChatOllama) as mock_ollama:
            get_local_llm(
                model_name="llama3.1:8b",
                endpoint="http://localhost:11434",
            )
            # Assert
            _, kwargs = mock_ollama.call_args
            assert "api_key" not in kwargs

    def test_api_key_passed_when_non_empty(self) -> None:
        """When *api_key* is a non-empty string, it should be forwarded to
        the ``ChatOllama`` constructor."""
        # Arrange / Act
        with patch("backend.utils.llm.ChatOllama", wraps=ChatOllama) as mock_ollama:
            get_local_llm(
                model_name="llama3.1:8b",
                endpoint="http://localhost:11434",
                api_key="my-secret",
            )
            # Assert
            _, kwargs = mock_ollama.call_args
            assert kwargs.get("api_key") == "my-secret"


# ---------------------------------------------------------------------------
# Tests — get_llm dispatcher
# ---------------------------------------------------------------------------


class TestGetLLM:
    """Verify that ``get_llm()`` reads the correct environment variables and
    dispatches to the right leaf function."""

    # -- cloud path ----------------------------------------------------------

    def test_cloud_calls_get_cloud_llm_with_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``type_="cloud"``, ``get_llm`` should call ``get_cloud_llm``
        with the values from ``CLOUD_*`` environment variables."""
        # Arrange
        _set_env(
            monkeypatch,
            CLOUD_MODEL="deepseek-reasoner",
            CLOUD_API_ENDPOINT="https://deepseek.custom.example.com",
            CLOUD_API_KEY="sk-env-key",
            CLOUD_TEMPERATURE="0.5",
            CLOUD_MAX_TOKENS="1024",
            CLOUD_TIMEOUT="15.0",
            CLOUD_MAX_RETRIES="3",
        )

        # Act
        with patch("backend.utils.llm.get_cloud_llm") as mock_cloud:
            mock_cloud.return_value = "fake-llm"
            result = get_llm("cloud")

        # Assert
        mock_cloud.assert_called_once_with(
            model_name="deepseek-reasoner",
            endpoint="https://deepseek.custom.example.com",
            api_key="sk-env-key",
            temperature=0.5,
            max_tokens=1024,
            timeout=15.0,
            max_retries=3,
        )
        assert result == "fake-llm"

    def test_cloud_falls_back_to_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When only mandatory ``CLOUD_*`` variables are set, ``get_llm``
        forwards sentinel values for the optional parameters so that
        ``get_cloud_llm`` applies its own defaults.  The ``_isolate_env``
        fixture purges all env vars beforehand."""
        # Arrange — set mandatory vars so the guard passes, but leave
        # optional vars unset.
        _set_env(
            monkeypatch,
            CLOUD_MODEL="deepseek-chat",
            CLOUD_API_ENDPOINT="https://api.deepseek.com/v1",
            CLOUD_API_KEY="sk-test",
        )

        # Act
        with patch("backend.utils.llm.get_cloud_llm") as mock_cloud:
            mock_cloud.return_value = "fake-llm"
            result = get_llm("cloud")

        # Assert — sentinel values indicate "use the leaf default".
        mock_cloud.assert_called_once_with(
            model_name="deepseek-chat",
            endpoint="https://api.deepseek.com/v1",
            api_key="sk-test",
            temperature=0.0,
            max_tokens=-1,
            timeout=-1.0,
            max_retries=2,
        )
        assert result == "fake-llm"

    # -- local path ----------------------------------------------------------

    def test_local_calls_get_local_llm_with_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``type_="local"``, ``get_llm`` should call ``get_local_llm``
        with the values from ``LOCAL_*`` environment variables."""
        # Arrange
        _set_env(
            monkeypatch,
            LOCAL_MODEL="mistral:7b",
            LOCAL_API_ENDPOINT="http://ollama.local:11434",
            LOCAL_API_KEY="local-key",
            LOCAL_TEMPERATURE="0.8",
            LOCAL_MAX_TOKENS="512",
            LOCAL_TIMEOUT="10.0",
        )

        # Act
        with patch("backend.utils.llm.get_local_llm") as mock_local:
            mock_local.return_value = "fake-llm"
            result = get_llm("local")

        # Assert
        mock_local.assert_called_once_with(
            model_name="mistral:7b",
            endpoint="http://ollama.local:11434",
            api_key="local-key",
            temperature=0.8,
            max_tokens=512,
            timeout=10.0,
        )
        assert result == "fake-llm"

    def test_local_falls_back_to_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When only mandatory ``LOCAL_*`` variables are set, ``get_llm``
        forwards sentinel values for the optional parameters so that
        ``get_local_llm`` applies its own defaults."""
        # Arrange — set mandatory vars so the guard passes, but leave
        # optional vars unset.
        _set_env(
            monkeypatch,
            LOCAL_MODEL="llama3.1:8b",
            LOCAL_API_ENDPOINT="http://localhost:11434",
        )

        # Act
        with patch("backend.utils.llm.get_local_llm") as mock_local:
            mock_local.return_value = "fake-llm"
            result = get_llm("local")

        # Assert — sentinel values indicate "use the leaf default".
        mock_local.assert_called_once_with(
            model_name="llama3.1:8b",
            endpoint="http://localhost:11434",
            api_key="",
            temperature=0.0,
            max_tokens=-1,
            timeout=-1.0,
        )
        assert result == "fake-llm"

    # -- error path ----------------------------------------------------------

    def test_invalid_type_raises(self) -> None:
        """Passing an unrecognized *type_* should raise ``ValueError``."""
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown LLM type"):
            get_llm("invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests — mocked invocation
# ---------------------------------------------------------------------------


class TestLLMInvocation:
    """Verify that LLM instances returned by the utility functions can be
    invoked and streamed without making real network calls."""

    # -- cloud invocation ----------------------------------------------------

    def test_cloud_mocked_invoke_returns_ai_message(self) -> None:
        """Mock ``_generate`` on a ``ChatDeepSeek`` instance to return a
        controlled ``AIMessage``."""
        # Arrange
        chat_result = ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="Hello from DeepSeek mock!")),
            ],
        )
        llm = get_cloud_llm(
            model_name="test",
            endpoint="http://test.example.com",
            api_key="test-key",
        )

        # Act
        with patch.object(llm, "_generate", return_value=chat_result):
            result = llm.invoke("Say hello")

        # Assert
        assert isinstance(result, AIMessage)
        assert result.content == "Hello from DeepSeek mock!"

    def test_cloud_mocked_stream_returns_chunks(self) -> None:
        """Mock ``_stream`` on a ``ChatDeepSeek`` instance to yield
        controlled ``ChatGenerationChunk`` objects."""
        # Arrange
        chunks_in = [
            ChatGenerationChunk(message=AIMessageChunk(content="Hello")),
            ChatGenerationChunk(message=AIMessageChunk(content=" world!")),
        ]
        llm = get_cloud_llm(
            model_name="test",
            endpoint="http://test.example.com",
            api_key="test-key",
        )

        # Act
        with patch.object(llm, "_stream", return_value=iter(chunks_in)):
            result = list(llm.stream("Say hello"))

        # Assert — at minimum the two content-bearing chunks are present.
        assert len(result) >= 2
        assert result[0].content == "Hello"
        assert result[1].content == " world!"

    # -- local invocation ----------------------------------------------------

    def test_local_mocked_invoke_returns_ai_message(self) -> None:
        """Mock ``_generate`` on a ``ChatOllama`` instance to return a
        controlled ``AIMessage``."""
        # Arrange
        chat_result = ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="Hello from Ollama mock!")),
            ],
        )
        llm = get_local_llm(
            model_name="test",
            endpoint="http://test.example.com",
        )

        # Act
        with patch.object(llm, "_generate", return_value=chat_result):
            result = llm.invoke("Say hello")

        # Assert
        assert isinstance(result, AIMessage)
        assert result.content == "Hello from Ollama mock!"

    def test_local_mocked_stream_returns_chunks(self) -> None:
        """Mock ``_stream`` on a ``ChatOllama`` instance to yield
        controlled ``ChatGenerationChunk`` objects."""
        # Arrange
        chunks_in = [
            ChatGenerationChunk(message=AIMessageChunk(content="Bonjour")),
            ChatGenerationChunk(message=AIMessageChunk(content=" le monde!")),
        ]
        llm = get_local_llm(
            model_name="test",
            endpoint="http://test.example.com",
        )

        # Act
        with patch.object(llm, "_stream", return_value=iter(chunks_in)):
            result = list(llm.stream("Say hello"))

        # Assert
        assert len(result) >= 2
        assert result[0].content == "Bonjour"
        assert result[1].content == " le monde!"
