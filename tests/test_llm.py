"""Tests for the LLM connection utilities."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

from backend.utils.llm import (
    get_cloud_llm,
    get_llm,
    get_local_llm,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _set_env(monkeypatch: pytest.MonkeyPatch, **kwargs: str) -> None:
    """Set environment variables for the duration of a test."""
    for key, value in kwargs.items():
        monkeypatch.setenv(key, value)


_CLOUD_MANDATORY = {
    "CLOUD_MODEL": "deepseek-chat",
    "CLOUD_API_ENDPOINT": "https://api.deepseek.com/v1",
    "CLOUD_API_KEY": "sk-test",
}

_LOCAL_MANDATORY = {
    "LOCAL_MODEL": "llama3.1:8b",
    "LOCAL_API_ENDPOINT": "http://localhost:11434",
}


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
        """When only mandatory ``CLOUD_*`` variables are set, unset optional
        variables are omitted from the ``get_cloud_llm`` call so that the
        leaf function's documented defaults apply.  The ``_isolate_env``
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

        # Assert — only mandatory parameters are forwarded.
        mock_cloud.assert_called_once_with(
            model_name="deepseek-chat",
            endpoint="https://api.deepseek.com/v1",
            api_key="sk-test",
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
        """When only mandatory ``LOCAL_*`` variables are set, unset optional
        variables are omitted from the ``get_local_llm`` call so that the
        leaf function's documented defaults apply."""
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

        # Assert — only mandatory parameters are forwarded.
        mock_local.assert_called_once_with(
            model_name="llama3.1:8b",
            endpoint="http://localhost:11434",
            api_key="",
        )
        assert result == "fake-llm"

    # -- error path ----------------------------------------------------------

    def test_invalid_type_raises(self) -> None:
        """Passing an unrecognized *type_* should raise ``ValueError``."""
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown LLM type"):
            get_llm("invalid")  # type: ignore[arg-type]

    def test_cloud_applies_documented_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With mandatory variables set and optional ones unset, the
        constructed ``ChatDeepSeek`` should carry the documented defaults."""
        # Arrange
        _set_env(monkeypatch, **_CLOUD_MANDATORY)

        # Act
        llm = get_llm("cloud")

        # Assert
        assert isinstance(llm, ChatDeepSeek)
        assert llm.temperature == 0.0
        assert llm.max_tokens == 4096
        assert llm.request_timeout == 60.0
        assert llm.max_retries == 2

    def test_local_applies_documented_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With mandatory variables set and optional ones unset, the
        constructed ``ChatOllama`` should carry the documented defaults."""
        # Arrange
        _set_env(monkeypatch, **_LOCAL_MANDATORY)

        # Act
        llm = get_llm("local")

        # Assert
        assert isinstance(llm, ChatOllama)
        assert llm.temperature == 0.0
        assert llm.num_predict == 4096


class TestGetLLMValidationOrder:
    """Verify that mandatory variables are validated before the model is
    constructed, and that the error names every missing variable."""

    @pytest.mark.parametrize(
        ("backend", "leaf_module_attr", "mandatory_envs", "removed"),
        [
            (
                "cloud",
                "backend.utils.llm.get_cloud_llm",
                _CLOUD_MANDATORY,
                "CLOUD_MODEL",
            ),
            (
                "cloud",
                "backend.utils.llm.get_cloud_llm",
                _CLOUD_MANDATORY,
                "CLOUD_API_ENDPOINT",
            ),
            (
                "cloud",
                "backend.utils.llm.get_cloud_llm",
                _CLOUD_MANDATORY,
                "CLOUD_API_KEY",
            ),
            (
                "local",
                "backend.utils.llm.get_local_llm",
                _LOCAL_MANDATORY,
                "LOCAL_MODEL",
            ),
            (
                "local",
                "backend.utils.llm.get_local_llm",
                _LOCAL_MANDATORY,
                "LOCAL_API_ENDPOINT",
            ),
        ],
        ids=[
            "cloud-model",
            "cloud-endpoint",
            "cloud-api-key",
            "local-model",
            "local-endpoint",
        ],
    )
    def test_missing_mandatory_var_raises_before_construction(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend: str,
        leaf_module_attr: str,
        mandatory_envs: dict,
        removed: str,
    ) -> None:
        """Removing one mandatory variable should raise ``OSError`` naming it,
        without ever calling the leaf constructor."""
        # Arrange — set every mandatory variable except the one under test.
        _set_env(
            monkeypatch, **{k: v for k, v in mandatory_envs.items() if k != removed}
        )

        # Act & Assert
        with (
            patch(leaf_module_attr) as mock_leaf,
            pytest.raises(OSError, match=removed),
        ):
            get_llm(backend)  # type: ignore[arg-type]
        mock_leaf.assert_not_called()
