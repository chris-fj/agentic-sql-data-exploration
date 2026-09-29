"""LLM connection utilities — returns langchain chat-model instances configured
from environment variables or explicit parameters.

Two backends are supported:

* **cloud** — DeepSeek via ``langchain-deepseek`` (``ChatDeepSeek``).
* **local** — Ollama via ``langchain-ollama`` (``ChatOllama``).

Each backend reads its own ``CLOUD_*`` / ``LOCAL_*`` environment variables; the
dispatcher :func:`get_llm` picks the right one based on the ``type_`` argument.

Environment variables
---------------------
CLOUD_MODEL : str
    Model identifier (default ``"deepseek-chat"``).
CLOUD_API_ENDPOINT : str
    Base URL of the DeepSeek API (default ``"https://api.deepseek.com/v1"``).
CLOUD_API_KEY : str
    API key for authentication.
CLOUD_TEMPERATURE : float, optional
    Sampling temperature (default ``0.0``).
CLOUD_MAX_TOKENS : int, optional
    Maximum tokens in the completion (default ``4096``).
CLOUD_TIMEOUT : float, optional
    Request timeout in seconds (default ``60.0``).
CLOUD_MAX_RETRIES : int, optional
    Maximum retries on transient failures (default ``2``).

LOCAL_MODEL : str
    Model identifier (default ``"llama3.1:8b"``).
LOCAL_API_ENDPOINT : str
    Base URL of the Ollama API (default ``"http://localhost:11434"``).
LOCAL_API_KEY : str
    API key for the local endpoint (default ``""``).
LOCAL_TEMPERATURE : float, optional
    Sampling temperature (default ``0.0``).
LOCAL_MAX_TOKENS : int, optional
    Maximum tokens in the completion (default ``4096``).
LOCAL_TIMEOUT : float, optional
    Request timeout in seconds (default ``60.0``).
LOCAL_MAX_RETRIES : int, optional
    Maximum retries on transient failures (default ``2``).  Note: this value
    is **not** forwarded to ``ChatOllama``, which does not expose a
    ``max_retries`` parameter, so it has no effect.
"""

import os
from typing import (
    Any,
    Literal,
)

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

# Load .env once at import time so every caller picks up the same values.
load_dotenv()


def get_cloud_llm(
    model_name: str,
    endpoint: str,
    api_key: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> ChatDeepSeek:
    """Return a :class:`~langchain_deepseek.ChatDeepSeek` instance configured
    for the DeepSeek cloud API.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g. ``"deepseek-chat"``).
    endpoint : str
        Base URL of the DeepSeek API.
    api_key : str
        API key for authentication.
    temperature : float
        Sampling temperature (0.0 = deterministic).
    max_tokens : int
        Maximum tokens in the completion.
    timeout : float
        Request timeout in seconds.
    max_retries : int
        Maximum retries on transient failures.

    Returns
    -------
    ChatDeepSeek
    """
    return ChatDeepSeek(
        model=model_name,
        api_base=endpoint,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        extra_body={"thinking": {"type": "disabled"}},
    )


def get_local_llm(
    model_name: str,
    endpoint: str,
    api_key: str = "",
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: float = 60.0,
) -> ChatOllama:
    """Return a :class:`~langchain_ollama.ChatOllama` instance configured
    for a local Ollama server.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g. ``"llama3.1:8b"``).
    endpoint : str
        Base URL of the Ollama API.
    api_key : str
        API key for the endpoint.  Only forwarded to ``ChatOllama`` when
        non-empty; leave as the default (``""``) when no authentication is
        required.
    temperature : float
        Sampling temperature (0.0 = deterministic).
    max_tokens : int
        Maximum tokens in the completion (mapped to ``num_predict``).
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    ChatOllama

    Notes
    -----
    ``ChatOllama`` does not expose ``max_retries``.  The ``max_tokens``
    parameter is forwarded as ``num_predict``, which is the Ollama-native
    parameter name.
    """
    kwargs: dict = {
        "model": model_name,
        "base_url": endpoint,
        "temperature": temperature,
        "num_predict": max_tokens,
        "timeout": timeout,
    }
    if api_key:
        kwargs["api_key"] = api_key
    return ChatOllama(**kwargs)


def _optional_env_kwargs(*specs: tuple[str, type, str]) -> dict[str, Any]:
    """Return kwargs for optional numeric environment variables.

    Parameters
    ----------
    *specs : tuple[str, type, str]
        Each item is a ``(parameter, cast, env_var)`` tuple. Only variables
        that are actually set produce an entry, so the leaf constructor's
        documented defaults apply for unset variables.

    Returns
    -------
    dict[str, Any]
        Mapping of parameter names to the cast environment values.
    """
    kwargs: dict[str, Any] = {}
    for param, cast, env_var in specs:
        if (raw := os.getenv(env_var)) is not None:
            kwargs[param] = cast(raw)
    return kwargs


def get_llm(
    type_: Literal["local", "cloud"],
) -> ChatDeepSeek | ChatOllama:
    """Return a chat-model instance for the given backend type.

    Mandatory environment variables are validated before the model is
    constructed, so a missing mandatory variable raises ``OSError``
    immediately. Optional variables are forwarded only when set; otherwise
    the leaf constructor's documented defaults apply.

    Parameters
    ----------
    type_ : ``"local"`` or ``"cloud"``
        Which backend to configure.

    Returns
    -------
    ChatDeepSeek or ChatOllama
        Configured chat-model instance ready for use.

    Raises
    ------
    ValueError
        If *type_* is not ``"local"`` or ``"cloud"``.
    OSError
        If a mandatory environment variable is not set.

    Environment variables
    ---------------------
    CLOUD_MODEL, CLOUD_API_ENDPOINT, CLOUD_API_KEY
    CLOUD_TEMPERATURE, CLOUD_MAX_TOKENS, CLOUD_TIMEOUT, CLOUD_MAX_RETRIES
    LOCAL_MODEL, LOCAL_API_ENDPOINT, LOCAL_API_KEY
    LOCAL_TEMPERATURE, LOCAL_MAX_TOKENS, LOCAL_TIMEOUT, LOCAL_MAX_RETRIES
    """
    if type_ not in ["cloud", "local"]:
        raise ValueError(f"Unknown LLM type: {type_!r}.  Expected 'local' or 'cloud'.")

    if type_ == "cloud":
        mandatory = ("CLOUD_MODEL", "CLOUD_API_ENDPOINT", "CLOUD_API_KEY")
        missing = [name for name in mandatory if os.getenv(name) is None]
        if missing:
            raise OSError(
                f"Required environment variable(s) {', '.join(missing)} not set."
            )

        return get_cloud_llm(
            model_name=os.environ["CLOUD_MODEL"],
            endpoint=os.environ["CLOUD_API_ENDPOINT"],
            api_key=os.environ["CLOUD_API_KEY"],
            **_optional_env_kwargs(
                ("temperature", float, "CLOUD_TEMPERATURE"),
                ("max_tokens", int, "CLOUD_MAX_TOKENS"),
                ("timeout", float, "CLOUD_TIMEOUT"),
                ("max_retries", int, "CLOUD_MAX_RETRIES"),
            ),
        )

    # type_ == "local"
    mandatory = ("LOCAL_MODEL", "LOCAL_API_ENDPOINT")
    missing = [name for name in mandatory if os.getenv(name) is None]
    if missing:
        raise OSError(f"Required environment variable(s) {', '.join(missing)} not set.")

    return get_local_llm(
        model_name=os.environ["LOCAL_MODEL"],
        endpoint=os.environ["LOCAL_API_ENDPOINT"],
        api_key=os.getenv("LOCAL_API_KEY", ""),
        **_optional_env_kwargs(
            ("temperature", float, "LOCAL_TEMPERATURE"),
            ("max_tokens", int, "LOCAL_MAX_TOKENS"),
            ("timeout", float, "LOCAL_TIMEOUT"),
        ),
    )
