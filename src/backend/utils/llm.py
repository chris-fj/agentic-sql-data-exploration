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
    is read but **not** forwarded to ``ChatOllama``, which does not expose a
    ``max_retries`` parameter.
"""

from __future__ import annotations

import os
from typing import Literal

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


def get_llm(
    type_: Literal["local", "cloud"],
) -> ChatDeepSeek | ChatOllama:
    """Return a chat-model instance for the given backend type.

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

    Environment variables
    ---------------------
    CLOUD_MODEL, CLOUD_API_ENDPOINT, CLOUD_API_KEY
    CLOUD_TEMPERATURE, CLOUD_MAX_TOKENS, CLOUD_TIMEOUT, CLOUD_MAX_RETRIES
    LOCAL_MODEL, LOCAL_API_ENDPOINT, LOCAL_API_KEY
    LOCAL_TEMPERATURE, LOCAL_MAX_TOKENS, LOCAL_TIMEOUT, LOCAL_MAX_RETRIES
    """
    if type_ == "cloud":
        return get_cloud_llm(
            model_name=os.getenv("CLOUD_MODEL", "deepseek-chat"),
            endpoint=os.getenv("CLOUD_API_ENDPOINT", "https://api.deepseek.com/v1"),
            api_key=os.getenv("CLOUD_API_KEY", ""),
            temperature=float(os.getenv("CLOUD_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("CLOUD_MAX_TOKENS", "4096")),
            timeout=float(os.getenv("CLOUD_TIMEOUT", "60.0")),
            max_retries=int(os.getenv("CLOUD_MAX_RETRIES", "2")),
        )

    if type_ == "local":
        return get_local_llm(
            model_name=os.getenv("LOCAL_MODEL", "llama3.1:8b"),
            endpoint=os.getenv("LOCAL_API_ENDPOINT", "http://localhost:11434"),
            api_key=os.getenv("LOCAL_API_KEY", ""),
            temperature=float(os.getenv("LOCAL_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("LOCAL_MAX_TOKENS", "4096")),
            timeout=float(os.getenv("LOCAL_TIMEOUT", "60.0")),
        )

    raise ValueError(f"Unknown LLM type: {type_!r}.  Expected 'local' or 'cloud'.")
