"""LLM connection utilities.

Returns a single :class:`~langchain_openai.ChatOpenAI` instance configured
from environment variables. The endpoint is assumed to be OpenAI-compatible,
so cloud providers and self-hosted servers are configured the same way.

Environment variables
---------------------
LLM_MODEL : str
    Model identifier (required).
LLM_ENDPOINT : str
    OpenAI-compatible base URL (required).
LLM_API_KEY : str
    API key forwarded to ``ChatOpenAI``. No fallback to ``OPENAI_API_KEY`` is
    performed; if the endpoint does not require authentication, set this to a
    dummy non-empty value such as ``"not-needed"``.
LLM_KWARGS__<NAME> : str, optional
    Additional keyword arguments for ``ChatOpenAI``. The suffix after
    ``LLM_KWARGS__`` is lower-cased and used as the keyword name. Values are
    parsed as JSON when possible, otherwise kept as strings.
"""

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env once at import time so every caller picks up the same values.
load_dotenv()

_KWARGS_PREFIX = "LLM_KWARGS__"


def _kwargs_from_env() -> dict[str, Any]:
    """Build a kwargs dictionary from ``LLM_KWARGS__*`` environment variables.

    Returns
    -------
    dict[str, Any]
        Keyword arguments parsed from the environment. Each matching variable
        contributes one entry; the variable name suffix is lower-cased to form
        the key. Values are JSON-decoded when possible.
    """
    kwargs: dict[str, Any] = {}
    prefix_len = len(_KWARGS_PREFIX)

    for env_name, env_value in os.environ.items():
        if not env_name.startswith(_KWARGS_PREFIX):
            continue

        key = env_name[prefix_len:].lower()
        if not key:
            continue

        try:
            value: Any = json.loads(env_value)
        except json.JSONDecodeError:
            value = env_value

        kwargs[key] = value

    return kwargs


def get_llm() -> ChatOpenAI:
    """Return a :class:`~langchain_openai.ChatOpenAI` configured from env vars.

    Returns
    -------
    ChatOpenAI
        Configured chat-model instance ready for use.

    Raises
    ------
    OSError
        If ``LLM_MODEL``, ``LLM_ENDPOINT`` or ``LLM_API_KEY`` is not set.
    """
    model = os.getenv("LLM_MODEL")
    endpoint = os.getenv("LLM_ENDPOINT")
    api_key = os.getenv("LLM_API_KEY")

    missing = [
        name
        for name, value in (
            ("LLM_MODEL", model),
            ("LLM_ENDPOINT", endpoint),
            ("LLM_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        raise OSError(f"Required environment variable(s) {', '.join(missing)} not set.")

    kwargs = _kwargs_from_env()
    kwargs["model"] = model
    kwargs["base_url"] = endpoint
    kwargs["api_key"] = api_key

    return ChatOpenAI(**kwargs)
