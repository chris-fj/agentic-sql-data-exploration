"""Tests for the prompt enhancing utilities."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend.api.model.prompt_enhancing_model import (
    ClarifyingUserIntent,
    ContextBlock,
    UserIntent,
)
from backend.utils.prompts import build_enhanced_output, clarify_user_intent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_httpx_response(output_dict: dict) -> MagicMock:
    """Build a mock ``httpx.Response`` whose ``.json()`` returns
    ``{"output": json.dumps(output_dict)}``."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"output": json.dumps(output_dict)}
    return mock_resp


def _make_intent(**overrides: object) -> ClarifyingUserIntent:
    """Build a ``ClarifyingUserIntent`` with sensible defaults, overridable
    per field."""
    defaults: dict[str, object] = {
        "type_of_request": UserIntent.QUESTION,
        "core_intent": "Explain what Docker is",
        "keywords": ["docker", "containers"],
        "style": "prose",
        "output_format": "markdown",
        "tone": "neutral",
        "additional_instructions": "",
        "context_blocks": [],
    }
    defaults.update(overrides)
    return ClarifyingUserIntent(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests — clarify_user_intent
# ---------------------------------------------------------------------------


class TestClarifyUserIntent:
    """Verify that ``clarify_user_intent()`` calls the structured-LLM endpoint
    and returns a ``ClarifyingUserIntent`` parsed from the response."""

    async def test_returns_clarifying_user_intent(self) -> None:
        """Happy path: mock ``httpx.AsyncClient.post`` to return a controlled
        response and verify the returned object has the expected field values."""
        # Arrange
        output_dict = {
            "type_of_request": "question",
            "core_intent": "Explain what Docker is",
            "keywords": ["docker", "containers"],
            "style": "prose",
            "output_format": "markdown",
            "tone": "neutral",
            "additional_instructions": "",
            "context_blocks": [],
        }

        # Act
        with patch.object(
            httpx.AsyncClient,
            "post",
            AsyncMock(return_value=_mock_httpx_response(output_dict)),
        ):
            result = await clarify_user_intent("What is Docker?")

        # Assert
        assert isinstance(result, ClarifyingUserIntent)
        assert result.core_intent == "Explain what Docker is"
        assert result.type_of_request == UserIntent.QUESTION
        assert result.keywords == ["docker", "containers"]

    async def test_sends_prompt_and_schema_in_request(self) -> None:
        """The POST body must include the enhancing prompt (concatenated with
        the raw query) and the ``ClarifyingUserIntent`` JSON schema."""
        # Arrange
        raw_query = "What is Docker?"
        output_dict = {
            "type_of_request": "question",
            "core_intent": "Explain what Docker is",
            "keywords": [],
            "style": "prose",
            "output_format": "markdown",
            "tone": "neutral",
            "additional_instructions": "",
            "context_blocks": [],
        }

        # Act
        with patch.object(
            httpx.AsyncClient,
            "post",
            AsyncMock(return_value=_mock_httpx_response(output_dict)),
        ) as mock_post:
            await clarify_user_intent(raw_query)

        # Assert
        call_kwargs = mock_post.call_args.kwargs
        body = call_kwargs["json"]
        assert "prompt" in body
        assert raw_query in body["prompt"]
        assert "output_structure" in body
        assert body["output_structure"] == ClarifyingUserIntent.model_json_schema()

    async def test_propagates_http_error(self) -> None:
        """When ``httpx.AsyncClient.post`` raises ``httpx.HTTPStatusError``,
        it should propagate to the caller."""
        # Arrange
        with patch.object(
            httpx.AsyncClient,
            "post",
            AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Server error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
            ),
        ):
            # Act & Assert
            with pytest.raises(httpx.HTTPStatusError):
                await clarify_user_intent("What is Docker?")


# ---------------------------------------------------------------------------
# Tests — build_enhanced_output
# ---------------------------------------------------------------------------


class TestBuildEnhancedOutput:
    """Verify that ``build_enhanced_output()`` renders the Jinja2 template
    correctly from a ``ClarifyingUserIntent``."""

    def test_minimal_intent(self) -> None:
        """Only required fields: output contains the core intent and the
        output-format line, but no optional sections."""
        # Arrange
        intent = _make_intent()

        # Act
        output = build_enhanced_output(intent)

        # Assert
        assert "Explain what Docker is" in output
        assert "Respond in markdown format." in output
        assert "Additional instructions:" not in output
        assert "Tone:" not in output
        assert "Presentation style:" not in output

    def test_includes_additional_instructions_when_non_empty(self) -> None:
        """When ``additional_instructions`` is set, it appears in the output."""
        # Arrange
        intent = _make_intent(additional_instructions="Keep it under 100 words.")

        # Act
        output = build_enhanced_output(intent)

        # Assert
        assert "Additional instructions: Keep it under 100 words." in output

    def test_includes_tone_when_not_neutral(self) -> None:
        """When ``tone`` is not ``'neutral'``, it appears in the output."""
        # Arrange
        intent = _make_intent(tone="formal")

        # Act
        output = build_enhanced_output(intent)

        # Assert
        assert "Tone: formal" in output

    def test_omits_tone_when_neutral(self) -> None:
        """When ``tone`` is ``'neutral'`` (the default), no tone line appears."""
        # Arrange
        intent = _make_intent(tone="neutral")

        # Act
        output = build_enhanced_output(intent)

        # Assert
        assert "Tone:" not in output

    def test_includes_style_when_not_prose(self) -> None:
        """When ``style`` is not ``'prose'``, it appears in the output."""
        # Arrange
        intent = _make_intent(style="bullet points")

        # Act
        output = build_enhanced_output(intent)

        # Assert
        assert "Presentation style: bullet points" in output

    def test_omits_style_when_prose(self) -> None:
        """When ``style`` is ``'prose'`` (the default), no style line appears."""
        # Arrange
        intent = _make_intent(style="prose")

        # Act
        output = build_enhanced_output(intent)

        # Assert
        assert "Presentation style:" not in output

    def test_renders_code_context_block_with_fences(self) -> None:
        """A ``ContextBlock`` with ``block_type='code'`` is rendered inside
        fenced code blocks with the language marker."""
        # Arrange
        intent = _make_intent(
            context_blocks=[
                ContextBlock(
                    block_type="code",
                    language="python",
                    content="print('hello')",
                ),
            ],
        )

        # Act
        output = build_enhanced_output(intent)

        # Assert
        assert "```python" in output
        assert "print('hello')" in output

    def test_renders_text_context_block_as_plain(self) -> None:
        """A ``ContextBlock`` with ``block_type='text'`` is rendered as plain
        text without code fences."""
        # Arrange
        intent = _make_intent(
            context_blocks=[
                ContextBlock(
                    block_type="text",
                    content="Some quoted passage.",
                ),
            ],
        )

        # Act
        output = build_enhanced_output(intent)

        # Assert
        assert "```" not in output
        assert "Some quoted passage." in output

    def test_full_intent_with_all_fields(self) -> None:
        """All fields populated with multiple context blocks of different
        types — verify the complete output structure."""
        # Arrange
        intent = _make_intent(
            type_of_request=UserIntent.GENERATE,
            core_intent="Generate a Python script to sort a list",
            keywords=["python", "sorting", "algorithm"],
            style="step-by-step instructions",
            output_format="python code",
            tone="technical",
            additional_instructions="Use type hints and include docstrings.",
            context_blocks=[
                ContextBlock(
                    block_type="code",
                    language="python",
                    content="def sort_list(items):\n    pass",
                ),
                ContextBlock(
                    block_type="text",
                    content="The function should handle empty lists.",
                ),
            ],
        )

        # Act
        output = build_enhanced_output(intent)

        # Assert
        assert "Generate a Python script to sort a list" in output
        assert "Respond in python code format." in output
        assert "Additional instructions: Use type hints and include docstrings." in output
        assert "Tone: technical" in output
        assert "Presentation style: step-by-step instructions" in output
        # Code block
        assert "```python" in output
        assert "def sort_list(items):" in output
        assert "    pass" in output
        # Text block
        assert "The function should handle empty lists." in output
