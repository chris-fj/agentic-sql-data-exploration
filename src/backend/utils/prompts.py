from backend.api.model.prompt_enhancing_model import ClarifyingUserIntent
from backend.api.prompts import USER_REQUEST_ENHANCING_PROMPT
from jinja2 import Template
import httpx
import json
import os

BACKEND_URL = os.getenv("BACKEND_URL")

ENHANCED_OUTPUT_TEMPLATE = Template(
    """\
{{ core_intent }}\
{%- if additional_instructions %}

Additional instructions: {{ additional_instructions }}
{%- endif %}
{%- if tone != 'neutral' %}

Tone: {{ tone }}
{%- endif %}
{%- if style != 'prose' %}

Presentation style: {{ style }}
{%- endif %}

Respond in {{ output_format }} format.
{%- for block in context_blocks %}

{% if block.block_type == 'code' -%}
```{{ block.language or '' }}
{{ block.content }}
{%- else -%}
{{ block.content }}
{%- endif %}
{%- endfor %}"""
)

def clarify_user_intent(
    raw_query: str,
    llm_type: str | None = None,
) -> ClarifyingUserIntent:
    """
    Enhances the user prompt

    Follows a structured approach to understand the request provided by the user without making up details.
    """

    enhancing_prompt = f"""\
    {USER_REQUEST_ENHANCING_PROMPT}

    {raw_query}
    """

    user_prompt_clarified = response = httpx.post(
        f"{BACKEND_URL}/api/structured-llm",
        json={
            "prompt": enhancing_prompt,
            "output_structure": ClarifyingUserIntent.model_json_schema(),
        },
        timeout=60,
    )

    structured_answer = json.loads(user_prompt_clarified.json()["output"])

    user_intent = ClarifyingUserIntent(**structured_answer)

    return user_intent


def build_enhanced_output(intent: ClarifyingUserIntent) -> str:
    """Build a deterministic prompt from a structured ClarifyingUserIntent instance."""
    return ENHANCED_OUTPUT_TEMPLATE.render(
        core_intent=intent.core_intent,
        type_of_request=intent.type_of_request.value,
        keywords=intent.keywords,
        style=intent.style,
        output_format=intent.output_format,
        tone=intent.tone,
        additional_instructions=intent.additional_instructions,
        context_blocks=[block.model_dump() for block in intent.context_blocks],
    )
