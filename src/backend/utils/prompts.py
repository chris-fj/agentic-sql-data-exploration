"""Prompt templates used by the SQL agent StateGraph nodes.

The two-pass clarification pipeline (``clarify_user_intent`` →
``build_enhanced_output`` → second LLM call) has been folded into
the graph's ``clarify_intent_node`` and ``chat_response_node``.
"""

from jinja2 import Template

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
```
{%- else -%}
{{ block.content }}
{%- endif %}
{%- endfor %}"""
)
