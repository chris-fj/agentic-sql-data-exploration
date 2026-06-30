import json

from fastapi import APIRouter

from backend.utils.llm import get_llm

from .request_model import LLMRequest
from .response_model import LLMResponse

router = APIRouter()


@router.post("/call_structured_llm", response_model=LLMResponse)
async def call_structured_llm(request: LLMRequest) -> LLMResponse:
    """Call an LLM with an optional structured output schema."""
    print(request)
    llm_instance = get_llm(request.llm)

    if request.structure_output is not None:
        structured_llm = llm_instance.with_structured_output(request.structure_output)
        result = await structured_llm.ainvoke(request.prompt)
        output_text = result if isinstance(result, str) else json.dumps(result)
    else:
        result = await llm_instance.ainvoke(request.prompt)
        output_text = result.content

    return LLMResponse(output=output_text)
