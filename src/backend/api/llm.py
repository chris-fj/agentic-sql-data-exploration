import json

from fastapi import APIRouter

from backend.utils.llm import get_llm

from backend.api.model.request_model import LLMRequest
from backend.api.model.response_model import LLMResponse

router = APIRouter()


@router.post("/structured-llm", response_model=LLMResponse)
async def call_structured_llm(request: LLMRequest) -> LLMResponse:
    """Call an LLM with an optional structured output schema."""
    print(request)
    llm_instance = get_llm(request.llm)

    if request.output_structure is not None:
        structured_llm = llm_instance.with_structured_output(request.output_structure)
        result = await structured_llm.ainvoke(request.prompt)
        output_text = result if isinstance(result, str) else json.dumps(result)
    else:
        result = await llm_instance.ainvoke(request.prompt)
        output_text = result.content

    return LLMResponse(output=output_text)
