from fastapi import APIRouter

from backend.api.model.request_model import EchoRequest
from backend.api.model.response_model import EchoResponse

router = APIRouter()


@router.post("/echo", response_model=EchoResponse)
async def echo(request: EchoRequest) -> EchoResponse:
    """Echo back the text sent by the client."""
    return EchoResponse(echo=request.text)
