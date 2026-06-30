from fastapi import APIRouter

from .request_model import EchoRequest
from .response_model import EchoResponse

router = APIRouter()


@router.post("/echo", response_model=EchoResponse)
async def echo(request: EchoRequest) -> EchoResponse:
    """Echo back the text sent by the client."""
    return EchoResponse(echo=request.text)
