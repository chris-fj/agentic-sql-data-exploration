import logging

from fastapi import APIRouter

from backend.api.model.request_model import EchoRequest
from backend.api.model.response_model import EchoResponse

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/echo", response_model=EchoResponse)
async def echo(request: EchoRequest) -> EchoResponse:
    """Echo back the text sent by the client."""

    logger.info("Received request %s", str(request.model_dump_json()))
    return EchoResponse(echo=request.text)
