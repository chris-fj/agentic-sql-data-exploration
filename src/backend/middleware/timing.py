import logging
import time
from collections.abc import Awaitable
from collections.abc import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


class ToolTimingMiddleware(AgentMiddleware):
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_name = request.tool_call["name"]
        # logger = logging.getLogger(__name__)

        logger.info("Tool %s started", tool_name)
        start = time.perf_counter()
        result = handler(request)
        elapsed = time.perf_counter() - start
        logger.info("Tool %s completed in %.3fs", tool_name, elapsed)

        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        tool_name = request.tool_call["name"]
        # logger = logging.getLogger(__name__)

        logger.info("Tool %s started", tool_name)
        start = time.perf_counter()
        result = await handler(request)
        elapsed = time.perf_counter() - start
        logger.info("Tool %s completed in %.3fs", tool_name, elapsed)
        print("This message was printed with `print`")

        return result
