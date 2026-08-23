"""Timing utilities for graph node execution.

This module provides a ``timed_node`` decorator that logs
each node's start, elapsed wall-clock time, and any failures.
It works with both sync and async node functions.
"""

import inspect
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def timed_node(name: str):
    """Decorator that logs the execution time of a graph node.

    Usage::

        @timed_node("generate_sql")
        async def generate_sql_node(state: AgentState) -> dict:
            ...

    Emits:
        ``INFO  [generate_sql] started``
        ``INFO  [generate_sql] completed in 1.234s``
        ``ERROR [generate_sql] failed after 0.567s: <exception>``
    """

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(state):
                logger.info("[%s] started", name)
                start = time.perf_counter()
                try:
                    result = await func(state)
                    elapsed = time.perf_counter() - start
                    logger.info("[%s] completed in %.3fs", name, elapsed)
                    return result
                except Exception:
                    elapsed = time.perf_counter() - start
                    logger.exception("[%s] failed after %.3fs", name, elapsed)
                    raise

            return async_wrapper

        @wraps(func)
        def sync_wrapper(state):
            logger.info("[%s] started", name)
            start = time.perf_counter()
            try:
                result = func(state)
                elapsed = time.perf_counter() - start
                logger.info("[%s] completed in %.3fs", name, elapsed)
                return result
            except Exception:
                elapsed = time.perf_counter() - start
                logger.exception("[%s] failed after %.3fs", name, elapsed)
                raise

        return sync_wrapper

    return decorator
