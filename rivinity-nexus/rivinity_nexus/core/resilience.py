import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")
logger = logging.getLogger("rivinity_nexus.resilience")


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    delay_seconds: float = 0.25,
    op_name: str = "operation",
) -> T:
    last_exc: Exception | None = None
    for idx in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            logger.warning("retryable_failure", extra={"op": op_name, "attempt": idx, "error": str(exc)})
            if idx < attempts:
                time.sleep(delay_seconds * idx)
    assert last_exc is not None
    raise last_exc


async def retry_async_call(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    delay_seconds: float = 0.25,
    op_name: str = "operation",
) -> T:
    last_exc: Exception | None = None
    for idx in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            logger.warning("retryable_async_failure", extra={"op": op_name, "attempt": idx, "error": str(exc)})
            if idx < attempts:
                await asyncio.sleep(delay_seconds * idx)
    assert last_exc is not None
    raise last_exc
