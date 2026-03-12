import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        logger = logging.getLogger("rivinity_nexus.request")

        try:
            response = await call_next(request)
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "request_complete",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": round(latency_ms, 2),
                },
            )

        response.headers["X-Request-ID"] = request_id
        return response
