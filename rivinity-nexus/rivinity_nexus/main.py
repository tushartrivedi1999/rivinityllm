import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rivinity_nexus.api.router import api_router
from rivinity_nexus.config.settings import get_settings
from rivinity_nexus.core.database import init_db
from rivinity_nexus.core.logging import RequestLoggingMiddleware, configure_logging
from rivinity_nexus.monitoring.metrics import metrics_router

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("rivinity_nexus")

app = FastAPI(title=settings.app_name)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix=settings.api_v1_prefix)
app.include_router(metrics_router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    logger.info("startup_complete")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception", exc_info=exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}
