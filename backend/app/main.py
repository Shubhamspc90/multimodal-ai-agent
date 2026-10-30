from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.routes.audio import router as audio_router
from backend.app.api.routes.conversations import (
    router as conversations_router,
)
from backend.app.api.routes.documents import router as documents_router
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.image import router as image_router
from backend.app.api.routes.text import router as text_router
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown."""

    yield


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(text_router, prefix="/api/v1")
app.include_router(image_router, prefix="/api/v1")
app.include_router(audio_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(conversations_router, prefix="/api/v1")