"""FastAPI application entry point for Video-to-PGN."""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes.api import router as api_router
from app.routes.pages import router as pages_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Video-to-PGN",
    description="Convert YouTube chess videos into PGN files",
    version="1.0.0",
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
(static_dir / "css").mkdir(exist_ok=True)
(static_dir / "js").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routers
app.include_router(api_router)
app.include_router(pages_router)


@app.on_event("startup")
async def startup():
    """Run on application startup."""
    settings.ensure_dirs()
    logging.getLogger(__name__).info("Video-to-PGN application started")
    if not settings.gemini_api_key:
        logging.getLogger(__name__).warning(
            "GEMINI_API_KEY not set! Set VTP_GEMINI_API_KEY environment variable."
        )
