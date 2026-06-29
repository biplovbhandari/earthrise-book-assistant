import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from earthrise_rag import __version__
from earthrise_rag.config import get_settings

from api.dependencies import create_pipelines
from api.routes.ask import router as ask_router
from api.routes.search import router as search_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = get_settings()
    try:
        app.state.pipelines = create_pipelines(app.state.settings)
    except Exception:
        logger.exception("Failed to build retrieval pipelines; /search will return 503")
        app.state.pipelines = None
    yield


app = FastAPI(
    title="EarthRISE Book Assistant",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    pipelines = getattr(app.state, "pipelines", None)
    generation_ready = (
        pipelines is not None
        and pipelines.query is not None
        and pipelines.query._llm_client is not None
    )
    return {
        "status": "ok",
        "version": __version__,
        "generation": "ready" if generation_ready else "unavailable",
    }


app.include_router(ask_router)
app.include_router(search_router)

# --- Static book HTML below (catch-all, must be last) ---
_book_html_dir = Path(get_settings().book_html_dir)
_book_html_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=_book_html_dir, html=True), name="book")
