import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from earthrise_rag import __version__
from earthrise_rag.config import get_settings

from api.dependencies import create_pipelines
from api.routes.ask import router as ask_router
from api.routes.chat import check_chat_ready, check_generation_ready, check_retrieval_ready
from api.routes.chat import router as chat_router
from api.routes.search import router as search_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = get_settings()
    try:
        app.state.pipelines = create_pipelines(app.state.settings)
    except Exception:
        logger.exception("Failed to build pipelines; /search, /ask, and /chat will return 503")
        app.state.pipelines = None
    yield


app = FastAPI(
    title="EarthRISE Book Assistant",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Return application readiness for retrieval, generation, and streaming chat."""
    pipelines = getattr(app.state, "pipelines", None)
    ret_ready, _ = check_retrieval_ready(pipelines)
    gen_ready, _ = check_generation_ready(pipelines)
    stream_ok = gen_ready and callable(
        getattr(pipelines.query._llm_client, "chat_stream", None)
    )
    chat_ok = ret_ready and gen_ready and stream_ok
    return {
        "status": "ok",
        "version": __version__,
        "retrieval": "ready" if ret_ready else "unavailable",
        "generation": "ready" if gen_ready else "unavailable",
        "chat": "ready" if chat_ok else "unavailable",
    }


app.include_router(ask_router)
app.include_router(chat_router)
app.include_router(search_router)

# --- Static book HTML below (catch-all, must be last) ---
_book_html_dir = Path(get_settings().book_html_dir)
_book_html_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=_book_html_dir, html=True), name="book")
