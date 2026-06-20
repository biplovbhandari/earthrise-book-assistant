import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from earthrise_rag import __version__
from earthrise_rag.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = get_settings()
    yield


app = FastAPI(
    title="EarthRISE Book Assistant",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


# --- API routes above this line ---
# --- Static book HTML below (catch-all, must be last) ---
_book_html_dir = Path(os.environ.get("BOOK_HTML_DIR", "_book"))
_book_html_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=_book_html_dir, html=True), name="book")
