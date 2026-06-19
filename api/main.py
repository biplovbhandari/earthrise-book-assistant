from contextlib import asynccontextmanager

from fastapi import FastAPI

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
