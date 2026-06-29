from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedding_provider: str = "local"
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    hf_home: str = "/models"

    vector_store_provider: str = "qdrant"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "earthrise_book"

    retrieval_strategy: str = "dense"
    retrieval_top_k: int = Field(default=8, ge=1, le=50)
    reranker_provider: str = "noop"

    llm_provider: str = "openai_compatible"
    llm_base_url: str = "https://proxy.fast.luna.nasa.gov/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0)

    book_html_dir: str = "_book"
    book_source_dir: str = "book"
    book_commit_sha: str = ""

    app_env: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
