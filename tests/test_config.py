from earthrise_rag.config import Settings


def test_settings_respects_env_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "remote")
    monkeypatch.setenv("APP_ENV", "production")
    settings = Settings()
    assert settings.embedding_provider == "remote"
    assert settings.app_env == "production"
