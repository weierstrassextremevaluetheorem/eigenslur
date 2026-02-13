from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EIGENSLUR_", env_file=".env", extra="ignore"
    )

    app_name: str = "EigenSlur Backend"
    app_version: str = "0.1.0"
    default_locale: str = "en-US"
    fusion_threshold_review: float = 0.35
    fusion_threshold_block: float = 0.65
    embedding_dim: int = 256
    enable_persistence: bool = True
    database_path: str = "data/eigenslur.db"
    use_llm_labeler: bool = True
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_timeout_seconds: float = 20.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
