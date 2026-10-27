from functools import lru_cache
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=os.getenv("PBPK_ENV_FILE", ".env"),
        env_file_encoding="utf-8", extra="ignore",
    )

    service_name: str = Field(default="pbpk-stage1-public-kb", alias="PBPK_SERVICE_NAME")
    service_version: str = Field(default="0.1.0", alias="PBPK_SERVICE_VERSION")
    database_url: str = Field(
        default="postgresql+psycopg://pbpk:pbpk_password@localhost:5432/pbpk_kb",
        alias="DATABASE_URL",
    )
    source_registry_path: str = Field(default="pbpk_kb/config/sources.yaml", alias="PBPK_SOURCE_REGISTRY_PATH")
    llm_api_key: str | None = Field(default=None, alias="PBPK_LLM_API_KEY")
    llm_model: str = Field(default="Qwen3-8B-BF16", alias="PBPK_LLM_MODEL")
    llm_base_url: str = Field(default="local://qwen3-8b", alias="PBPK_LLM_BASE_URL")
    embedding_api_url: str | None = Field(default=None, alias="PBPK_EMBEDDING_API_URL")
    embedding_model: str = Field(default="NeuML/pubmedbert-base-embeddings", alias="PBPK_EMBEDDING_MODEL")
    embedding_timeout_seconds: float = Field(default=30.0, alias="PBPK_EMBEDDING_TIMEOUT_SECONDS")


@lru_cache
def get_settings() -> Settings:
    return Settings()
