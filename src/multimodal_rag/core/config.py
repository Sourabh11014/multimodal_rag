from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Multimodal RAG API"
    app_version: str = "0.1.0"
    data_dir: Path = Field(default=Path("data"))
    raw_dir: Path = Field(default=Path("data/raw"))
    index_dir: Path = Field(default=Path("data/index"))
    chunk_size: int = 500
    chunk_overlap: int = 75
    top_k: int = 4

    model_config = SettingsConfigDict(env_prefix="MMRAG_", env_file=".env", extra="ignore")


settings = Settings()
settings.raw_dir.mkdir(parents=True, exist_ok=True)
settings.index_dir.mkdir(parents=True, exist_ok=True)
