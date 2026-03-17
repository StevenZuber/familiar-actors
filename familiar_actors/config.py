from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    tmdb_read_access_token: str = ""

    data_dir: Path = Path("data")
    headshots_dir: Path = Path("data/headshots")
    embeddings_dir: Path = Path("data/embeddings_clip")
    db_path: Path = Path("data/familiar_actors.db")

    embedding_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    similarity_top_n: int = 10

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.db_path}"


settings = Settings()
