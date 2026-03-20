from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    tmdb_read_access_token: str = ""

    data_dir: Path = Path("data")

    embedding_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    similarity_top_n: int = 10
    multi_image_size: str = "w500"
    min_image_width: int = 500
    max_photos_per_actor: int = 5

    @property
    def headshots_dir(self) -> Path:
        return self.data_dir / "headshots"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings_clip"

    @property
    def headshots_multi_dir(self) -> Path:
        return self.data_dir / "headshots_multi"

    @property
    def embeddings_avg_dir(self) -> Path:
        return self.data_dir / "embeddings_avg"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "familiar_actors.db"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.db_path}"


settings = Settings()
