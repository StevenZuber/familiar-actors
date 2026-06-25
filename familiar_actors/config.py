from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    tmdb_read_access_token: str = ""
    data_release_url: str = ""

    data_dir: Path = Path("data")

    embedding_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    similarity_top_n: int = 10
    # Results at or above this cosine to one already shown are treated as the
    # same person (a duplicate TMDB entry / near-identical photo) and skipped.
    # Well above real lookalike scores (doppelgangers top out ~0.94).
    dedup_similarity_threshold: float = 0.98
    multi_image_size: str = "w500"
    min_image_width: int = 500
    max_photos_per_actor: int = 5

    # Which embedding space the search index and upload path use:
    # "clip" (CLIP on the whole image) or "facecrop" (CLIP on the detected
    # face crop). Both are stored per actor; this selects the live one.
    embedding_space: str = "clip"
    # Fraction the detected face box is expanded on each side before cropping,
    # so hair/jaw/head-shape stay in frame (matters for perceived likeness).
    face_crop_pad: float = 0.35

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
    def facecrop_embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings_facecrop"

    def consolidated_index_paths(self, space: str | None = None) -> tuple[Path, Path]:
        """(index.npy, ids.json) paths for an embedding space's consolidated index.

        The original CLIP space keeps its legacy filenames (embeddings_index.npy
        / embeddings_ids.json) so existing deployments aren't disturbed; other
        spaces get {space}_index.npy / {space}_ids.json.
        """
        space = space or self.embedding_space
        stem = "embeddings" if space == "clip" else space
        return (
            self.data_dir / f"{stem}_index.npy",
            self.data_dir / f"{stem}_ids.json",
        )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "familiar_actors.db"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.db_path}"


settings = Settings()
