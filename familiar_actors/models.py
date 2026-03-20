from sqlmodel import Field, SQLModel


class Actor(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    tmdb_id: int = Field(unique=True, index=True)
    name: str = Field(index=True)
    image_path: str | None = None
    embedding_path: str | None = None
    clip_embedding_path: str | None = None
    clip_avg_embedding_path: str | None = None
    tmdb_image_url: str | None = None


class ActorResult(SQLModel):
    """Response model for similarity search results."""

    id: int
    tmdb_id: int
    name: str
    tmdb_image_url: str | None
    similarity_score: float
