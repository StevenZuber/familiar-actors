from sqlmodel import Field, SQLModel


class Actor(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    tmdb_id: int = Field(unique=True, index=True)
    name: str = Field(index=True)
    image_path: str | None = None
    clip_embedding_path: str | None = None
    clip_avg_embedding_path: str | None = None
    # CLIP embedding computed on the detected+cropped face (averaged over the
    # actor's photos when multiple exist, else the single headshot). A separate
    # embedding space from clip_*; which one is live is set by
    # settings.embedding_space. See PLAN_face_embeddings.md.
    facecrop_embedding_path: str | None = None
    tmdb_image_url: str | None = None
    # Set when TMDB returned no profile images meeting min_image_width — keeps
    # us from re-querying /person/{id}/images for the same dead-end every run.
    multi_photo_unavailable: bool = Field(default=False)
    # Set when no face could be detected in any of the actor's photos, so the
    # facecrop pipeline skips them on resume instead of re-detecting.
    face_unavailable: bool = Field(default=False)
    # Estimated by the buffalo_l genderage model during the facecrop pass
    # (majority gender, mean age across photos). Used only as optional search
    # filters — they're estimates, not ground truth.
    gender: str | None = None  # "M" | "F"
    age: int | None = None


class ActorResult(SQLModel):
    """Response model for similarity search results."""

    id: int
    tmdb_id: int
    name: str
    tmdb_image_url: str | None
    similarity_score: float
