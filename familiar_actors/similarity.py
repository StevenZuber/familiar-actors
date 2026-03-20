import logging
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import or_
from sqlmodel import Session, select

from familiar_actors.config import settings
from familiar_actors.models import Actor, ActorResult

logger = logging.getLogger(__name__)


@dataclass
class SimilarityIndex:
    """In-memory index of actor embeddings for fast cosine similarity search.

    Loaded once at app startup from .npy files on disk. Prefers averaged
    multi-photo embeddings when available, falls back to single-photo.
    All embeddings are L2-normalized on load so similarity is a simple dot product.
    """

    actor_ids: list[int] = field(default_factory=list)
    embeddings: np.ndarray | None = None

    @property
    def is_loaded(self) -> bool:
        return self.embeddings is not None and len(self.actor_ids) > 0

    def load(self, session: Session) -> None:
        """Load all actor embeddings into memory.

        First tries the consolidated index files (embeddings_index.npy +
        embeddings_ids.json) for fast bulk loading. Falls back to loading
        individual .npy files per actor if the consolidated files don't exist.
        L2-normalizes all embeddings for cosine similarity via dot product.
        """
        index_path = settings.data_dir / "embeddings_index.npy"
        ids_path = settings.data_dir / "embeddings_ids.json"

        # Try individual .npy files first (always correct for the current DB),
        # fall back to consolidated index (used on Railway where individual files
        # aren't deployed)
        self._load_individual(session)
        if not self.is_loaded and index_path.exists() and ids_path.exists():
            self._load_consolidated(index_path, ids_path)

        if self.is_loaded:
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / norms
            logger.info(f"Loaded {len(self.actor_ids)} actor embeddings into index")
        else:
            logger.warning("No valid embeddings loaded")

    def _load_consolidated(self, index_path, ids_path) -> None:
        """Load from pre-built consolidated files (two files instead of 100k+)."""
        import json

        with open(ids_path) as f:
            self.actor_ids = json.load(f)
        self.embeddings = np.load(index_path)
        logger.info(f"Loaded consolidated index: {self.embeddings.shape}")

    def _load_individual(self, session: Session) -> None:
        """Fall back to loading individual .npy files per actor."""
        actors = session.exec(
            select(Actor).where(
                or_(
                    Actor.clip_avg_embedding_path.isnot(None),  # type: ignore[union-attr]
                    Actor.clip_embedding_path.isnot(None),  # type: ignore[union-attr]
                )
            )
        ).all()

        if not actors:
            logger.warning("No actors with CLIP embeddings found")
            return

        ids = []
        vecs = []

        for actor in actors:
            try:
                embedding_path = (
                    actor.clip_avg_embedding_path or actor.clip_embedding_path
                )
                embedding = np.load(embedding_path)
                ids.append(actor.id)
                vecs.append(embedding)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Failed to load embedding for {actor.name}: {e}")

        if vecs:
            self.actor_ids = ids
            self.embeddings = np.array(vecs)

    def search(
        self, actor_id: int, session: Session, top_n: int | None = None
    ) -> list[ActorResult]:
        """Find the most similar actors to the given actor by cosine similarity.

        Returns the top N most similar actors, excluding the queried actor.
        Scores range from -1 to 1, where 1 means identical embeddings.
        """
        if not self.is_loaded:
            return []

        top_n = top_n or settings.similarity_top_n

        if actor_id not in self.actor_ids:
            return []

        idx = self.actor_ids.index(actor_id)
        query_vec = self.embeddings[idx]

        # Cosine similarity (embeddings are pre-normalized)
        similarities = self.embeddings @ query_vec

        # Get top N+1 (excluding self), then trim
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for i in top_indices:
            if self.actor_ids[i] == actor_id:
                continue
            if len(results) >= top_n:
                break

            matched_actor = session.get(Actor, self.actor_ids[i])
            if matched_actor:
                results.append(
                    ActorResult(
                        id=matched_actor.id,
                        tmdb_id=matched_actor.tmdb_id,
                        name=matched_actor.name,
                        tmdb_image_url=matched_actor.tmdb_image_url,
                        similarity_score=round(float(similarities[i]), 4),
                    )
                )

        return results
