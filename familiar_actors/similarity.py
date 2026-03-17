import logging
from dataclasses import dataclass, field

import numpy as np
from sqlmodel import Session, select

from familiar_actors.config import settings
from familiar_actors.models import Actor, ActorResult

logger = logging.getLogger(__name__)


@dataclass
class SimilarityIndex:
    """In-memory index of actor embeddings for fast similarity search."""

    actor_ids: list[int] = field(default_factory=list)
    embeddings: np.ndarray | None = None

    @property
    def is_loaded(self) -> bool:
        return self.embeddings is not None and len(self.actor_ids) > 0

    def load(self, session: Session) -> None:
        """Load all actor embeddings into memory."""
        actors = session.exec(
            select(Actor).where(
                Actor.embedding_path.isnot(None)  # type: ignore[union-attr]
            )
        ).all()

        if not actors:
            logger.warning("No actors with embeddings found")
            return

        ids = []
        vecs = []

        for actor in actors:
            try:
                embedding = np.load(actor.embedding_path)
                ids.append(actor.id)
                vecs.append(embedding)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Failed to load embedding for {actor.name}: {e}")

        if vecs:
            self.actor_ids = ids
            self.embeddings = np.array(vecs)
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / norms
            logger.info(f"Loaded {len(ids)} actor embeddings into index")
        else:
            logger.warning("No valid embeddings loaded")

    def search(
        self, actor_id: int, session: Session, top_n: int | None = None
    ) -> list[ActorResult]:
        """Find the most similar actors to the given actor."""
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
