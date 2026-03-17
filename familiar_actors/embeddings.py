import logging

import numpy as np
from deepface import DeepFace
from sqlmodel import Session, select

from familiar_actors.config import settings
from familiar_actors.models import Actor

logger = logging.getLogger(__name__)


def generate_embedding(image_path: str) -> np.ndarray | None:
    """Generate a face embedding for a single image."""
    try:
        embeddings = DeepFace.represent(
            img_path=image_path,
            model_name=settings.embedding_model,
            enforce_detection=True,
        )
        if embeddings:
            return np.array(embeddings[0]["embedding"])
    except (ValueError, Exception) as e:
        logger.warning(f"Failed to generate embedding for {image_path}: {e}")
    return None


def process_all_embeddings(session: Session) -> int:
    """Generate embeddings for all actors that have headshots but no embedding."""
    actors = session.exec(
        select(Actor).where(
            Actor.image_path.isnot(None),  # type: ignore[union-attr]
            Actor.embedding_path.is_(None),  # type: ignore[union-attr]
        )
    ).all()

    if not actors:
        logger.info("No actors need embedding generation")
        return 0

    settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
    processed = 0

    for actor in actors:
        embedding = generate_embedding(actor.image_path)
        if embedding is None:
            continue

        embedding_path = settings.embeddings_dir / f"{actor.tmdb_id}.npy"
        np.save(embedding_path, embedding)

        actor.embedding_path = str(embedding_path)
        session.add(actor)
        session.commit()
        processed += 1

        if processed % 25 == 0:
            logger.info(f"Generated {processed}/{len(actors)} embeddings")

    logger.info(f"Generated {processed} embeddings total")
    return processed
