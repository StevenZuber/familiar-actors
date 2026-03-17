import logging

import numpy as np
import open_clip
import torch
from PIL import Image
from sqlmodel import Session, select

from familiar_actors.config import settings
from familiar_actors.models import Actor

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_model = None
_preprocess = None


def _get_model():
    global _model, _preprocess
    if _model is None:
        logger.info(
            f"Loading CLIP model {settings.embedding_model} "
            f"({settings.clip_pretrained})..."
        )
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            settings.embedding_model,
            pretrained=settings.clip_pretrained,
        )
        _model.eval()
        logger.info("CLIP model loaded")
    return _model, _preprocess


def generate_embedding(image_path: str) -> np.ndarray | None:
    """Generate a CLIP embedding for a single image."""
    try:
        model, preprocess = _get_model()
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            embedding = model.encode_image(image_tensor)

        return embedding.squeeze().numpy()
    except Exception as e:
        logger.warning(f"Failed to generate embedding for {image_path}: {e}")
    return None


def process_all_embeddings(session: Session) -> int:
    """Generate CLIP embeddings for all actors that have headshots but no CLIP embedding."""
    actors = session.exec(
        select(Actor).where(
            Actor.image_path.isnot(None),  # type: ignore[union-attr]
            Actor.clip_embedding_path.is_(None),  # type: ignore[union-attr]
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

        actor.clip_embedding_path = str(embedding_path)
        session.add(actor)
        session.commit()
        processed += 1

        if processed % 25 == 0:
            logger.info(f"Generated {processed}/{len(actors)} embeddings")

    logger.info(f"Generated {processed} embeddings total")
    return processed
