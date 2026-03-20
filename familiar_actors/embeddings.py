import logging

import numpy as np
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
        try:
            import open_clip
        except ImportError:
            raise RuntimeError(
                "open-clip-torch is not installed. "
                "Install the pipeline dependencies: uv sync --group pipeline"
            )
        import torch  # noqa: F811

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
    """Generate a 512-dimensional CLIP embedding for a single image.

    Loads the image, applies CLIP preprocessing, and runs it through
    the ViT-B-32 image encoder. Returns a numpy array or None on failure.
    The CLIP model is lazy-loaded on first call.
    """
    try:
        import torch
        from PIL import Image

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
    """Generate single-photo CLIP embeddings for all actors with headshots.

    Processes actors that have an image_path but no clip_embedding_path.
    Saves each embedding as a .npy file in data/embeddings_clip/{tmdb_id}.npy.
    Skips actors already processed. Commits per-actor for incremental progress.
    """
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


def process_multi_photo_embeddings(session: Session) -> int:
    """Generate averaged CLIP embeddings from multiple photos per actor.

    For each actor with photos in data/headshots_multi/{tmdb_id}/, generates
    a CLIP embedding for each photo, L2-normalizes them so each contributes
    equally, averages, and re-normalizes. The result is a single embedding
    that represents the actor's general appearance more stably than any
    single photo. Saves to data/embeddings_avg/{tmdb_id}.npy.
    Skips actors already processed. Safe to interrupt and resume.
    """
    actors = session.exec(
        select(Actor).where(
            Actor.clip_avg_embedding_path.is_(None),  # type: ignore[union-attr]
        )
    ).all()

    if not actors:
        logger.info("No actors need multi-photo embedding generation")
        return 0

    settings.embeddings_avg_dir.mkdir(parents=True, exist_ok=True)
    processed = 0

    for actor in actors:
        actor_photo_dir = settings.headshots_multi_dir / str(actor.tmdb_id)
        if not actor_photo_dir.exists():
            continue

        photos = sorted(actor_photo_dir.glob("*.jpg"))
        if not photos:
            continue

        # Generate embeddings for each photo
        embeddings = []
        for photo in photos:
            emb = generate_embedding(str(photo))
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            continue

        # L2-normalize each, average, re-normalize
        normalized = [emb / np.linalg.norm(emb) for emb in embeddings]
        avg = np.mean(normalized, axis=0)
        avg = avg / np.linalg.norm(avg)

        avg_path = settings.embeddings_avg_dir / f"{actor.tmdb_id}.npy"
        np.save(avg_path, avg)

        actor.clip_avg_embedding_path = str(avg_path)
        session.add(actor)
        session.commit()
        processed += 1

        if processed % 25 == 0:
            logger.info(f"Generated {processed} averaged embeddings")

    logger.info(f"Generated {processed} averaged embeddings total")
    return processed
