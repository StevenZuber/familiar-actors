"""Face-crop CLIP embeddings (pipeline side).

Detects the face in each of an actor's photos (InsightFace buffalo_l / SCRFD),
crops to it, and runs the same CLIP encoder used elsewhere on the crop. This
strips background / clothing / pose that contaminate whole-image CLIP, while
keeping CLIP's strong looks-alike signal (validated in PLAN_face_embeddings.md).

One embedding per actor is stored in `facecrop_embedding_path`: averaged over
the w500 multi-photos when present, else computed from the single w185
headshot. Actors with no detectable face in any photo get `face_unavailable`.

Requires the `face` and `pipeline` dependency groups (insightface + open-clip).
The serving/upload path does NOT use this module — it embeds with onnxruntime.
"""

import logging
from typing import Any

import numpy as np
from sqlmodel import Session, select

from familiar_actors.config import settings
from familiar_actors.face_crop import crop_to_face, largest_face
from familiar_actors.models import Actor

logger = logging.getLogger(__name__)

# Lazy-loaded singletons. Typed Any because neither open-clip nor insightface
# ship type stubs visible to the checker.
_clip_model: Any = None
_clip_preprocess: Any = None
_detector: Any = None


def _get_clip() -> tuple[Any, Any]:
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        try:
            import open_clip  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "open-clip-torch is not installed. Install: uv sync --group pipeline"
            )
        logger.info(
            f"Loading CLIP {settings.embedding_model} ({settings.clip_pretrained})..."
        )
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            settings.embedding_model, pretrained=settings.clip_pretrained
        )
        _clip_model.eval()
        logger.info("CLIP model loaded")
    return _clip_model, _clip_preprocess


def _get_detector() -> Any:
    global _detector
    if _detector is None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "insightface is not installed. Install: uv sync --group face"
            )
        logger.info("Loading face detector (buffalo_l / SCRFD)...")
        _detector = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        _detector.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("Face detector loaded")
    return _detector


def embed_facecrop(image_path: str) -> np.ndarray | None:
    """CLIP-embed the largest detected face in one image. None if no face."""
    try:
        import cv2  # type: ignore[import-untyped]
        import torch
        from PIL import Image

        detector = _get_detector()
        model, preprocess = _get_clip()

        img = cv2.imread(image_path)
        if img is None:
            return None
        face = largest_face(detector.get(img))
        if face is None:
            return None
        crop = crop_to_face(img, face.bbox)
        if crop.size == 0:
            return None

        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        with torch.no_grad():  # type: ignore[no-untyped-call]
            emb = model.encode_image(preprocess(pil).unsqueeze(0))
        return emb.squeeze().numpy()
    except Exception as e:
        logger.warning(f"Failed to face-embed {image_path}: {e}")
        return None


def _photos_for(actor: Actor) -> list[str]:
    """Prefer the w500 multi-photos (higher-res faces); else the headshot."""
    multi_dir = settings.headshots_multi_dir / str(actor.tmdb_id)
    if multi_dir.exists():
        photos = sorted(multi_dir.glob("*.jpg"))[: settings.max_photos_per_actor]
        if photos:
            return [str(p) for p in photos]
    if actor.image_path:
        return [actor.image_path]
    return []


def generate_facecrop_embedding(actor: Actor, session: Session) -> bool:
    """Compute and store one face-crop CLIP embedding for an actor.

    Averages the per-photo face-crop embeddings (L2-normalized so each photo
    contributes equally), renormalizes, and saves to
    data/embeddings_facecrop/{tmdb_id}.npy. Sets facecrop_embedding_path on
    success, or face_unavailable when no photo yielded a detectable face.
    Commits either way so progress is durable. Returns True iff an embedding
    was written.
    """
    photos = _photos_for(actor)
    embeddings = []
    for path in photos:
        emb = embed_facecrop(path)
        if emb is not None:
            embeddings.append(emb / np.linalg.norm(emb))

    if not embeddings:
        actor.face_unavailable = True
        session.add(actor)
        session.commit()
        return False

    avg = np.mean(embeddings, axis=0)
    avg = avg / np.linalg.norm(avg)

    out_path = settings.facecrop_embeddings_dir / f"{actor.tmdb_id}.npy"
    np.save(out_path, avg)

    actor.facecrop_embedding_path = str(out_path)
    session.add(actor)
    session.commit()
    return True


def process_facecrop_embeddings(session: Session, limit: int | None = None) -> int:
    """Generate face-crop embeddings for every actor that needs one.

    Targets actors with a headshot but no facecrop_embedding_path yet, skipping
    those already flagged face_unavailable. Safe to interrupt and resume —
    progress is committed per actor. `limit` caps the batch (for validation).
    """
    query = select(Actor).where(
        Actor.image_path.isnot(None),  # type: ignore[union-attr]
        Actor.facecrop_embedding_path.is_(None),  # type: ignore[union-attr]
        Actor.face_unavailable.is_(False),  # type: ignore[union-attr]
    )
    if limit is not None:
        query = query.limit(limit)
    actors = session.exec(query).all()

    if not actors:
        logger.info("No actors need face-crop embeddings")
        return 0

    settings.facecrop_embeddings_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    no_face = 0

    for i, actor in enumerate(actors):
        if generate_facecrop_embedding(actor, session):
            processed += 1
        else:
            no_face += 1
        if (i + 1) % 50 == 0:
            logger.info(
                f"Face-crop embeddings: {i + 1}/{len(actors)} "
                f"({processed} done, {no_face} no-face)"
            )

    logger.info(f"Generated {processed} face-crop embeddings ({no_face} no-face)")
    return processed
