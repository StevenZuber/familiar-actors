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
from collections import Counter
from typing import Any

import numpy as np
from sqlalchemy import func, or_
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
        logger.info("Loading face detector + genderage (buffalo_l)...")
        _detector = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "genderage"],
            providers=["CPUExecutionProvider"],
        )
        _detector.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("Face detector loaded")
    return _detector


def _process_photo(
    image_path: str, need_embedding: bool
) -> tuple[np.ndarray | None, str | None, int | None]:
    """Detect the largest face in one image and return (embedding, sex, age).

    `sex` is "M"/"F" and `age` an int from the genderage model. `embedding` is
    the CLIP face-crop vector, computed only when `need_embedding` (so the
    gender/age backfill of already-embedded actors skips the CLIP cost). Any
    element is None when unavailable (no face, or embedding not requested).
    """
    try:
        import cv2  # type: ignore[import-untyped]
        import torch
        from PIL import Image

        detector = _get_detector()

        img = cv2.imread(image_path)
        if img is None:
            return None, None, None
        face = largest_face(detector.get(img))
        if face is None:
            return None, None, None

        sex = "M" if int(face.gender) == 1 else "F"
        age = int(face.age)

        embedding = None
        if need_embedding:
            crop = crop_to_face(img, face.bbox)
            if crop.size > 0:
                model, preprocess = _get_clip()
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                with torch.no_grad():  # type: ignore[no-untyped-call]
                    emb = model.encode_image(preprocess(pil).unsqueeze(0))
                embedding = emb.squeeze().numpy()
        return embedding, sex, age
    except Exception as e:
        logger.warning(f"Failed to process {image_path}: {e}")
        return None, None, None


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
    """Process one actor's photos: face-crop embedding (if missing) + gender/age.

    Detects the largest face in each photo. The face-crop embedding is the
    L2-normalized average of the per-photo CLIP crops (computed only when the
    actor has no facecrop_embedding_path yet — so re-running just backfills
    gender/age on already-embedded actors). Gender is the majority vote and age
    the mean across photos. Sets face_unavailable when no photo yielded a face.
    Commits either way so progress is durable. Returns False only on no-face.
    """
    need_embedding = actor.facecrop_embedding_path is None

    embeddings: list[np.ndarray] = []
    sexes: list[str] = []
    ages: list[int] = []
    for path in _photos_for(actor):
        emb, sex, age = _process_photo(path, need_embedding)
        if sex is not None:
            sexes.append(sex)
            if age is not None:
                ages.append(age)
        if emb is not None:
            embeddings.append(emb / np.linalg.norm(emb))

    if not sexes:
        # No face detected in any photo — flag and skip on future runs.
        actor.face_unavailable = True
        session.add(actor)
        session.commit()
        return False

    actor.gender = Counter(sexes).most_common(1)[0][0]
    if ages:
        actor.age = int(round(sum(ages) / len(ages)))

    if need_embedding and embeddings:
        avg = np.mean(embeddings, axis=0)
        avg = avg / np.linalg.norm(avg)
        out_path = settings.facecrop_embeddings_dir / f"{actor.tmdb_id}.npy"
        np.save(out_path, avg)
        actor.facecrop_embedding_path = str(out_path)

    session.add(actor)
    session.commit()
    return True


def _needs_processing():
    """WHERE clause for actors needing a facecrop embedding and/or gender/age."""
    return (
        Actor.image_path.isnot(None),  # type: ignore[union-attr]
        Actor.face_unavailable.is_(False),  # type: ignore[union-attr]
        or_(
            Actor.facecrop_embedding_path.is_(None),  # type: ignore[union-attr]
            Actor.gender.is_(None),  # type: ignore[union-attr]
        ),
    )


def process_facecrop_embeddings(
    session: Session, limit: int | None = None, batch_size: int = 200
) -> int:
    """Generate face-crop embeddings + gender/age for every actor that needs it.

    Targets actors with a headshot that either lack a facecrop_embedding_path
    or lack a gender estimate (so actors embedded before genderage was added
    get backfilled without recomputing their embedding). Skips actors already
    flagged face_unavailable. Safe to interrupt and resume — committed per
    actor.

    Processes in bounded batches (re-querying "still needs processing" each
    round, since a processed actor no longer matches) so memory stays flat over
    a 406k-row run rather than loading every actor up front. `limit` caps total
    actors processed (for validation).
    """
    settings.facecrop_embeddings_dir.mkdir(parents=True, exist_ok=True)

    total = session.exec(
        select(func.count()).select_from(Actor).where(*_needs_processing())
    ).one()
    if not total:
        logger.info("No actors need face-crop processing")
        return 0

    processed = 0
    no_face = 0
    seen = 0
    while limit is None or seen < limit:
        take = batch_size if limit is None else min(batch_size, limit - seen)
        batch = session.exec(
            select(Actor).where(*_needs_processing()).limit(take)
        ).all()
        if not batch:
            break

        for actor in batch:
            if generate_facecrop_embedding(actor, session):
                processed += 1
            else:
                no_face += 1
            seen += 1
            if seen % 50 == 0:
                logger.info(
                    f"Face-crop processing: {seen}/{total} "
                    f"({processed} done, {no_face} no-face)"
                )

        # Detach processed actors so the identity map doesn't grow all run.
        session.expunge_all()

    logger.info(f"Processed {processed} actors ({no_face} no-face)")
    return processed
