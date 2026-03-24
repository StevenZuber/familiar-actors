"""Rebuild the consolidated embedding index files for Railway deployment.

Reads all individual .npy embedding files, combines them into a single
numpy array (embeddings_index.npy) and ID mapping (embeddings_ids.json).
Run this after generating new embeddings to prepare for deployment.

Usage:
    uv run python scripts/consolidate_index.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sqlalchemy import or_
from sqlmodel import Session, select

sys.path.insert(0, str(Path(__file__).parent.parent))

from familiar_actors.config import settings
from familiar_actors.database import create_db_and_tables, engine
from familiar_actors.models import Actor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    create_db_and_tables()

    with Session(engine) as session:
        actors = session.exec(
            select(Actor).where(
                or_(
                    Actor.clip_avg_embedding_path.isnot(None),
                    Actor.clip_embedding_path.isnot(None),
                )
            )
        ).all()

        ids = []
        vecs = []
        failed = 0

        for actor in actors:
            try:
                path = actor.clip_avg_embedding_path or actor.clip_embedding_path
                emb = np.load(path)
                ids.append(actor.id)
                vecs.append(emb)
            except Exception:
                failed += 1

        logger.info(f"Loaded {len(ids)} embeddings, {failed} failed")

    if not vecs:
        logger.error("No embeddings found. Run 'familiar-actors embed' first.")
        return

    embeddings = np.array(vecs)
    index_path = settings.data_dir / "embeddings_index.npy"
    ids_path = settings.data_dir / "embeddings_ids.json"

    np.save(index_path, embeddings)
    with open(ids_path, "w") as f:
        json.dump(ids, f)

    logger.info(f"Saved {index_path}: {embeddings.shape}")
    logger.info(f"Saved {ids_path}: {len(ids)} IDs")
    logger.info(
        f"Total size: {index_path.stat().st_size / 1024 / 1024:.1f}MB + "
        f"{ids_path.stat().st_size / 1024:.0f}KB"
    )


if __name__ == "__main__":
    main()
