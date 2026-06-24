"""Rebuild the consolidated embedding index files for Railway deployment.

Reads all individual .npy embedding files for an embedding space, combines
them into a single numpy array + ID mapping. Run after generating embeddings
to prepare for deployment.

The space defaults to settings.embedding_space; pass --space to override.
Output filenames follow settings.consolidated_index_paths():
  clip     -> embeddings_index.npy / embeddings_ids.json (legacy names)
  facecrop -> facecrop_index.npy   / facecrop_ids.json

Usage:
    uv run python scripts/consolidate_index.py [--space clip|facecrop]
"""

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--space",
        default=settings.embedding_space,
        choices=["clip", "facecrop"],
        help="Embedding space to consolidate (default: settings.embedding_space)",
    )
    args = parser.parse_args()
    space = args.space
    logger.info(f"Consolidating '{space}' embedding space")

    create_db_and_tables()

    with Session(engine) as session:
        if space == "facecrop":
            condition = Actor.facecrop_embedding_path.isnot(None)  # type: ignore[union-attr]
        else:
            condition = or_(
                Actor.clip_avg_embedding_path.isnot(None),  # type: ignore[union-attr]
                Actor.clip_embedding_path.isnot(None),  # type: ignore[union-attr]
            )
        actors = session.exec(select(Actor).where(condition)).all()

        ids = []
        vecs = []
        failed = 0

        for actor in actors:
            try:
                if space == "facecrop":
                    path = actor.facecrop_embedding_path
                else:
                    path = actor.clip_avg_embedding_path or actor.clip_embedding_path
                if not path:
                    continue
                emb = np.load(path)
                ids.append(actor.id)
                vecs.append(emb)
            except Exception:
                failed += 1

        logger.info(f"Loaded {len(ids)} embeddings, {failed} failed")

    if not vecs:
        logger.error(
            f"No '{space}' embeddings found. "
            f"Run 'familiar-actors embed{'-facecrop' if space == 'facecrop' else ''}' first."
        )
        return

    embeddings = np.array(vecs)
    index_path, ids_path = settings.consolidated_index_paths(space)

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
