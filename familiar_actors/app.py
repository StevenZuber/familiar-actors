import logging
import tarfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session

from familiar_actors.config import settings
from familiar_actors.database import create_db_and_tables, engine
from familiar_actors.routes.search import router as search_router
from familiar_actors.similarity import SimilarityIndex

logger = logging.getLogger(__name__)

index = SimilarityIndex()


def _download_data_if_needed():
    """Download and extract the dataset from a GitHub Release if the data directory is empty.

    Only runs when DATA_RELEASE_URL is set (i.e., on Railway) and the database
    doesn't exist yet. Downloads a tarball via httpx, extracts safe members
    into the data directory, then cleans up the tarball.
    """
    logger.info(
        f"Checking data: release_url={bool(settings.data_release_url)}, "
        f"data_dir={settings.data_dir}, exists={settings.data_dir.exists()}"
    )

    if not settings.data_release_url:
        logger.info("No DATA_RELEASE_URL set, skipping download")
        return

    # Check if we have a substantial number of embeddings (not a partial failed extraction)
    from itertools import islice

    embedding_count = 0
    for emb_dir in [settings.embeddings_dir, settings.embeddings_avg_dir]:
        if emb_dir.exists():
            embedding_count += len(list(islice(emb_dir.glob("*.npy"), 1000)))
    logger.info(f"Found {embedding_count} embedding files on disk")
    if embedding_count >= 1000:
        logger.info("Sufficient embeddings found, skipping download")
        return

    logger.info("Data directory is empty — downloading dataset from GitHub Release...")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = settings.data_dir / "data.tar.gz"

    try:
        with httpx.stream(
            "GET", settings.data_release_url, follow_redirects=True, timeout=600.0
        ) as response:
            response.raise_for_status()
            with open(tarball_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)

        logger.info("Download complete. Extracting...")
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=settings.data_dir, filter="data")
        tarball_path.unlink()
        logger.info("Dataset extracted successfully")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        if tarball_path.exists():
            tarball_path.unlink()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _download_data_if_needed()
    create_db_and_tables()
    with Session(engine) as session:
        index.load(session)
    yield


app = FastAPI(title="Familiar Actors", lifespan=lifespan)

templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

app.include_router(search_router)
