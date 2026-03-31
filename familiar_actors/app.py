import logging
import tarfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session

from familiar_actors.actor_search import ActorSearchIndex
from familiar_actors.config import settings
from familiar_actors.database import create_db_and_tables, engine
from familiar_actors.routes.search import router as search_router
from familiar_actors.similarity import SimilarityIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

index = SimilarityIndex()
search_index = ActorSearchIndex()


def _is_data_stale() -> bool:
    """Check if the remote dataset has changed by comparing Content-Length.

    Makes a HEAD request to the release URL and compares the remote file size
    against the size we recorded when we last downloaded. Returns True if
    the data should be re-downloaded.
    """
    size_file = settings.data_dir / ".data_size"
    if not size_file.exists():
        return True

    try:
        response = httpx.head(
            settings.data_release_url, follow_redirects=True, timeout=30.0
        )
        response.raise_for_status()
        remote_size = response.headers.get("content-length", "")
        local_size = size_file.read_text().strip()
        if remote_size != local_size:
            logger.info(
                f"Remote data size changed ({local_size} -> {remote_size}), "
                "re-downloading..."
            )
            return True
    except Exception as e:
        logger.warning(f"Could not check remote data size: {e}")

    return False


def _download_data_if_needed():
    """Download and extract the consolidated dataset from a GitHub Release.

    Downloads when DATA_RELEASE_URL is set and either the consolidated index
    files don't exist yet or the remote file size has changed (indicating
    an updated dataset). The tarball contains the SQLite DB,
    embeddings_index.npy, and embeddings_ids.json.
    """
    if not settings.data_release_url:
        return

    index_path = settings.data_dir / "embeddings_index.npy"
    if index_path.exists() and not _is_data_stale():
        return

    logger.info("Downloading dataset from GitHub Release...")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = settings.data_dir / "data.tar.gz"

    try:
        with httpx.stream(
            "GET", settings.data_release_url, follow_redirects=True, timeout=600.0
        ) as response:
            response.raise_for_status()
            remote_size = response.headers.get("content-length", "")
            with open(tarball_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)

        logger.info("Download complete. Extracting...")
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=settings.data_dir, filter="data")
        tarball_path.unlink()

        # Record the size so we can detect changes on next startup
        size_file = settings.data_dir / ".data_size"
        size_file.write_text(remote_size)

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
        search_index.load(session)
    yield


app = FastAPI(title="Familiar Actors", lifespan=lifespan)

templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

app.include_router(search_router)
