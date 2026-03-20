import asyncio
import logging
import sys

from sqlmodel import Session

from familiar_actors.config import settings
from familiar_actors.database import create_db_and_tables, engine
from familiar_actors.embeddings import (
    process_all_embeddings,
    process_multi_photo_embeddings,
)
from familiar_actors.tmdb import TMDBClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch(num_pages: int = 25):
    """Fetch popular actors from TMDB and download their headshots.

    Paginates through TMDB's /person/popular endpoint, inserts new actors
    into the database, and downloads w185 thumbnails. Skips actors already
    in the DB. Default 25 pages (~500 actors).
    """
    create_db_and_tables()
    settings.headshots_dir.mkdir(parents=True, exist_ok=True)

    client = TMDBClient()

    with Session(engine) as session:
        logger.info(f"Fetching {num_pages} pages of popular actors from TMDB...")
        actors = asyncio.run(client.fetch_popular_actors(session, num_pages))
        logger.info(f"Fetched {len(actors)} actors")

        logger.info("Downloading headshots...")
        downloaded = asyncio.run(client.download_headshots(session))
        logger.info(f"Downloaded {downloaded} new headshots")


def embed():
    """Generate single-photo CLIP embeddings for all downloaded headshots.

    Processes actors with headshots that don't yet have a clip_embedding_path.
    Each embedding is a 512-dimensional vector saved as a .npy file.
    """
    create_db_and_tables()
    settings.embeddings_dir.mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        logger.info("Generating face embeddings...")
        processed = process_all_embeddings(session)
        logger.info(f"Processed {processed} embeddings")


def fetch_credits(num_pages: int = 25, source: str = "movie"):
    """Fetch actors from movie/TV credits and download their headshots.

    Crawls cast lists from TMDB's top-rated movies (or TV shows with source="tv").
    This is the best way to discover obscure character actors beyond the popular
    endpoint. Default 25 pages (~500 titles, potentially 10k-25k new actors).
    """
    create_db_and_tables()
    settings.headshots_dir.mkdir(parents=True, exist_ok=True)

    client = TMDBClient()
    source_label = "TV shows" if source == "tv" else "movies"

    with Session(engine) as session:
        logger.info(
            f"Crawling credits from {num_pages} pages of top-rated {source_label}..."
        )
        new_actors = asyncio.run(
            client.fetch_actors_from_credits(session, num_pages, source)
        )
        logger.info(f"Found {len(new_actors)} new actors")

        logger.info("Downloading headshots...")
        downloaded = asyncio.run(client.download_headshots(session))
        logger.info(f"Downloaded {downloaded} new headshots")


def fetch_images():
    """Download up to 5 extra photos per actor and generate averaged CLIP embeddings.

    Fetches the top-rated profile images from TMDB for each actor, generates
    CLIP embeddings for each, and averages them into a single stronger embedding.
    Incremental — skips actors already processed. Safe to interrupt and resume.
    """
    create_db_and_tables()
    settings.headshots_multi_dir.mkdir(parents=True, exist_ok=True)
    settings.embeddings_avg_dir.mkdir(parents=True, exist_ok=True)

    client = TMDBClient()

    with Session(engine) as session:
        logger.info("Downloading multi-photos from TMDB...")
        downloaded = asyncio.run(client.download_multi_headshots(session))
        logger.info(f"Downloaded multi-photos for {downloaded} actors")

        logger.info("Generating averaged embeddings...")
        processed = process_multi_photo_embeddings(session)
        logger.info(f"Generated {processed} averaged embeddings")


def build(num_pages: int = 25):
    """Run the full pipeline: fetch popular actors, download headshots, generate embeddings.

    Shortcut for running `fetch` then `embed`. Does not include fetch-credits
    or fetch-images — run those separately for a more complete dataset.
    """
    fetch(num_pages)
    embed()
    logger.info("Build complete!")


def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the FastAPI web server with hot-reload enabled."""
    import uvicorn

    uvicorn.run("familiar_actors.app:app", host=host, port=port, reload=True)


def main():
    commands = {
        "fetch": fetch,
        "fetch-credits": fetch_credits,
        "fetch-images": fetch_images,
        "embed": embed,
        "build": build,
        "serve": serve,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: familiar-actors <{'|'.join(commands)}>")
        print()
        print("Commands:")
        print(
            "  fetch [num_pages]              Fetch popular actors and headshots from TMDB"
        )
        print(
            "  fetch-credits [num_pages] [tv] Crawl cast from top-rated movies (or TV)"
        )
        print(
            "  fetch-images                   Download extra photos, generate averaged embeddings"
        )
        print("  embed                          Generate CLIP embeddings for headshots")
        print("  build [num_pages]              Run full pipeline (fetch + embed)")
        print("  serve                          Start the web server")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "fetch-credits":
        num_pages = 25
        source = "movie"
        for arg in args:
            if arg == "tv":
                source = "tv"
            else:
                num_pages = int(arg)
        fetch_credits(num_pages=num_pages, source=source)
    elif args and command in ("fetch", "build"):
        commands[command](num_pages=int(args[0]))
    elif args and command == "serve":
        commands[command](*args)
    else:
        commands[command]()


if __name__ == "__main__":
    main()
