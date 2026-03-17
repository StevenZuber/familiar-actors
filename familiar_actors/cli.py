import asyncio
import logging
import sys

from sqlmodel import Session

from familiar_actors.config import settings
from familiar_actors.database import create_db_and_tables, engine
from familiar_actors.embeddings import process_all_embeddings
from familiar_actors.tmdb import TMDBClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch(num_pages: int = 25):
    """Fetch popular actors and download their headshots from TMDB."""
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
    """Generate face embeddings for all downloaded headshots."""
    create_db_and_tables()
    settings.embeddings_dir.mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        logger.info("Generating face embeddings...")
        processed = process_all_embeddings(session)
        logger.info(f"Processed {processed} embeddings")


def build(num_pages: int = 25):
    """Run the full pipeline: fetch actors, download headshots, generate embeddings."""
    fetch(num_pages)
    embed()
    logger.info("Build complete!")


def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the web server."""
    import uvicorn

    uvicorn.run("familiar_actors.app:app", host=host, port=port, reload=True)


def main():
    commands = {
        "fetch": fetch,
        "embed": embed,
        "build": build,
        "serve": serve,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: familiar-actors <{'|'.join(commands)}>")
        print()
        print("Commands:")
        print("  fetch [num_pages]  Fetch actors and headshots from TMDB")
        print("  embed              Generate face embeddings for headshots")
        print("  build [num_pages]  Run full pipeline (fetch + embed)")
        print("  serve              Start the web server")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    fn = commands[command]
    if args and command in ("fetch", "build"):
        fn(num_pages=int(args[0]))
    elif args and command == "serve":
        # Allow serve host:port
        fn(*args)
    else:
        fn()


if __name__ == "__main__":
    main()
