"""Background data crawler for Familiar Actors.

Systematically crawls TMDB's discover endpoints to find actors across
decades of film and TV. Designed to run for hours/days unattended.

IMPORTANT: This script has TWO PHASES that run sequentially:

  Phase 1: DISCOVER — Crawls TMDB's discover endpoints to find movies/TV shows,
           fetches cast lists for each, and inserts actor metadata into the database.
           This is fast (mostly small JSON responses) and adds actors as rows in the DB,
           but does NOT download headshot images.

  Phase 2: DOWNLOAD HEADSHOTS — After discovery is complete, downloads the primary
           headshot image for every actor that doesn't have one yet. This is slower
           (one image per actor, rate-limited) and can take a long time for large batches.

  After both phases, you still need to run separately:
    - `uv run familiar-actors embed` to generate CLIP embeddings from the headshots
    - `uv run python scripts/consolidate_index.py` to rebuild the Railway deployment index

FULL PIPELINE (end to end):
    # 1. Crawl actors + download headshots (takes days)
    caffeinate -i uv run python scripts/crawl.py

    # 2. Generate CLIP embeddings for all new headshots (takes hours)
    caffeinate -i uv run familiar-actors embed

    # 3. (Optional) Generate averaged embeddings from multiple photos per actor
    caffeinate -i uv run familiar-actors fetch-images

    # 4. Rebuild consolidated index for Railway deployment
    uv run python scripts/consolidate_index.py

    # 5. Check stats
    uv run python scripts/crawl.py --stats

CRAWLER USAGE:
    # Crawl movies AND tv (default: both, 1970-2026)
    caffeinate -i uv run python scripts/crawl.py

    # Crawl only movies
    caffeinate -i uv run python scripts/crawl.py --source movie

    # Crawl only TV shows
    caffeinate -i uv run python scripts/crawl.py --source tv

    # Crawl a specific year range
    caffeinate -i uv run python scripts/crawl.py --years 2000-2010

    # Crawl actors only, skip headshot downloads (faster, metadata only)
    caffeinate -i uv run python scripts/crawl.py --skip-headshots

    # Deep crawl — use 5 different sort orders per year to find actors beyond
    # the 500-page limit. ~5x more coverage, ~5x longer runtime.
    # Sort orders: popularity, vote_count, revenue, release_date, vote_average
    caffeinate -i uv run python scripts/crawl.py --deep

    # Check database stats without crawling
    uv run python scripts/crawl.py --stats

    # Start fresh (delete resume state file)
    uv run python scripts/crawl.py --reset

RESUME BEHAVIOR:
    The script saves progress to data/crawl_state.json after each page.
    If interrupted (Ctrl+C, laptop sleep, crash), re-running the same command
    picks up where it left off. Movie and TV crawls track progress independently.
    Delete the state file (--reset) to start a crawl from scratch.

RATE LIMITING:
    TMDB allows 40 requests per 10 seconds. This script uses a concurrency
    semaphore with delays to stay safely under that limit. If you hit 429
    errors, the script backs off and retries automatically.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import httpx
from sqlmodel import Session, select

# Add project root to path so we can import familiar_actors
sys.path.insert(0, str(Path(__file__).parent.parent))

from familiar_actors.config import settings
from familiar_actors.database import create_db_and_tables, engine
from familiar_actors.models import Actor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.data_dir / "crawl.log"),
    ],
)
logger = logging.getLogger(__name__)

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w185"
STATE_FILE = settings.data_dir / "crawl_state.json"
MAX_CONCURRENT = 6
RATE_LIMIT_DELAY = (
    0.2  # seconds between requests per slot (~30 req/10s, under TMDB's 40/10s limit)
)


def get_headers() -> dict:
    if not settings.tmdb_read_access_token:
        raise RuntimeError(
            "TMDB_READ_ACCESS_TOKEN is not set. Add it to your .env file."
        )
    return {
        "Authorization": f"Bearer {settings.tmdb_read_access_token}",
        "accept": "application/json",
    }


def load_state() -> dict:
    """Load crawl progress from state file."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    """Save crawl progress to state file."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def upsert_actor(session: Session, person: dict) -> bool:
    """Insert an actor if they don't exist. Returns True if new actor added."""
    if not person.get("profile_path"):
        return False

    existing = session.exec(select(Actor).where(Actor.tmdb_id == person["id"])).first()
    if existing:
        return False

    actor = Actor(
        tmdb_id=person["id"],
        name=person["name"],
        tmdb_image_url=f"{TMDB_IMAGE_BASE_URL}{person['profile_path']}",
    )
    session.add(actor)
    session.commit()
    return True


async def rate_limited_get(
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore, url: str, **kwargs
) -> httpx.Response | None:
    """Make a rate-limited GET request with retry on 429."""
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.get(url, **kwargs)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", "2"))
                    logger.warning(f"Rate limited, waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue
                response.raise_for_status()
                await asyncio.sleep(RATE_LIMIT_DELAY)
                return response
            except httpx.HTTPError as e:
                if attempt == 2:
                    logger.warning(f"Failed after 3 attempts: {url} — {e}")
                    return None
                await asyncio.sleep(1)
    return None


SORT_ORDERS = [
    "popularity.desc",
    "vote_count.desc",
    "revenue.desc",
    "primary_release_date.desc",
    "vote_average.desc",
]


async def crawl_discover(
    source: str,
    year: int,
    page: int,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    session: Session,
    sort_by: str = "popularity.desc",
) -> tuple[int, int]:
    """Crawl one page of discover results and fetch cast for each title.

    Returns (new_actors_count, titles_processed).
    """
    new_actors = 0
    titles_processed = 0

    # Fetch discover page
    params = {"page": page, "sort_by": sort_by}
    if source == "movie":
        params["primary_release_year"] = year
    else:
        params["first_air_date_year"] = year

    response = await rate_limited_get(
        client, semaphore, f"{TMDB_BASE_URL}/discover/{source}", params=params
    )
    if not response:
        return 0, 0

    titles = response.json().get("results", [])

    for title in titles:
        title_id = title["id"]

        # Fetch credits
        if source == "tv":
            credits_url = f"{TMDB_BASE_URL}/tv/{title_id}/aggregate_credits"
        else:
            credits_url = f"{TMDB_BASE_URL}/{source}/{title_id}/credits"

        credits_response = await rate_limited_get(client, semaphore, credits_url)
        if not credits_response:
            continue

        cast = credits_response.json().get("cast", [])
        for person in cast:
            if upsert_actor(session, person):
                new_actors += 1

        titles_processed += 1

    return new_actors, titles_processed


async def run_discover_crawl(
    source: str,
    start_year: int,
    end_year: int,
    max_pages: int,
    deep: bool = False,
):
    """Crawl TMDB discover endpoint across a range of years.

    In normal mode, uses popularity.desc sort only (up to 500 pages/year).
    In deep mode, cycles through 5 different sort orders per year, effectively
    multiplying coverage by up to 5x to get past TMDB's 500 page limit.
    """
    headers = get_headers()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    sort_orders = SORT_ORDERS if deep else ["popularity.desc"]

    total_new = 0
    total_titles = 0
    start_time = time.time()

    for sort_by in sort_orders:
        state = load_state()
        state_key = f"discover_{source}_{sort_by}"

        # Resume from last position for this sort order
        last_year = state.get(state_key, {}).get("last_year", start_year)
        last_page = state.get(state_key, {}).get("last_page", 0)

        sort_label = sort_by.replace(".", " ").replace("_", " ")
        logger.info(
            f"Starting {source} discover crawl (sort: {sort_label}): "
            f"years {last_year}-{end_year}, up to {max_pages} pages per year"
        )

        async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
            for year in range(last_year, end_year + 1):
                start_page = last_page + 1 if year == last_year else 1

                for page in range(start_page, max_pages + 1):
                    with Session(engine) as session:
                        new, titles = await crawl_discover(
                            source,
                            year,
                            page,
                            client,
                            semaphore,
                            session=session,
                            sort_by=sort_by,
                        )
                    total_new += new
                    total_titles += titles

                    # Save state after each page
                    if state_key not in state:
                        state[state_key] = {}
                    state[state_key]["last_year"] = year
                    state[state_key]["last_page"] = page
                    save_state(state)

                    # No titles on this page = no more pages for this year
                    if titles == 0:
                        break

                    elapsed = time.time() - start_time
                    years_done = year - start_year + (page / max_pages)
                    years_total = end_year - start_year + 1
                    if years_done > 0:
                        rate = elapsed / years_done
                        remaining = (years_total - years_done) * rate
                        hours, rem = divmod(int(remaining), 3600)
                        mins = rem // 60
                        eta = f"{hours}h {mins}m" if hours else f"{mins}m"
                    else:
                        eta = "calculating..."
                    logger.info(
                        f"[{source}/{sort_label}] Year {year}, page {page}: "
                        f"+{new} new actors ({total_new} total new, "
                        f"{total_titles} titles crawled, "
                        f"{elapsed:.0f}s elapsed, ~{eta} remaining)"
                    )

                # Reset page counter for next year
                last_page = 0

        logger.info(f"Completed sort order: {sort_label}")

    elapsed = time.time() - start_time
    logger.info(
        f"Discover crawl complete: {total_new} new actors from "
        f"{total_titles} titles in {elapsed:.0f}s"
    )


async def download_new_headshots():
    """Download headshots for all actors that don't have one yet."""
    get_headers()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    with Session(engine) as session:
        actors = session.exec(
            select(Actor).where(
                Actor.tmdb_image_url.isnot(None),  # type: ignore[union-attr]
                Actor.image_path.is_(None),  # type: ignore[union-attr]
            )
        ).all()

    if not actors:
        logger.info("No headshots to download")
        return

    settings.headshots_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    start_time = time.time()

    logger.info(f"Downloading headshots for {len(actors)} actors...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        for actor in actors:
            if not actor.tmdb_image_url:
                continue

            response = await rate_limited_get(client, semaphore, actor.tmdb_image_url)
            if not response:
                continue

            image_path = settings.headshots_dir / f"{actor.tmdb_id}.jpg"
            image_path.write_bytes(response.content)

            with Session(engine) as session:
                db_actor = session.get(Actor, actor.id)
                if db_actor:
                    db_actor.image_path = str(image_path)
                    session.add(db_actor)
                    session.commit()

            downloaded += 1
            if downloaded % 100 == 0:
                elapsed = time.time() - start_time
                rate = downloaded / elapsed if elapsed > 0 else 0
                remaining = (len(actors) - downloaded) / rate if rate > 0 else 0
                hours, rem = divmod(int(remaining), 3600)
                mins = rem // 60
                eta = f"{hours}h {mins}m" if hours else f"{mins}m"
                logger.info(
                    f"Downloaded {downloaded}/{len(actors)} headshots "
                    f"({elapsed:.0f}s elapsed, ~{eta} remaining)"
                )

    logger.info(f"Downloaded {downloaded} headshots")


def print_stats():
    """Print current database statistics."""
    with Session(engine) as session:
        total = session.exec(select(Actor)).all()
        with_headshots = [a for a in total if a.image_path]
        with_embeddings = [
            a for a in total if a.clip_embedding_path or a.clip_avg_embedding_path
        ]

    logger.info(f"Database stats:")
    logger.info(f"  Total actors:      {len(total):,}")
    logger.info(f"  With headshots:    {len(with_headshots):,}")
    logger.info(f"  With embeddings:   {len(with_embeddings):,}")
    logger.info(f"  Need headshots:    {len(total) - len(with_headshots):,}")
    logger.info(f"  Need embeddings:   {len(with_headshots) - len(with_embeddings):,}")


def main():
    parser = argparse.ArgumentParser(
        description="Crawl TMDB for actor data. Use with: caffeinate -i uv run python scripts/crawl.py"
    )
    parser.add_argument(
        "--source",
        choices=["movie", "tv", "both"],
        default="both",
        help="Crawl movies, TV shows, or both (default: both)",
    )
    parser.add_argument(
        "--years",
        default="1970-2026",
        help="Year range to crawl, e.g. 1990-2010 (default: 1970-2026)",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=500,
        help="Max pages per year (default: 500, TMDB max)",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Cycle through 5 sort orders per year to find more actors beyond the 500-page limit",
    )
    parser.add_argument(
        "--skip-headshots",
        action="store_true",
        help="Skip headshot downloads (just crawl actors)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print database stats and exit",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete crawl state file and start fresh",
    )
    args = parser.parse_args()

    create_db_and_tables()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    if args.stats:
        print_stats()
        return

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        logger.info("Crawl state reset")

    start_year, end_year = map(int, args.years.split("-"))

    print_stats()
    logger.info("---")

    # Crawl discover endpoints
    sources = ["movie", "tv"] if args.source == "both" else [args.source]
    for source in sources:
        asyncio.run(
            run_discover_crawl(source, start_year, end_year, args.pages, deep=args.deep)
        )

    # Download headshots
    if not args.skip_headshots:
        asyncio.run(download_new_headshots())

    logger.info("---")
    print_stats()
    logger.info(
        "Next steps: run 'uv run familiar-actors embed' to generate embeddings, "
        "then 'uv run python scripts/consolidate_index.py' to rebuild the index."
    )


if __name__ == "__main__":
    main()
