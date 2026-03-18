import logging

import httpx
from sqlmodel import Session, select

from familiar_actors.config import settings
from familiar_actors.models import Actor

logger = logging.getLogger(__name__)

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w185"


class TMDBClient:
    def __init__(self):
        if not settings.tmdb_read_access_token:
            raise RuntimeError(
                "TMDB_READ_ACCESS_TOKEN is not set. Add it to your .env file."
            )
        self.headers = {
            "Authorization": f"Bearer {settings.tmdb_read_access_token}",
            "accept": "application/json",
        }

    def _upsert_actor(self, session: Session, person: dict) -> Actor | None:
        """Insert an actor if they don't already exist. Returns the Actor or None if skipped."""
        if not person.get("profile_path"):
            return None

        existing = session.exec(
            select(Actor).where(Actor.tmdb_id == person["id"])
        ).first()
        if existing:
            return existing

        actor = Actor(
            tmdb_id=person["id"],
            name=person["name"],
            tmdb_image_url=f"{TMDB_IMAGE_BASE_URL}{person['profile_path']}",
        )
        session.add(actor)
        session.commit()
        session.refresh(actor)
        return actor

    async def fetch_popular_actors(
        self, session: Session, num_pages: int = 25
    ) -> list[Actor]:
        """Fetch popular actors from TMDB and store them in the database."""
        actors: list[Actor] = []

        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            for page in range(1, num_pages + 1):
                response = await client.get(
                    f"{TMDB_BASE_URL}/person/popular",
                    params={"page": page},
                )
                response.raise_for_status()
                data = response.json()

                for person in data["results"]:
                    actor = self._upsert_actor(session, person)
                    if actor:
                        actors.append(actor)

                logger.info(f"Fetched page {page}/{num_pages} ({len(actors)} actors)")

        return actors

    async def fetch_actors_from_credits(
        self, session: Session, num_pages: int = 25, source: str = "movie"
    ) -> list[Actor]:
        """Fetch actors by crawling cast lists from top-rated movies or TV shows."""
        new_actors: list[Actor] = []

        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            for page in range(1, num_pages + 1):
                response = await client.get(
                    f"{TMDB_BASE_URL}/{source}/top_rated",
                    params={"page": page},
                )
                response.raise_for_status()
                titles = response.json()["results"]

                for title in titles:
                    title_name = title.get("title") or title.get("name", "Unknown")
                    title_id = title["id"]

                    credits_response = await client.get(
                        f"{TMDB_BASE_URL}/{source}/{title_id}/credits",
                    )
                    credits_response.raise_for_status()
                    cast = credits_response.json().get("cast", [])

                    for person in cast:
                        actor = self._upsert_actor(session, person)
                        if actor and actor.image_path is None:
                            new_actors.append(actor)

                    logger.info(
                        f"Page {page}/{num_pages} — {title_name}: "
                        f"{len(cast)} cast, {len(new_actors)} new actors total"
                    )

        return new_actors

    async def search_titles(self, query: str, limit: int = 10) -> list[dict]:
        """Search for movies and TV shows by title."""
        results = []

        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            for source in ("movie", "tv"):
                response = await client.get(
                    f"{TMDB_BASE_URL}/search/{source}",
                    params={"query": query, "page": 1},
                )
                response.raise_for_status()

                for item in response.json().get("results", []):
                    title = item.get("title") or item.get("name", "")
                    year = (
                        item.get("release_date", "")[:4]
                        or item.get("first_air_date", "")[:4]
                    )
                    results.append(
                        {
                            "tmdb_id": item["id"],
                            "title": title,
                            "year": year,
                            "source": source,
                        }
                    )

        # Sort by popularity (TMDB returns results pre-sorted, but we merged two lists)
        return results[:limit]

    async def fetch_cast(
        self, title_id: int, source: str = "movie"
    ) -> tuple[str, list[dict]]:
        """Fetch the cast for a movie or TV show. Returns (title_name, cast_list)."""
        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            # Get the title name
            detail_response = await client.get(
                f"{TMDB_BASE_URL}/{source}/{title_id}",
            )
            detail_response.raise_for_status()
            detail = detail_response.json()
            title_name = detail.get("title") or detail.get("name", "Unknown")

            # Get the cast
            credits_response = await client.get(
                f"{TMDB_BASE_URL}/{source}/{title_id}/credits",
            )
            credits_response.raise_for_status()
            raw_cast = credits_response.json().get("cast", [])

        cast = []
        for person in raw_cast:
            image_url = None
            if person.get("profile_path"):
                image_url = f"{TMDB_IMAGE_BASE_URL}{person['profile_path']}"
            cast.append(
                {
                    "tmdb_id": person["id"],
                    "name": person.get("name", "Unknown"),
                    "character": person.get("character", ""),
                    "image_url": image_url,
                }
            )

        return title_name, cast

    async def download_headshots(self, session: Session) -> int:
        """Download headshots for all actors that don't have one yet."""
        actors = session.exec(
            select(Actor).where(
                Actor.tmdb_image_url.isnot(None),  # type: ignore[union-attr]
                Actor.image_path.is_(None),  # type: ignore[union-attr]
            )
        ).all()

        if not actors:
            logger.info("No headshots to download")
            return 0

        settings.headshots_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            for actor in actors:
                if not actor.tmdb_image_url:
                    continue
                try:
                    response = await client.get(actor.tmdb_image_url)
                    response.raise_for_status()

                    image_path = settings.headshots_dir / f"{actor.tmdb_id}.jpg"
                    image_path.write_bytes(response.content)

                    actor.image_path = str(image_path)
                    session.add(actor)
                    session.commit()
                    downloaded += 1

                    if downloaded % 50 == 0:
                        logger.info(f"Downloaded {downloaded}/{len(actors)} headshots")

                except httpx.HTTPError:
                    logger.warning(
                        f"Failed to download headshot for {actor.name} "
                        f"(tmdb_id={actor.tmdb_id})"
                    )

        logger.info(f"Downloaded {downloaded} headshots")
        return downloaded
