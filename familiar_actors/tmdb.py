import asyncio
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
        """Insert an actor into the database if they don't already exist.

        Expects a TMDB person dict with 'id', 'name', and 'profile_path' fields.
        Skips people without a profile photo. Returns the Actor (new or existing),
        or None if skipped.
        """
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
        """Fetch popular actors from TMDB's /person/popular endpoint.

        Paginates through results, inserting new actors into the database.
        Skips actors already in the DB. Each page returns ~20 actors.
        Caps at ~10k actors (500 pages max on TMDB's end).
        """
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
        """Fetch actors by crawling cast lists from top-rated movies or TV shows.

        For each page of top-rated titles, fetches the full cast list and inserts
        new actors into the database. This is the primary way to discover obscure
        character actors who don't appear on the popular endpoint. Each movie
        typically yields 20-50 cast members.
        """
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
        """Search for movies and TV shows by title via TMDB's search endpoints.

        Queries both /search/movie and /search/tv, merges results, and sorts
        by TMDB popularity so the most recognizable titles surface first
        regardless of type. Returns up to `limit` results.
        """
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
                            "popularity": item.get("popularity", 0),
                        }
                    )

        results.sort(key=lambda x: x["popularity"], reverse=True)
        # Drop popularity before returning — frontend doesn't need it
        return [
            {k: v for k, v in r.items() if k != "popularity"} for r in results[:limit]
        ]

    async def fetch_cast(
        self, title_id: int, source: str = "movie"
    ) -> tuple[str, list[dict]]:
        """Fetch the cast for a specific movie or TV show from TMDB.

        For movies, uses /movie/{id}/credits (cast in billing order).
        For TV, uses /tv/{id}/aggregate_credits (all cast across all seasons,
        sorted by episode count). Returns a tuple of (title_name, cast_list)
        where each cast member has tmdb_id, name, character, and image_url.
        """
        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            # Get the title name
            detail_response = await client.get(
                f"{TMDB_BASE_URL}/{source}/{title_id}",
            )
            detail_response.raise_for_status()
            detail = detail_response.json()
            title_name = detail.get("title") or detail.get("name", "Unknown")

            # Get the cast — use aggregate_credits for TV to get full cast
            if source == "tv":
                credits_response = await client.get(
                    f"{TMDB_BASE_URL}/tv/{title_id}/aggregate_credits",
                )
            else:
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

            # aggregate_credits uses a roles array; regular credits uses a flat character string
            if source == "tv":
                roles = person.get("roles", [])
                character = roles[0].get("character", "") if roles else ""
                episode_count = person.get("total_episode_count", 0)
            else:
                character = person.get("character", "")
                episode_count = None

            cast.append(
                {
                    "tmdb_id": person["id"],
                    "name": person.get("name", "Unknown"),
                    "character": character,
                    "image_url": image_url,
                    "episode_count": episode_count,
                }
            )

        return title_name, cast

    async def fetch_person_images(
        self, tmdb_id: int, client: httpx.AsyncClient
    ) -> list[dict]:
        """Fetch all profile images for a person from TMDB's /person/{id}/images endpoint.

        Returns a list of profile dicts with file_path, width, height, vote_average, etc.
        These are community-uploaded portrait headshots, not on-set or group photos.
        """
        response = await client.get(
            f"{TMDB_BASE_URL}/person/{tmdb_id}/images",
        )
        response.raise_for_status()
        return response.json().get("profiles", [])

    async def download_multi_headshots(self, session: Session) -> int:
        """Download multiple headshots per actor for embedding averaging.

        For each actor without an averaged embedding, fetches their profile images
        from TMDB, filters by minimum resolution, takes the top N by community
        vote_average, and downloads them to data/headshots_multi/{tmdb_id}/.
        Skips actors already processed. Safe to interrupt and resume.
        """
        actors = session.exec(
            select(Actor).where(
                Actor.clip_avg_embedding_path.is_(None),  # type: ignore[union-attr]
                Actor.tmdb_id.isnot(None),  # type: ignore[union-attr]
            )
        ).all()

        if not actors:
            logger.info("No actors need multi-photo downloads")
            return 0

        settings.headshots_multi_dir.mkdir(parents=True, exist_ok=True)
        processed = 0
        image_base = f"https://image.tmdb.org/t/p/{settings.multi_image_size}"

        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            for actor in actors:
                try:
                    profiles = await self.fetch_person_images(actor.tmdb_id, client)

                    # Filter by minimum width, sort by vote_average, take top N
                    profiles = [
                        p
                        for p in profiles
                        if p.get("width", 0) >= settings.min_image_width
                    ]
                    profiles.sort(key=lambda p: p.get("vote_average", 0), reverse=True)
                    profiles = profiles[: settings.max_photos_per_actor]

                    if not profiles:
                        continue

                    actor_dir = settings.headshots_multi_dir / str(actor.tmdb_id)
                    actor_dir.mkdir(parents=True, exist_ok=True)

                    for i, profile in enumerate(profiles):
                        image_path = actor_dir / f"{i}.jpg"
                        if image_path.exists():
                            continue
                        img_response = await client.get(
                            f"{image_base}{profile['file_path']}"
                        )
                        img_response.raise_for_status()
                        image_path.write_bytes(img_response.content)

                    processed += 1
                    if processed % 50 == 0:
                        logger.info(
                            f"Downloaded multi-photos for {processed}/{len(actors)} actors"
                        )

                except httpx.HTTPError:
                    logger.warning(
                        f"Failed to fetch images for {actor.name} "
                        f"(tmdb_id={actor.tmdb_id})"
                    )

                # Respect TMDB rate limits (40 req/10s)
                await asyncio.sleep(0.25)

        logger.info(f"Downloaded multi-photos for {processed} actors")
        return processed

    async def download_headshots(self, session: Session) -> int:
        """Download primary headshots for all actors that don't have one yet.

        Downloads w185 thumbnails from TMDB's image CDN to data/headshots/{tmdb_id}.jpg.
        These are used for display in the UI and for single-photo CLIP embeddings.
        Skips actors that already have an image_path set.
        """
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
