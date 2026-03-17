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
                    if not person.get("profile_path"):
                        continue

                    # Skip if we already have this actor
                    existing = session.exec(
                        select(Actor).where(Actor.tmdb_id == person["id"])
                    ).first()
                    if existing:
                        actors.append(existing)
                        continue

                    actor = Actor(
                        tmdb_id=person["id"],
                        name=person["name"],
                        tmdb_image_url=f"{TMDB_IMAGE_BASE_URL}{person['profile_path']}",
                    )
                    session.add(actor)
                    session.commit()
                    session.refresh(actor)
                    actors.append(actor)

                logger.info(f"Fetched page {page}/{num_pages} ({len(actors)} actors)")

        return actors

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
