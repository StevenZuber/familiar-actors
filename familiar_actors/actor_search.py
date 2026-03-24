import logging
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process
from sqlmodel import Session, select

from familiar_actors.models import Actor

logger = logging.getLogger(__name__)


@dataclass
class ActorSearchIndex:
    """In-memory index of actor names for fast fuzzy search.

    Loaded once at app startup alongside the embedding index. Provides
    two search strategies: prefix matching (fast, for normal autocomplete)
    and fuzzy matching via rapidfuzz (for typos and misspellings).
    """

    _names: list[str] = field(default_factory=list)
    _ids: list[int] = field(default_factory=list)
    _tmdb_ids: list[int] = field(default_factory=list)
    _image_urls: list[str | None] = field(default_factory=list)
    _names_lower: list[str] = field(default_factory=list)

    @property
    def is_loaded(self) -> bool:
        return len(self._names) > 0

    def load(self, session: Session) -> None:
        """Load all actor names and metadata into memory."""
        actors = [a for a in session.exec(select(Actor)).all() if a.id is not None]
        self._names = [a.name for a in actors]
        self._ids = [a.id for a in actors]  # type: ignore[misc]
        self._tmdb_ids = [a.tmdb_id for a in actors]
        self._image_urls = [a.tmdb_image_url for a in actors]
        self._names_lower = [n.lower() for n in self._names]
        logger.info(f"Loaded {len(self._names)} actor names into search index")

    def _to_result(self, idx: int) -> dict:
        """Convert an index position to the API response dict."""
        return {
            "id": self._ids[idx],
            "tmdb_id": self._tmdb_ids[idx],
            "name": self._names[idx],
            "tmdb_image_url": self._image_urls[idx],
        }

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search for actors by name. Tries prefix match first, falls back to fuzzy.

        Returns results matching the /api/search response shape:
        [{id, tmdb_id, name, tmdb_image_url}, ...]
        """
        if not self.is_loaded:
            return []

        # Try prefix match first — fast and natural for autocomplete
        results = self._prefix_search(query, limit)
        if results:
            return results

        # Fall back to fuzzy match — catches typos and misspellings
        return self._fuzzy_search(query, limit)

    def _prefix_search(self, query: str, limit: int) -> list[dict]:
        """Case-insensitive prefix match on actor names."""
        query_lower = query.lower()
        matches = []
        for i, name in enumerate(self._names_lower):
            if name.startswith(query_lower):
                matches.append(i)
                if len(matches) >= limit:
                    break
        return [self._to_result(i) for i in matches]

    def _fuzzy_search(self, query: str, limit: int) -> list[dict]:
        """Fuzzy match using rapidfuzz WRatio scorer."""
        matches = process.extract(
            query,
            self._names,
            scorer=fuzz.WRatio,
            limit=limit,
            score_cutoff=60,
        )
        return [self._to_result(idx) for _, _, idx in matches]
