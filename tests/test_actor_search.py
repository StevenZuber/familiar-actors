import pytest
from sqlmodel import Session, SQLModel, create_engine

from familiar_actors.actor_search import ActorSearchIndex
from familiar_actors.models import Actor


@pytest.fixture
def db_session(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def search_index(db_session):
    """Create a search index with a handful of actors."""
    actors = [
        Actor(
            tmdb_id=1,
            name="Samuel L. Jackson",
            tmdb_image_url="https://example.com/1.jpg",
        ),
        Actor(tmdb_id=2, name="Tom Hanks", tmdb_image_url="https://example.com/2.jpg"),
        Actor(tmdb_id=3, name="Tom Cruise", tmdb_image_url="https://example.com/3.jpg"),
        Actor(tmdb_id=4, name="Brad Pitt", tmdb_image_url="https://example.com/4.jpg"),
        Actor(
            tmdb_id=5,
            name="Jessica Chastain",
            tmdb_image_url="https://example.com/5.jpg",
        ),
    ]
    for actor in actors:
        db_session.add(actor)
    db_session.commit()

    index = ActorSearchIndex()
    index.load(db_session)
    return index


@pytest.mark.unit
class TestActorSearchIndex:
    def test_load_populates_index(self, search_index):
        assert search_index.is_loaded
        assert len(search_index._names) == 5

    def test_prefix_match(self, search_index):
        results = search_index.search("Tom")
        names = {r["name"] for r in results}
        assert "Tom Hanks" in names
        assert "Tom Cruise" in names

    def test_prefix_match_case_insensitive(self, search_index):
        results = search_index.search("tom")
        names = {r["name"] for r in results}
        assert "Tom Hanks" in names

    def test_fuzzy_match_on_typo(self, search_index):
        results = search_index.search("Samule")
        names = {r["name"] for r in results}
        assert "Samuel L. Jackson" in names

    def test_fuzzy_match_reordered_tokens(self, search_index):
        results = search_index.search("Jackson Samuel")
        names = {r["name"] for r in results}
        assert "Samuel L. Jackson" in names

    def test_no_match_returns_empty(self, search_index):
        results = search_index.search("xyzzyplugh")
        assert results == []

    def test_returns_correct_shape(self, search_index):
        results = search_index.search("Brad")
        assert len(results) >= 1
        result = results[0]
        assert "id" in result
        assert "tmdb_id" in result
        assert "name" in result
        assert "tmdb_image_url" in result

    def test_respects_limit(self, search_index):
        results = search_index.search("T", limit=1)
        assert len(results) == 1

    def test_empty_index_returns_empty(self):
        index = ActorSearchIndex()
        assert index.search("anything") == []
