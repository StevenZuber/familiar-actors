from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from familiar_actors.models import Actor


@pytest.fixture
def db_engine(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def seeded_db(db_engine, tmp_path):
    """Seed the database with actors, some with embeddings."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    with Session(db_engine) as session:
        # Actor with embedding
        emb_path = embeddings_dir / "100.npy"
        np.save(emb_path, np.array([1.0, 0.0, 0.0, 0.0]))

        session.add(
            Actor(
                tmdb_id=100,
                name="Tom Hanks",
                tmdb_image_url="https://image.tmdb.org/t/p/w185/test1.jpg",
                clip_embedding_path=str(emb_path),
            )
        )

        emb_path_2 = embeddings_dir / "200.npy"
        np.save(emb_path_2, np.array([0.9, 0.1, 0.0, 0.0]))

        session.add(
            Actor(
                tmdb_id=200,
                name="Tom Cruise",
                tmdb_image_url="https://image.tmdb.org/t/p/w185/test2.jpg",
                clip_embedding_path=str(emb_path_2),
            )
        )

        # Actor without embedding
        session.add(
            Actor(
                tmdb_id=300,
                name="Brad Pitt",
                tmdb_image_url="https://image.tmdb.org/t/p/w185/test3.jpg",
            )
        )

        session.commit()

    return db_engine


@pytest.fixture
def client(seeded_db):
    """Create a test client with a seeded database."""
    from familiar_actors import app as app_module

    # Patch the engine and recreate the app dependencies
    app_module.engine

    with patch.object(app_module, "engine", seeded_db):
        from familiar_actors.app import app
        from familiar_actors.database import get_session
        from familiar_actors.similarity import SimilarityIndex

        def override_get_session():
            with Session(seeded_db) as session:
                yield session

        app.dependency_overrides[get_session] = override_get_session

        # Load the similarity index with test data
        with Session(seeded_db) as session:
            app_module.index = SimilarityIndex()
            app_module.index.load(session)

        yield TestClient(app, raise_server_exceptions=False)

        app.dependency_overrides.clear()


@pytest.mark.unit
class TestSearchActorsAPI:
    def test_search_by_name(self, client):
        response = client.get("/api/search?q=Tom")
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 2
        names = {r["name"] for r in results}
        assert names == {"Tom Hanks", "Tom Cruise"}

    def test_search_no_match(self, client):
        response = client.get("/api/search?q=zzzzzzz")
        assert response.status_code == 200
        assert response.json() == []

    def test_search_requires_query(self, client):
        response = client.get("/api/search?q=")
        assert response.status_code == 422


@pytest.mark.unit
class TestSimilarActorsAPI:
    def test_similar_returns_results(self, client):
        response = client.get("/api/similar/1")
        assert response.status_code == 200
        results = response.json()
        assert len(results) >= 1
        assert results[0]["name"] == "Tom Cruise"

    def test_similar_unknown_actor(self, client):
        response = client.get("/api/similar/9999")
        assert response.status_code == 200
        assert response.json() == []


@pytest.mark.unit
class TestSearchPage:
    def test_home_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Familiar Actors" in response.text

    def test_search_htmx_endpoint(self, client):
        response = client.get("/search?actor_id=1")
        assert response.status_code == 200
        assert "Actors who look like" in response.text


@pytest.mark.unit
class TestSearchTitlesAPI:
    def test_search_titles_calls_tmdb(self, client):
        mock_results = [
            {"tmdb_id": 550, "title": "Fight Club", "year": "1999", "source": "movie"}
        ]
        with patch("familiar_actors.routes.search.get_tmdb_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.search_titles.return_value = mock_results
            mock_get_client.return_value = mock_client

            response = client.get("/api/search-titles?q=fight")
            assert response.status_code == 200
            results = response.json()
            assert len(results) == 1
            assert results[0]["title"] == "Fight Club"


@pytest.mark.unit
class TestCastPage:
    def test_cast_page_renders(self, client):
        mock_cast = [
            {
                "tmdb_id": 100,
                "name": "Tom Hanks",
                "character": "Forrest Gump",
                "image_url": "https://image.tmdb.org/t/p/w185/test.jpg",
            }
        ]
        with patch("familiar_actors.routes.search.get_tmdb_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.fetch_cast.return_value = ("Forrest Gump", mock_cast)
            mock_get_client.return_value = mock_client

            response = client.get("/cast?title_id=13&source=movie")
            assert response.status_code == 200
            assert "Cast of Forrest Gump" in response.text
            assert "Tom Hanks" in response.text
