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
        # Distinct from Tom Hanks (~0.71 cosine) — below the dedup threshold.
        np.save(emb_path_2, np.array([0.7, 0.7, 0.0, 0.0]))

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
    from familiar_actors.actor_search import ActorSearchIndex
    from familiar_actors.app import app
    from familiar_actors.database import get_session
    from familiar_actors.similarity import SimilarityIndex

    def override_get_session():
        with Session(seeded_db) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session

    # Load the similarity and search indexes with test data
    with Session(seeded_db) as session:
        app_module.index = SimilarityIndex()
        app_module.index.load(session)
        app_module.search_index = ActorSearchIndex()
        app_module.search_index.load(session)

    yield TestClient(app)

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
class TestSearchPagination:
    def test_show_more_button_when_page_full(self, client):
        # One other embedded actor (Tom Cruise); limit=1 fills the page -> button.
        response = client.get(
            "/search?actor_id=1&limit=1", headers={"HX-Request": "true"}
        )
        assert response.status_code == 200
        assert "Show more" in response.text
        assert "limit=11" in response.text  # next page = limit + similarity_top_n

    def test_no_show_more_when_results_under_limit(self, client):
        response = client.get(
            "/search?actor_id=1&limit=50", headers={"HX-Request": "true"}
        )
        assert response.status_code == 200
        assert "Show more" not in response.text


@pytest.mark.unit
class TestSearchPage:
    def test_home_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Familiar Actors" in response.text

    def test_search_htmx_returns_partial(self, client):
        response = client.get("/search?actor_id=1", headers={"HX-Request": "true"})
        assert response.status_code == 200
        assert "Actors who look like" in response.text
        assert "<html" not in response.text

    def test_search_direct_returns_full_page(self, client):
        response = client.get("/search?actor_id=1")
        assert response.status_code == 200
        assert "Actors who look like" in response.text
        assert "actor-search" in response.text


@pytest.mark.unit
class TestAboutPage:
    def test_about_page_renders(self, client):
        response = client.get("/about")
        assert response.status_code == 200
        assert "Why this exists" in response.text
        assert "Tech stack" in response.text

    def test_about_page_has_live_stats(self, client):
        response = client.get("/about")
        assert response.status_code == 200
        # Should contain formatted numbers from the database
        assert "actors" in response.text.lower()


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


def _jpeg_bytes(size=(64, 64)) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color=(128, 64, 32)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.unit
class TestUploadSearch:
    def test_upload_returns_matches(self, client):
        # Embedding identical to Tom Hanks's vector -> he should match first
        with patch(
            "familiar_actors.routes.search.embed_image",
            return_value=np.array([1.0, 0.0, 0.0, 0.0]),
        ):
            response = client.post(
                "/upload", files={"photo": ("photo.jpg", _jpeg_bytes(), "image/jpeg")}
            )

        assert response.status_code == 200
        assert "Actors who look like your photo" in response.text
        assert response.text.index("Tom Hanks") < response.text.index("Tom Cruise")

    def test_upload_non_image_returns_friendly_error(self, client):
        response = client.post(
            "/upload", files={"photo": ("notes.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 422
        assert "read that file as an image" in response.text

    def test_upload_too_large_rejected(self, client):
        from familiar_actors.routes import search as search_routes

        big = b"x" * (search_routes.MAX_UPLOAD_BYTES + 1)
        response = client.post(
            "/upload", files={"photo": ("photo.jpg", big, "image/jpeg")}
        )
        assert response.status_code == 413
        assert "too large" in response.text


@pytest.mark.unit
class TestUploadFacecropMode:
    """Upload behaviour when the live embedding space is 'facecrop'."""

    def test_no_face_returns_friendly_error(self, client, monkeypatch):
        from familiar_actors.routes import search as s

        monkeypatch.setattr(s.settings, "embedding_space", "facecrop")
        monkeypatch.setattr(s, "detect_and_crop", lambda img: None)

        response = client.post(
            "/upload", files={"photo": ("p.jpg", _jpeg_bytes(), "image/jpeg")}
        )
        assert response.status_code == 422
        assert "find a face" in response.text

    def test_face_is_cropped_then_embedded(self, client, monkeypatch):
        from PIL import Image

        from familiar_actors.routes import search as s

        calls = {}

        def fake_crop(img):
            calls["cropped"] = True
            return Image.new("RGB", (64, 64))

        monkeypatch.setattr(s.settings, "embedding_space", "facecrop")
        monkeypatch.setattr(s, "detect_and_crop", fake_crop)
        monkeypatch.setattr(
            s, "embed_image", lambda img: np.array([1.0, 0.0, 0.0, 0.0])
        )

        response = client.post(
            "/upload", files={"photo": ("p.jpg", _jpeg_bytes(), "image/jpeg")}
        )
        assert response.status_code == 200
        assert calls.get("cropped") is True
        assert "Tom Hanks" in response.text

    def test_upload_when_encoder_unavailable_returns_503(self, client):
        from familiar_actors.query_embedding import QueryEmbeddingUnavailable

        with patch(
            "familiar_actors.routes.search.embed_image",
            side_effect=QueryEmbeddingUnavailable("no model"),
        ):
            response = client.post(
                "/upload", files={"photo": ("photo.jpg", _jpeg_bytes(), "image/jpeg")}
            )

        assert response.status_code == 503
        assert "available right now" in response.text


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
