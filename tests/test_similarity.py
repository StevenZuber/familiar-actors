import numpy as np
import pytest
from sqlmodel import Session, SQLModel, create_engine

from familiar_actors.models import Actor
from familiar_actors.similarity import SimilarityIndex


@pytest.fixture
def db_session(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def actors_with_embeddings(db_session, tmp_path):
    """Create actors with synthetic embeddings that have known similarity."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    # Actor A and B have similar embeddings, C is different
    vec_a = np.array([1.0, 0.0, 0.0, 0.1])
    vec_b = np.array([0.95, 0.05, 0.0, 0.1])  # very similar to A
    vec_c = np.array([0.0, 0.0, 1.0, 0.0])  # very different from A

    actors = []
    for i, (name, vec) in enumerate(
        [("Actor A", vec_a), ("Actor B", vec_b), ("Actor C", vec_c)]
    ):
        emb_path = embeddings_dir / f"{i}.npy"
        np.save(emb_path, vec)

        actor = Actor(
            tmdb_id=i,
            name=name,
            clip_embedding_path=str(emb_path),
        )
        db_session.add(actor)

    db_session.commit()
    for a in db_session.query(Actor).all():
        actors.append(a)

    return db_session, actors


@pytest.mark.unit
class TestSimilarityIndex:
    def test_load_populates_index(self, actors_with_embeddings):
        session, _ = actors_with_embeddings
        index = SimilarityIndex()
        index.load(session)

        assert index.is_loaded
        assert len(index.actor_ids) == 3

    def test_load_normalizes_embeddings(self, actors_with_embeddings):
        session, _ = actors_with_embeddings
        index = SimilarityIndex()
        index.load(session)

        assert index.embeddings is not None
        norms = np.linalg.norm(index.embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_search_returns_most_similar(self, actors_with_embeddings):
        session, actors = actors_with_embeddings
        index = SimilarityIndex()
        index.load(session)

        actor_a = actors[0]
        results = index.search(actor_a.id, session, top_n=2)

        assert len(results) == 2
        # Actor B should be the top match for Actor A
        assert results[0].name == "Actor B"
        # Actor C should be the least similar
        assert results[1].name == "Actor C"
        # B should have a higher similarity score than C
        assert results[0].similarity_score > results[1].similarity_score

    def test_search_excludes_self(self, actors_with_embeddings):
        session, actors = actors_with_embeddings
        index = SimilarityIndex()
        index.load(session)

        actor_a = actors[0]
        results = index.search(actor_a.id, session, top_n=10)

        result_ids = [r.id for r in results]
        assert actor_a.id not in result_ids

    def test_search_respects_top_n(self, actors_with_embeddings):
        session, actors = actors_with_embeddings
        index = SimilarityIndex()
        index.load(session)

        actor_a = actors[0]
        results = index.search(actor_a.id, session, top_n=1)

        assert len(results) == 1

    def test_search_unknown_actor_returns_empty(self, actors_with_embeddings):
        session, _ = actors_with_embeddings
        index = SimilarityIndex()
        index.load(session)

        results = index.search(9999, session)
        assert results == []

    def test_search_on_empty_index_returns_empty(self, db_session):
        index = SimilarityIndex()
        results = index.search(1, db_session)
        assert results == []

    def test_is_loaded_false_when_empty(self):
        index = SimilarityIndex()
        assert not index.is_loaded

    def test_load_consolidated_index(self, db_session, tmp_path):
        """Test loading from consolidated index files (the Railway code path)."""
        import json
        from unittest.mock import patch

        # Create consolidated index files
        ids = [1, 2, 3]
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.1],
                [0.95, 0.05, 0.0, 0.1],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        np.save(data_dir / "embeddings_index.npy", embeddings)
        with open(data_dir / "embeddings_ids.json", "w") as f:
            json.dump(ids, f)

        index = SimilarityIndex()
        with patch("familiar_actors.similarity.settings") as mock_settings:
            mock_settings.data_dir = data_dir
            mock_settings.similarity_top_n = 10
            index.load(db_session)

        assert index.is_loaded
        assert len(index.actor_ids) == 3
        assert index.embeddings.shape == (3, 4)

        # Verify normalization
        norms = np.linalg.norm(index.embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)
