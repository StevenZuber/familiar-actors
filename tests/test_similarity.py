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

    def test_search_by_vector_returns_ranked_results(self, actors_with_embeddings):
        session, _ = actors_with_embeddings
        index = SimilarityIndex()
        index.load(session)

        # Query vector close to Actor A's embedding; unnormalized on purpose
        results = index.search_by_vector(
            np.array([2.0, 0.0, 0.0, 0.2]), session, top_n=3
        )

        assert [r.name for r in results] == ["Actor A", "Actor B", "Actor C"]
        assert results[0].similarity_score > results[1].similarity_score

    def test_search_by_vector_excludes_actor(self, actors_with_embeddings):
        session, actors = actors_with_embeddings
        index = SimilarityIndex()
        index.load(session)

        results = index.search_by_vector(
            np.array([1.0, 0.0, 0.0, 0.1]),
            session,
            top_n=10,
            exclude_actor_id=actors[0].id,
        )

        assert actors[0].id not in [r.id for r in results]
        assert len(results) == 2

    def test_search_by_vector_zero_vector_returns_empty(self, actors_with_embeddings):
        session, _ = actors_with_embeddings
        index = SimilarityIndex()
        index.load(session)

        assert index.search_by_vector(np.zeros(4), session) == []

    def test_search_by_vector_on_empty_index_returns_empty(self, db_session):
        index = SimilarityIndex()
        assert index.search_by_vector(np.array([1.0, 0.0]), db_session) == []

    def test_load_facecrop_space(self, db_session, tmp_path, monkeypatch):
        """With embedding_space='facecrop', the index loads facecrop vectors."""
        from familiar_actors import similarity as sim

        emb_dir = tmp_path / "facecrop"
        emb_dir.mkdir()
        for i, vec in enumerate([[1.0, 0.0], [0.0, 1.0]]):
            p = emb_dir / f"{i}.npy"
            np.save(p, np.array(vec))
            db_session.add(
                Actor(tmdb_id=i, name=f"Actor {i}", facecrop_embedding_path=str(p))
            )
        db_session.commit()

        monkeypatch.setattr(sim.settings, "embedding_space", "facecrop")
        index = SimilarityIndex()
        index.load(db_session)

        assert index.is_loaded
        assert len(index.actor_ids) == 2

    def test_clip_space_ignores_facecrop_only_actors(
        self, db_session, tmp_path, monkeypatch
    ):
        """In the default 'clip' space, an actor with only a facecrop embedding
        is not loaded — the spaces are kept separate."""
        from familiar_actors import similarity as sim

        p = tmp_path / "fc.npy"
        np.save(p, np.array([1.0, 0.0]))
        db_session.add(
            Actor(tmdb_id=99, name="FaceOnly", facecrop_embedding_path=str(p))
        )
        db_session.commit()

        monkeypatch.setattr(sim.settings, "embedding_space", "clip")
        # Isolate data_dir so the real ./data consolidated index isn't picked up.
        monkeypatch.setattr(sim.settings, "data_dir", tmp_path)
        index = SimilarityIndex()
        index.load(db_session)
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
            mock_settings.embedding_space = "clip"
            mock_settings.consolidated_index_paths.return_value = (
                data_dir / "embeddings_index.npy",
                data_dir / "embeddings_ids.json",
            )
            index.load(db_session)

        assert index.is_loaded
        assert len(index.actor_ids) == 3
        assert index.embeddings.shape == (3, 4)

        # Verify normalization
        norms = np.linalg.norm(index.embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)
