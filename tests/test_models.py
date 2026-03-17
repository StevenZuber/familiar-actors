import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from familiar_actors.models import Actor, ActorResult


@pytest.fixture
def db_session(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.mark.unit
class TestActor:
    def test_create_actor(self, db_session):
        actor = Actor(tmdb_id=123, name="Test Actor")
        db_session.add(actor)
        db_session.commit()
        db_session.refresh(actor)

        assert actor.id is not None
        assert actor.tmdb_id == 123
        assert actor.name == "Test Actor"
        assert actor.image_path is None
        assert actor.embedding_path is None

    def test_tmdb_id_is_unique(self, db_session):
        actor1 = Actor(tmdb_id=123, name="Actor One")
        actor2 = Actor(tmdb_id=123, name="Actor Two")
        db_session.add(actor1)
        db_session.commit()

        db_session.add(actor2)
        with pytest.raises(Exception):
            db_session.commit()

    def test_query_by_name(self, db_session):
        db_session.add(Actor(tmdb_id=1, name="Tom Hanks"))
        db_session.add(Actor(tmdb_id=2, name="Tom Cruise"))
        db_session.add(Actor(tmdb_id=3, name="Brad Pitt"))
        db_session.commit()

        results = db_session.exec(
            select(Actor).where(Actor.name.ilike("%tom%"))  # type: ignore[union-attr]
        ).all()
        assert len(results) == 2


@pytest.mark.unit
class TestActorResult:
    def test_actor_result_creation(self):
        result = ActorResult(
            id=1,
            tmdb_id=123,
            name="Test Actor",
            tmdb_image_url="https://example.com/photo.jpg",
            similarity_score=0.9542,
        )
        assert result.name == "Test Actor"
        assert result.similarity_score == 0.9542

    def test_actor_result_nullable_image(self):
        result = ActorResult(
            id=1,
            tmdb_id=123,
            name="Test Actor",
            tmdb_image_url=None,
            similarity_score=0.5,
        )
        assert result.tmdb_image_url is None
