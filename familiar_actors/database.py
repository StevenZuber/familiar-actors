import logging

from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, create_engine

from familiar_actors.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(settings.database_url, echo=False)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    _run_migrations()


def _run_migrations():
    """Add any columns that exist in the model but not in the database."""
    inspector = inspect(engine)
    if not inspector.has_table("actor"):
        return

    existing_columns = {col["name"] for col in inspector.get_columns("actor")}
    if "clip_embedding_path" not in existing_columns:
        logger.info("Migrating: adding clip_embedding_path column to actor table")
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE actor ADD COLUMN clip_embedding_path TEXT"))


def get_session():
    with Session(engine) as session:
        yield session
