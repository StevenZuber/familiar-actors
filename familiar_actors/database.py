import logging

from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, create_engine

from familiar_actors.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(settings.database_url, echo=False)


def create_db_and_tables():
    """Create the database and all tables if they don't exist, then run migrations.

    Called at the start of every CLI command to ensure the schema is up to date.
    Safe to call repeatedly — SQLModel's create_all is idempotent.
    """
    SQLModel.metadata.create_all(engine)
    _run_migrations()


def _run_migrations():
    """Add columns that exist in the model but not yet in the database.

    SQLModel's create_all won't ALTER existing tables, so new columns added
    to the Actor model need manual migration. Each migration checks whether
    the column already exists before running ALTER TABLE.
    """
    inspector = inspect(engine)
    if not inspector.has_table("actor"):
        return

    existing_columns = {col["name"] for col in inspector.get_columns("actor")}
    if "clip_embedding_path" not in existing_columns:
        logger.info("Migrating: adding clip_embedding_path column to actor table")
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE actor ADD COLUMN clip_embedding_path TEXT"))
    if "clip_avg_embedding_path" not in existing_columns:
        logger.info("Migrating: adding clip_avg_embedding_path column to actor table")
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE actor ADD COLUMN clip_avg_embedding_path TEXT")
            )


def get_session():
    """FastAPI dependency that yields a SQLModel session for the request lifecycle."""
    with Session(engine) as session:
        yield session
