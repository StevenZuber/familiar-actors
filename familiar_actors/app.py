from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session

from familiar_actors.database import create_db_and_tables, engine
from familiar_actors.routes.search import router as search_router
from familiar_actors.similarity import SimilarityIndex

index = SimilarityIndex()


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    with Session(engine) as session:
        index.load(session)
    yield


app = FastAPI(title="Familiar Actors", lifespan=lifespan)

templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

app.include_router(search_router)
