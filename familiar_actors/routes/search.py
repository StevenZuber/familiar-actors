from fastapi import APIRouter, Depends, Query, Request
from sqlmodel import Session, select

from familiar_actors.database import get_session
from familiar_actors.models import Actor, ActorResult

router = APIRouter()


def get_index():
    from familiar_actors.app import index

    return index


def get_templates():
    from familiar_actors.app import templates

    return templates


@router.get("/")
async def home(request: Request):
    tmpl = get_templates()
    return tmpl.TemplateResponse("index.html", {"request": request})


@router.get("/api/search")
async def search_actors(
    q: str = Query(min_length=1),
    session: Session = Depends(get_session),
) -> list[dict]:
    """Search actors by name for autocomplete."""
    actors = session.exec(
        select(Actor)
        .where(Actor.name.ilike(f"%{q}%"))  # type: ignore[union-attr]
        .limit(10)
    ).all()
    return [
        {
            "id": a.id,
            "tmdb_id": a.tmdb_id,
            "name": a.name,
            "tmdb_image_url": a.tmdb_image_url,
        }
        for a in actors
    ]


@router.get("/api/similar/{actor_id}")
async def get_similar_actors(
    actor_id: int,
    session: Session = Depends(get_session),
) -> list[ActorResult]:
    """Get actors who look similar to the given actor."""
    similarity_index = get_index()
    return similarity_index.search(actor_id, session)


@router.get("/search")
async def search_page(
    request: Request,
    actor_id: int = Query(...),
    session: Session = Depends(get_session),
):
    """HTMX endpoint — returns rendered results partial."""
    tmpl = get_templates()
    similarity_index = get_index()

    actor = session.get(Actor, actor_id)
    results = similarity_index.search(actor_id, session)

    return tmpl.TemplateResponse(
        "results.html",
        {
            "request": request,
            "actor": actor,
            "results": results,
        },
    )
