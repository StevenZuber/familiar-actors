from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, or_
from sqlmodel import Session, select

from familiar_actors.database import get_session
from familiar_actors.models import Actor, ActorResult
from familiar_actors.tmdb import TMDBClient

router = APIRouter()


def get_index():
    from familiar_actors.app import index

    return index


def get_templates():
    from familiar_actors.app import templates

    return templates


def get_search_index():
    from familiar_actors.app import search_index

    return search_index


def get_tmdb_client():
    return TMDBClient()


def is_htmx_request(request: Request) -> bool:
    """Check if the request was made by HTMX (partial) vs direct navigation (full page)."""
    return request.headers.get("HX-Request") == "true"


@router.get("/")
async def home(request: Request):
    tmpl = get_templates()
    return tmpl.TemplateResponse("index.html", {"request": request})


@router.get("/about")
async def about(request: Request, session: Session = Depends(get_session)):
    """About page with project info and live dataset stats."""
    tmpl = get_templates()
    actor_count = session.exec(select(func.count()).select_from(Actor)).one()
    embedding_count = session.exec(
        select(func.count())
        .select_from(Actor)
        .where(
            or_(
                Actor.clip_avg_embedding_path.isnot(None),  # type: ignore[union-attr]
                Actor.clip_embedding_path.isnot(None),  # type: ignore[union-attr]
            )
        )
    ).one()
    return tmpl.TemplateResponse(
        "about.html",
        {
            "request": request,
            "actor_count": actor_count,
            "embedding_count": embedding_count,
        },
    )


@router.get("/technical")
async def technical(request: Request, session: Session = Depends(get_session)):
    """Technical deep dive page with live dataset stats."""
    tmpl = get_templates()
    actor_count = session.exec(select(func.count()).select_from(Actor)).one()
    embedding_count = session.exec(
        select(func.count())
        .select_from(Actor)
        .where(
            or_(
                Actor.clip_avg_embedding_path.isnot(None),  # type: ignore[union-attr]
                Actor.clip_embedding_path.isnot(None),  # type: ignore[union-attr]
            )
        )
    ).one()
    return tmpl.TemplateResponse(
        "technical.html",
        {
            "request": request,
            "actor_count": actor_count,
            "embedding_count": embedding_count,
        },
    )


@router.get("/api/search")
async def search_actors(
    q: str = Query(min_length=1),
) -> list[dict]:
    """Search actors by name for autocomplete. Prefix match first, fuzzy fallback."""
    actor_search = get_search_index()
    return actor_search.search(q, limit=10)


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
    """Returns results partial for HTMX, or full page for direct navigation."""
    tmpl = get_templates()
    similarity_index = get_index()

    actor = session.get(Actor, actor_id)
    results = similarity_index.search(actor_id, session)

    context = {
        "request": request,
        "actor": actor,
        "results": results,
    }

    if is_htmx_request(request):
        return tmpl.TemplateResponse("results.html", context)

    context.update(
        {
            "partial_template": "results.html",
            "search_mode": "actor",
            "search_value": actor.name if actor else "",
        }
    )
    return tmpl.TemplateResponse("full_page.html", context)


@router.get("/api/search-titles")
async def search_titles(
    q: str = Query(min_length=1),
) -> list[dict]:
    """Search movies and TV shows by title via TMDB."""
    client = get_tmdb_client()
    return await client.search_titles(q)


CAST_INITIAL_LIMIT = 20


@router.get("/cast")
async def cast_page(
    request: Request,
    title_id: int = Query(...),
    source: str = Query("movie", pattern="^(movie|tv)$"),
    show_all: bool = Query(False),
    session: Session = Depends(get_session),
):
    """HTMX endpoint — show cast for a movie/show."""
    tmpl = get_templates()
    client = get_tmdb_client()

    title_name, cast = await client.fetch_cast(title_id, source)

    total_cast_count = len(cast)
    has_more = not show_all and total_cast_count > CAST_INITIAL_LIMIT
    visible_cast = cast if show_all else cast[:CAST_INITIAL_LIMIT]

    # Check which cast members are already in our database
    cast_with_db_info = []
    for member in visible_cast:
        actor = session.exec(
            select(Actor).where(Actor.tmdb_id == member["tmdb_id"])
        ).first()
        cast_with_db_info.append(
            {
                **member,
                "actor_id": actor.id if actor else None,
                "in_database": actor is not None
                and (
                    actor.clip_avg_embedding_path is not None
                    or actor.clip_embedding_path is not None
                ),
            }
        )

    context = {
        "request": request,
        "title_name": title_name,
        "cast": cast_with_db_info,
        "has_more": has_more,
        "title_id": title_id,
        "source": source,
        "remaining_count": total_cast_count - CAST_INITIAL_LIMIT if has_more else 0,
    }

    if is_htmx_request(request):
        return tmpl.TemplateResponse("cast.html", context)

    context.update(
        {
            "partial_template": "cast.html",
            "search_mode": "title",
            "search_value": title_name,
        }
    )
    return tmpl.TemplateResponse("full_page.html", context)
