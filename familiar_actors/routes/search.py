import io
import logging

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener  # type: ignore[import-untyped]
from sqlalchemy import func, or_
from sqlmodel import Session, select

from familiar_actors.config import settings
from familiar_actors.database import get_session
from familiar_actors.face_detect import detect_and_crop
from familiar_actors.models import Actor, ActorResult
from familiar_actors.query_embedding import QueryEmbeddingUnavailable, embed_image
from familiar_actors.tmdb import TMDBClient

# iPhones (the main audience) upload HEIC when the client-side JPEG
# conversion doesn't run; this teaches Pillow to open it.
register_heif_opener()

logger = logging.getLogger(__name__)

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


MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _embed_query(image):
    """Embed an uploaded photo in the live embedding space.

    In the "facecrop" space the face is detected and cropped first (so the
    query matches the face-crop index); returns None if no face is found. In
    the "clip" space the whole image is embedded. Runs in a threadpool — both
    detection and ONNX inference are CPU-bound.
    """
    if settings.embedding_space == "facecrop":
        face = detect_and_crop(image)
        if face is None:
            return None
        image = face
    return embed_image(image)


@router.post("/upload")
async def upload_search(
    request: Request,
    photo: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    """Match an uploaded photo against the actor index.

    The photo is processed entirely in memory and never written to disk.
    Always returns the upload_results.html partial — errors render as a
    friendly message in place of the results grid.
    """
    tmpl = get_templates()
    similarity_index = get_index()

    def error_response(message: str, status_code: int):
        return tmpl.TemplateResponse(
            "upload_results.html",
            {"request": request, "error": message, "results": []},
            status_code=status_code,
        )

    data = await photo.read()
    if len(data) > MAX_UPLOAD_BYTES:
        return error_response("That photo is too large. Try one under 10MB.", 413)

    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return error_response(
            "We couldn't read that file as an image. Try a JPEG or PNG.", 422
        )

    try:
        # CPU-bound detection + ONNX inference — keep it off the event loop.
        embedding = await run_in_threadpool(_embed_query, image)
    except QueryEmbeddingUnavailable as e:
        logger.warning(f"Photo search unavailable: {e}")
        return error_response(
            "Photo search isn't available right now. Try searching by name.", 503
        )

    if embedding is None:
        return error_response(
            "We couldn't find a face in that photo. Try a clear, front-facing one.",
            422,
        )

    results = similarity_index.search_by_vector(embedding, session)
    return tmpl.TemplateResponse(
        "upload_results.html",
        {"request": request, "results": results, "error": None},
    )


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
