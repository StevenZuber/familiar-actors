# Familiar Actors — Project Plan

## Goal

A web app where you type an actor's name and get back a list of similar-looking actors. Solves the "who was I thinking of?" problem when you recognize someone but can't place them.

## Approach

Use CLIP embeddings (holistic visual similarity via OpenCLIP ViT-B-32) to numerically compare actor headshots sourced from TMDB, then rank by cosine similarity.

## Architecture

```text
┌─────────────────────────────────────────────────┐
│  Data Pipeline (CLI)                            │
│  TMDB API → Headshots → Face Embeddings → Store │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│  SQLite DB (actor metadata)                     │
│  + numpy arrays on disk (embeddings)            │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│  FastAPI Backend                                │
│  JSON API endpoints + Jinja2 templates + HTMX   │
└─────────────────────────────────────────────────┘
```

**Data pipeline** — batch process that fetches actors from TMDB, downloads headshots, generates embeddings. Run independently, re-run to grow the dataset.

**Web app** — reads the pre-built index, serves search UI, returns similar actors.

## Key Dependencies

| Package | Purpose |
| ------- | ------- |
| `fastapi` + `uvicorn` | API framework + server |
| `jinja2` + HTMX (CDN) | Server-rendered templates with async interactivity |
| `httpx` | Async HTTP client (TMDB API) |
| `open-clip-torch` | Visual similarity embeddings (ViT-B-32) |
| `numpy` | Embedding storage and cosine similarity |
| `sqlmodel` | Actor metadata in SQLite |
| `pydantic-settings` | Configuration (.env for API keys) |

## File Structure

```text
familiar_actors/
    __init__.py
    app.py              # FastAPI app, lifespan, route registration
    cli.py              # CLI: fetch, fetch-credits, embed, build, serve
    config.py           # Pydantic Settings (TMDB key, data paths, model config)
    models.py           # SQLModel Actor + Pydantic ActorResult
    database.py         # SQLite engine/session setup
    tmdb.py             # Async TMDB client (fetch actors, download images)
    embeddings.py       # OpenCLIP embedding generation
    similarity.py       # In-memory cosine similarity index
    routes/
        search.py       # JSON API + HTMX search endpoints
    templates/
        base.html       # Layout with HTMX script tag
        index.html      # Search page with autocomplete
        results.html    # Results grid partial (HTMX target)
    static/
        style.css
data/                   # gitignored — headshots, embeddings, SQLite DB
tests/
    test_models.py      # Actor model + DB tests
    test_similarity.py  # Similarity index tests with synthetic embeddings
```

## CLI Commands

```bash
uv run familiar-actors fetch [num_pages]              # Fetch popular actors + headshots
uv run familiar-actors fetch-credits [num_pages] [tv]  # Crawl cast from top-rated movies/TV
uv run familiar-actors embed                           # Generate CLIP embeddings
uv run familiar-actors build [num_pages]               # fetch + embed in one shot
uv run familiar-actors serve                           # Start FastAPI dev server
```

## TMDB Setup

Register at <https://www.themoviedb.org/settings/api> for a free Read Access Token. Store in `.env`:

```bash
TMDB_READ_ACCESS_TOKEN=your_token_here
```

## Progress

- [x] Phase 1: Configuration & data models
- [x] Phase 2: TMDB client & image pipeline
- [x] Phase 3: Face embedding generation
- [x] Phase 4: Similarity search
- [x] Phase 5: FastAPI backend
- [x] Phase 6: Web UI (Jinja2 + HTMX)
- [x] Phase 7: CLI for data pipeline
- [x] Unit tests (13 passing — models + similarity)
- [x] TMDB API key setup
- [x] First `build` run (~8k actors from popular endpoint)
- [x] Swap ArcFace → CLIP (dramatically better similarity results)
- [x] `fetch-credits` command to crawl movie/TV cast lists
- [x] Scale dataset via credits crawling (target: 50k+ actors)
- [ ] End-to-end verification with large dataset

## Scaling Path

**Data sources (in order of ROI):**

- `fetch` — popular actors endpoint (caps at ~10k)
- `fetch-credits` — cast from top-rated movies (10-25k new actors per 25 pages)
- `fetch-credits ... tv` — cast from top-rated TV shows
- Future: crawl by genre, decade, or discover endpoint for broader coverage

**Search performance:**

- numpy cosine similarity handles 10-50k actors fine in memory
- Beyond that: swap `similarity.py` internals to FAISS (same interface, faster search)

## Future Ideas

- Search by uploading a photo (not just by name)
- Filter by gender, age range, ethnicity
- Show what movies/shows the similar actors are known for
- "Explain" why two actors look similar (shared facial features)
