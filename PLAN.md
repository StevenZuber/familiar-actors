# Familiar Actors — Project Plan

## Goal

A web app where you type an actor's name and get back a list of similar-looking actors. Solves the "who was I thinking of?" problem when you recognize someone but can't place them.

## Approach

Use face embeddings (vector representations of facial features via deepface/ArcFace) to numerically compare actor headshots sourced from TMDB, then rank by cosine similarity.

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
| `deepface` | Face embedding generation (ArcFace model) |
| `numpy` | Embedding storage and cosine similarity |
| `sqlmodel` | Actor metadata in SQLite |
| `pydantic-settings` | Configuration (.env for API keys) |
| `tf-keras` | Required by deepface's TensorFlow backend |

## File Structure

```text
familiar_actors/
    __init__.py
    app.py              # FastAPI app, lifespan, route registration
    cli.py              # CLI: fetch, embed, build, serve
    config.py           # Pydantic Settings (TMDB key, data paths, model config)
    models.py           # SQLModel Actor + Pydantic ActorResult
    database.py         # SQLite engine/session setup
    tmdb.py             # Async TMDB client (fetch actors, download images)
    embeddings.py       # deepface ArcFace embedding generation
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
uv run familiar-actors fetch [num_pages]  # Fetch actors + headshots from TMDB
uv run familiar-actors embed              # Generate face embeddings
uv run familiar-actors build [num_pages]  # fetch + embed in one shot
uv run familiar-actors serve              # Start FastAPI dev server
```

## TMDB Setup

Register at <https://www.themoviedb.org/settings/api> for a free API key. Store in `.env`:

```bash
TMDB_API_KEY=your_key_here
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
- [ ] TMDB API key setup + integration test
- [ ] First real `build` run to populate the dataset
- [ ] End-to-end verification (search UI with real data)

## Scaling Path

Start with ~500 actors (25 pages from TMDB popular endpoint). To scale:

- `fetch` with more pages → more actors
- `embed` processes only new headshots (skips existing)
- numpy cosine similarity handles 10-50k actors fine in memory
- Beyond that: swap `similarity.py` internals to FAISS (same interface, faster search)

## Future Ideas

- Search by uploading a photo (not just by name)
- Filter by gender, age range, ethnicity
- Show what movies/shows the similar actors are known for
- "Explain" why two actors look similar (shared facial features)
