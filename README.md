# Familiar Actors

Ever watch a show, think "that actor looks familiar," look them up, and realize it's not who you thought? Then you're stuck wondering — who *was* I thinking of?

Familiar Actors solves this. Type in an actor's name and get back a list of actors who look similar to them, ranked by facial similarity.

## How it works

1. Actor headshots are sourced from [TMDB](https://www.themoviedb.org/)
2. Each headshot is processed through [OpenCLIP](https://github.com/mlfoundations/open_clip) (ViT-B-32) to generate an embedding — a numerical vector representing visual features
3. When you search for an actor, their embedding is compared against all others using cosine similarity
4. The most similar faces are returned, ranked by score

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager (`brew install uv` or see [install docs](https://docs.astral.sh/uv/getting-started/installation/))
- **A free [TMDB](https://www.themoviedb.org/) account** — for the API token used to fetch actor data

SQLite is included in Python's standard library. All other dependencies are installed automatically by `uv sync`.

## Setup

```bash
# Install dependencies
uv sync

# Get a free TMDB Read Access Token: https://www.themoviedb.org/settings/api
# Add it to .env
echo "TMDB_READ_ACCESS_TOKEN=your_token_here" > .env
```

## Usage

### Build the dataset

The more actors in the dataset, the better the results. All commands are incremental — re-running them skips actors, headshots, and embeddings that already exist.

**Quick start** (popular actors only, ~500 people):

```bash
uv run familiar-actors build
```

**Full pipeline** Individual steps:

```bash
uv run familiar-actors fetch 500            # Popular actors (max ~10k)
uv run familiar-actors fetch-credits 500    # Cast from top-rated movies
uv run familiar-actors fetch-credits 500 tv # Cast from top-rated TV shows
uv run familiar-actors embed                # Generate CLIP embeddings for all headshots
```

**Huge Dataset Run** If you want to let this run for the best results — set it and forget it for a few hours.

```bash
uv run familiar-actors fetch 500 && uv run familiar-actors fetch-credits 500 && uv run familiar-actors fetch-credits 500 tv && uv run familiar-actors embed
```

**Individual commands:**

| Command | What it does |
| ------- | ----------- |
| `fetch [num_pages]` | Fetch popular actors + download headshots |
| `fetch-credits [num_pages]` | Crawl cast lists from top-rated movies |
| `fetch-credits [num_pages] tv` | Crawl cast lists from top-rated TV shows |
| `embed` | Generate CLIP embeddings for all unprocessed headshots |
| `build [num_pages]` | Shortcut for `fetch` + `embed` |

### Start the app

```bash
uv run familiar-actors serve
```

Open <http://127.0.0.1:8000> and start searching.

## Tech stack

- **FastAPI** + **Jinja2** + **HTMX** — backend and server-rendered UI
- **OpenCLIP** (ViT-B-32) — visual similarity embeddings
- **SQLModel** / SQLite — actor metadata
- **numpy** — embedding storage and cosine similarity search
- **httpx** — async TMDB API client

## Tests

```bash
uv run pytest tests/ -v
```

## Poking around the database

Actor metadata is stored in a local SQLite database at `data/familiar_actors.db`. You can query it directly:

```bash
sqlite3 data/familiar_actors.db
```

```sql
-- How many actors are searchable
SELECT COUNT(*) FROM actor WHERE clip_embedding_path IS NOT NULL;

-- Look up an actor
SELECT id, name, tmdb_id FROM actor WHERE name LIKE '%Pitt%';

-- How many headshots failed embedding generation
SELECT COUNT(*) FROM actor WHERE image_path IS NOT NULL AND clip_embedding_path IS NULL;

-- Schema
.schema actor
```

## Upcoming Features

### Quality of life

- Show what they're known for — under each result, show 1-2 movie/TV titles. Helps answer "wait, where do I know them from?"

### Search improvements

- Search by photo upload — this is the killer feature. Skip the name entirely, upload a screenshot from whatever you're watching, and find similar actors directly. CLIP makes this almost free since it already embeds images — you'd just embed the uploaded photo and search the same index.
- "No good matches?" button — you mentioned this earlier. Could trigger a targeted crawl of movies the searched actor appeared in, expanding the pool in the exact direction that matters.

### Data quality

- Multiple photos per actor — some TMDB profile photos are bad (weird angles, sunglasses, black and white). Averaging embeddings from 2-3 photos per actor would produce more stable similarity scores. TMDB's /person/{id}/images endpoint returns multiple photos.
- Filter out non-actors — TMDB's popular people includes directors, producers, etc. We could filter by known_for_department == "Acting" during fetch.

### Performance

- Batch cast DB lookups — the cast page does one query per cast member (N+1). Fine for SQLite locally, but should batch into a single `WHERE tmdb_id IN (...)` query before deploying with a remote DB.
- Singleton TMDB client — currently creates a new client per request. If we add rate limiting or connection pooling, switch to a shared instance.

### Pre-Deployment Considerations

- Pre-build the dataset and ship it — don't make users run the pipeline. Bundle the SQLite DB and embeddings so the app works out of the box. Once deployed, be sure to make it easy to update the dataset.
- Cache TMDB images locally — right now we hotlink TMDB's CDN for display. Fine for personal use, but for a deployed app you'd want to serve the headshots yourself or at least proxy them.
