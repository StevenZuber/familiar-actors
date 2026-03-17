# Familiar Actors

Ever watch a show, think "that actor looks familiar," look them up, and realize it's not who you thought? Then you're stuck wondering — who *was* I thinking of?

Familiar Actors solves this. Type in an actor's name and get back a list of actors who look similar to them, ranked by facial similarity.

## How it works

1. Actor headshots are sourced from [TMDB](https://www.themoviedb.org/)
2. Each headshot is processed through [deepface](https://github.com/serengil/deepface) (ArcFace model) to generate a face embedding — a numerical vector representing facial features
3. When you search for an actor, their embedding is compared against all others using cosine similarity
4. The most similar faces are returned, ranked by score

## Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
# Install dependencies
uv sync

# Get a free TMDB Read Access Token: https://www.themoviedb.org/settings/api
# Add it to .env
echo "TMDB_READ_ACCESS_TOKEN=your_token_here" > .env
```

## Usage

### Build the dataset

```bash
# Fetch actors from TMDB, download headshots, generate embeddings
uv run familiar-actors build

# Or run steps individually:
uv run familiar-actors fetch        # Fetch actors + download headshots (default: 25 pages / ~500 actors)
uv run familiar-actors fetch 50     # Fetch more pages for a larger dataset
uv run familiar-actors embed        # Generate face embeddings for downloaded headshots
```

### Start the app

```bash
uv run familiar-actors serve
```

Open <http://127.0.0.1:8000> and start searching.

## Tech stack

- **FastAPI** + **Jinja2** + **HTMX** — backend and server-rendered UI
- **deepface** (ArcFace) — face embedding generation
- **SQLModel** / SQLite — actor metadata
- **numpy** — embedding storage and cosine similarity search
- **httpx** — async TMDB API client

## Tests

```bash
uv run pytest tests/ -v
```
