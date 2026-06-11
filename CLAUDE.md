# CLAUDE.md

Familiar Actors — a face-similarity search demo. Type an actor's name, get back actors who look similar by CLIP visual embeddings of their TMDB headshots.

## Quick map

- [familiar_actors/cli.py](familiar_actors/cli.py) — entry point, dispatches subcommands (`fetch`, `fetch-credits`, `fetch-images`, `embed`, `build`, `serve`).
- [familiar_actors/tmdb.py](familiar_actors/tmdb.py) — async TMDB API client. Handles popular-actor pagination, credit crawling, multi-photo downloads.
- [familiar_actors/embeddings.py](familiar_actors/embeddings.py) — OpenCLIP (ViT-B-32) embedding generation. Lazy-loads the model; `pipeline` dependency group required.
- [familiar_actors/query_embedding.py](familiar_actors/query_embedding.py) — request-time embedding of uploaded photos via an ONNX export of the same encoder (onnxruntime, no torch). Preprocessing is replicated in PIL/numpy and must stay in sync with open_clip's transforms.
- [familiar_actors/similarity.py](familiar_actors/similarity.py) — in-memory cosine-similarity index. Loaded once at app startup.
- [familiar_actors/actor_search.py](familiar_actors/actor_search.py) — in-memory name index (prefix → rapidfuzz fallback).
- [familiar_actors/app.py](familiar_actors/app.py) — FastAPI app, lifespan hooks, dataset bootstrap from a GitHub release.
- [familiar_actors/routes/search.py](familiar_actors/routes/search.py) — all HTTP routes (home, /search, /cast, /api/*).
- [familiar_actors/models.py](familiar_actors/models.py) — `Actor` SQLModel (one row per person).
- [familiar_actors/database.py](familiar_actors/database.py) — SQLite engine + hand-rolled column-add migrations.
- [familiar_actors/config.py](familiar_actors/config.py) — pydantic-settings, `.env`-driven config.
- [scripts/crawl.py](scripts/crawl.py) — long-running discovery crawler (separate from CLI commands).
- [scripts/consolidate_index.py](scripts/consolidate_index.py) — builds `embeddings_index.npy` + `embeddings_ids.json` for deployment.
- [scripts/export_onnx.py](scripts/export_onnx.py) — exports the CLIP image encoder to `data/clip_image_encoder.onnx` for photo-upload search, with a torch-vs-ONNX parity check. Re-run if `embedding_model`/`clip_pretrained` ever change.

## Data layout

All under `data/`:

- `familiar_actors.db` — SQLite, one `actor` table.
- `headshots/{tmdb_id}.jpg` — primary w185 thumbnail.
- `headshots_multi/{tmdb_id}/{0..N}.jpg` — extra photos for averaged embeddings.
- `embeddings_clip/{tmdb_id}.npy` — single-photo 512-d embeddings.
- `embeddings_avg/{tmdb_id}.npy` — averaged multi-photo embeddings (preferred when present).
- `embeddings_index.npy` + `embeddings_ids.json` — consolidated index for deployment.
- `clip_image_encoder.onnx` — ONNX export of the ViT-B-32 image encoder (~335MB), used by `/upload` photo search. Built by `scripts/export_onnx.py`; must ship in the release tarball.
- `.data_size` — Content-Length of the last-downloaded release tarball; drives stale-dataset detection.

The `Actor` row points at files via `image_path`, `clip_embedding_path`, `clip_avg_embedding_path`. `clip_avg_*` takes precedence over `clip_*` everywhere.

## Pipeline (build the dataset)

All steps are incremental — re-running skips already-processed work. Safe to interrupt at any point.

1. `fetch [pages]` — TMDB `/person/popular` pagination, inserts new actors, downloads w185 headshots.
2. `fetch-credits [pages] [tv]` — crawls cast lists from top-rated movies (or TV). Best source for obscure character actors.
3. `fetch-images` — calls TMDB `/person/{id}/images`, filters by `min_image_width`, downloads top-N by `vote_average`. Then generates averaged embeddings. **Long-running**: hours for a real dataset.
4. `embed` — single-photo CLIP embeddings for any actor that has a headshot but no embedding.
5. `build [pages]` — shortcut for `fetch` + `embed` (not the full pipeline).

Rate limit: 40 req/10s on TMDB. `download_multi_headshots` sleeps 0.25s between actors to stay polite.

## Serving

- `uv run familiar-actors serve` runs uvicorn with reload.
- On startup, `lifespan` calls `_download_data_if_needed()` — if `DATA_RELEASE_URL` is set and either the index file is missing or Content-Length differs from `.data_size`, the dataset tarball is downloaded and extracted.
- Indices (`SimilarityIndex`, `ActorSearchIndex`) load once into memory and live on the app object.
- Search index tries individual `.npy` files first (dev); falls back to the consolidated index (Railway/deployed).
- Templates use Jinja2 + HTMX. `is_htmx_request()` decides partial vs full page.
- `POST /upload` matches a user photo against the index: photos are embedded in memory with onnxruntime (never written to disk) and searched via `SimilarityIndex.search_by_vector`. The client (`search.js`) downscales/re-encodes to JPEG before upload — iPhone HEIC and EXIF rotation get normalized in the browser, with `pillow-heif` + `exif_transpose` as the server-side backstop.

## Conventions

- **Python 3.12+, async httpx, SQLModel.**
- `# type: ignore[union-attr]` comments on SQLModel columns are intentional — SQLAlchemy column expressions confuse mypy with the SQLModel optional-typing pattern. Keep them.
- Migrations are hand-rolled in [database.py](familiar_actors/database.py) — add a column to `Actor`, then add an `ALTER TABLE` block guarded by `if "col" not in existing_columns`.
- Pipeline deps (`open-clip-torch`) are in the `pipeline` dependency group, not default. `embed` and `fetch-images` need `uv sync --group pipeline`; the web app does not.
- Tests are in `tests/`, run with `uv run pytest tests/ -v`.

## Long-running CLI commands

`fetch-images`, `fetch-credits`, and the consolidated `build && fetch-credits && ...` chain can run for hours. They make tens of thousands of TMDB requests. If a single request returns a malformed payload, anything not caught will kill the whole run — guard parsing inside the per-actor loop, not just at the outer level.

## Deployment

- Railway, single service, `railway.json` present.
- Production dataset ships as a GitHub release tarball; `DATA_RELEASE_URL` env var triggers download-on-boot.
- Static dataset means a redeploy is needed to refresh actors. Stale-detection is by Content-Length, not hash.
- The release tarball must include `clip_image_encoder.onnx` (alongside the DB and consolidated index) or `/upload` returns a 503 partial.
- Build the tarball with [scripts/build_release.sh](scripts/build_release.sh) — it packages the four required files flat (the app extracts directly into `data/`) and fails loudly if any are missing. Upload the result as `data.tar.gz` on a `data-vN` GitHub release.
