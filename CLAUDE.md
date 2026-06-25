# CLAUDE.md

Familiar Actors — a face-similarity search demo. Type an actor's name, get back actors who look similar by CLIP visual embeddings of their TMDB headshots.

## Quick map

- [familiar_actors/cli.py](familiar_actors/cli.py) — entry point, dispatches subcommands (`fetch`, `fetch-credits`, `fetch-images`, `embed`, `build`, `serve`).
- [familiar_actors/tmdb.py](familiar_actors/tmdb.py) — async TMDB API client. Handles popular-actor pagination, credit crawling, multi-photo downloads.
- [familiar_actors/embeddings.py](familiar_actors/embeddings.py) — OpenCLIP (ViT-B-32) embedding generation. Lazy-loads the model; `pipeline` dependency group required.
- [familiar_actors/query_embedding.py](familiar_actors/query_embedding.py) — request-time embedding of uploaded photos via an ONNX export of the same encoder (onnxruntime, no torch). Preprocessing is replicated in PIL/numpy and must stay in sync with open_clip's transforms.
- [familiar_actors/facecrop_embeddings.py](familiar_actors/facecrop_embeddings.py) — pipeline-side face-crop CLIP embeddings: detect (insightface buffalo_l/SCRFD) → crop → CLIP. Needs the `pipeline` + `face` dependency groups.
- [familiar_actors/face_detect.py](familiar_actors/face_detect.py) — request-time face detection for uploads via OpenCV YuNet (tiny ONNX in `familiar_actors/models/`, opencv-headless only — no torch/insightface in prod).
- [familiar_actors/face_crop.py](familiar_actors/face_crop.py) — shared crop geometry (pad + clamp) used by both the pipeline and serving detectors so crops are identical.
- [familiar_actors/similarity.py](familiar_actors/similarity.py) — in-memory cosine-similarity index. Loaded once at app startup; loads whichever space `settings.embedding_space` selects.
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
- `embeddings_facecrop/{tmdb_id}.npy` — CLIP embeddings of the detected+cropped face (the `facecrop` space; averaged over multi-photos, else the headshot).
- `embeddings_index.npy` + `embeddings_ids.json` — consolidated `clip` index for deployment.
- `facecrop_index.npy` + `facecrop_ids.json` — consolidated `facecrop` index for deployment.
- `clip_image_encoder.onnx` — ONNX export of the ViT-B-32 image encoder (~335MB), used by `/upload` photo search. Built by `scripts/export_onnx.py`; must ship in the release tarball.
- `.data_size` — Content-Length of the last-downloaded release tarball; drives stale-dataset detection.

- `clip_image_encoder.onnx` — ONNX CLIP image encoder; used by `/upload` in BOTH spaces (the facecrop query path runs it on the cropped face).
- `familiar_actors/models/face_detection_yunet.onnx` — YuNet detector (in the repo, not `data/`); used at request time to crop uploaded photos.

The `Actor` row points at files via `image_path`, `clip_embedding_path`, `clip_avg_embedding_path`, and `facecrop_embedding_path`. Within the `clip` space `clip_avg_*` takes precedence over `clip_*`. `settings.embedding_space` ("clip" | "facecrop", env `EMBEDDING_SPACE`) selects which space the index and upload path use; the two coexist for A/B + rollback. `face_unavailable` flags actors with no detectable face (skipped by the facecrop pipeline). `gender` ("M"/"F") and `age` are estimates from buffalo_l's genderage model, computed during the facecrop pass — intended as optional search filters, not ground truth.

## Pipeline (build the dataset)

All steps are incremental — re-running skips already-processed work. Safe to interrupt at any point.

1. `fetch [pages]` — TMDB `/person/popular` pagination, inserts new actors, downloads w185 headshots.
2. `fetch-credits [pages] [tv]` — crawls cast lists from top-rated movies (or TV). Best source for obscure character actors.
3. `fetch-images` — calls TMDB `/person/{id}/images`, filters by `min_image_width`, downloads top-N by `vote_average`. Then generates averaged embeddings. **Long-running**: hours for a real dataset.
4. `embed` — single-photo CLIP embeddings for any actor that has a headshot but no embedding.
5. `build [pages]` — shortcut for `fetch` + `embed` (not the full pipeline).
6. `embed-facecrop [limit]` — face-crop CLIP embeddings (detect+crop+CLIP) for the `facecrop` space, and gender/age estimates (buffalo_l genderage) in the same detection pass. Needs `uv sync --group pipeline --group face`. **Long-running** over a full dataset; resumable, commits per actor. Targets actors missing a facecrop embedding **or** a gender estimate, so a re-run backfills gender/age onto already-embedded actors without recomputing their embedding. Pass a `limit` to process a small batch.

Rate limit: 40 req/10s on TMDB. `download_multi_headshots` sleeps 0.25s between actors to stay polite.

## Serving

- `uv run familiar-actors serve` runs uvicorn with reload.
- On startup, `lifespan` calls `_download_data_if_needed()` — if `DATA_RELEASE_URL` is set and either the index file is missing or Content-Length differs from `.data_size`, the dataset tarball is downloaded and extracted.
- Indices (`SimilarityIndex`, `ActorSearchIndex`) load once into memory and live on the app object.
- Search index tries individual `.npy` files first (dev); falls back to the consolidated index (Railway/deployed).
- Templates use Jinja2 + HTMX. `is_htmx_request()` decides partial vs full page.
- `POST /upload` matches a user photo against the index: photos are embedded in memory with onnxruntime (never written to disk) and searched via `SimilarityIndex.search_by_vector`. The client (`search.js`) downscales/re-encodes to JPEG before upload — iPhone HEIC and EXIF rotation get normalized in the browser, with `pillow-heif` + `exif_transpose` as the server-side backstop.
- In the `facecrop` space, `/upload` first detects+crops the face (YuNet) before CLIP; no detectable face returns a friendly 422. In the `clip` space it embeds the whole image. The detector/crop differ from the pipeline's (YuNet vs buffalo_l) but produce near-identical CLIP embeddings (verified crop-parity ~0.99), since CLIP is framing-robust.
- `/search` takes a `limit` (capped at `MAX_RESULTS`) powering the "Show more" button: each click re-requests with a larger limit and re-renders the re-ranked list (`results.html`). `SimilarityIndex.search_by_vector` drops results within `settings.dedup_similarity_threshold` (0.98) cosine of one already shown — collapsing duplicate TMDB person entries, well above real lookalike scores (~0.94). Upload-path pagination isn't wired yet (would need the query vector persisted).

## Conventions

- **Python 3.12+, async httpx, SQLModel.**
- `# type: ignore[union-attr]` comments on SQLModel columns are intentional — SQLAlchemy column expressions confuse mypy with the SQLModel optional-typing pattern. Keep them.
- Migrations are hand-rolled in [database.py](familiar_actors/database.py) — add a column to `Actor`, then add an `ALTER TABLE` block guarded by `if "col" not in existing_columns`.
- Pipeline deps (`open-clip-torch`) are in the `pipeline` dependency group, not default. `embed` and `fetch-images` need `uv sync --group pipeline`; the web app does not. `embed-facecrop` also needs the `face` group (insightface, for buffalo_l detection). The web app's only face dep is `opencv-python-headless` (main deps), for YuNet.
- Tests are in `tests/`, run with `uv run pytest tests/ -v`.

## Long-running CLI commands

`fetch-images`, `fetch-credits`, and the consolidated `build && fetch-credits && ...` chain can run for hours. They make tens of thousands of TMDB requests. If a single request returns a malformed payload, anything not caught will kill the whole run — guard parsing inside the per-actor loop, not just at the outer level.

## Deployment

- Railway, single service, `railway.json` present.
- Production dataset ships as a GitHub release tarball; `DATA_RELEASE_URL` env var triggers download-on-boot.
- Static dataset means a redeploy is needed to refresh actors. Stale-detection is by Content-Length, not hash.
- The release tarball must include `clip_image_encoder.onnx` (alongside the DB and consolidated index) or `/upload` returns a 503 partial.
- Build the tarball with [scripts/build_release.sh](scripts/build_release.sh) — it packages the four required files flat (the app extracts directly into `data/`) and fails loudly if any are missing. Upload the result as `data.tar.gz` on a `data-vN` GitHub release.
- The tarball ships only the **live space's** index (each is ~800MB). For `facecrop`: `scripts/build_release.sh facecrop` packages `facecrop_index.npy` + `facecrop_ids.json` (+ DB + CLIP onnx). The YuNet detector ships in the repo, not the tarball.
- **Switching to face-crop search (runbook):** (1) `uv sync --group pipeline --group face`; (2) `caffeinate -i uv run familiar-actors embed-facecrop` (hours, resumable); (3) `uv run python scripts/consolidate_index.py --space facecrop`; (4) `scripts/build_release.sh facecrop`; (5) upload as `data-v4`; (6) on Railway set `EMBEDDING_SPACE=facecrop` and point `DATA_RELEASE_URL` at data-v4. Roll back by reverting both env vars to clip/data-v3.
