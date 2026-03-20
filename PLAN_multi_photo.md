# Multi-Photo Averaged Embeddings

## Context

Single-photo CLIP embeddings are sensitive to the specific photo used — bad angles, lighting, or styling can distort an actor's representation. TMDB's `/person/{id}/images` endpoint returns multiple portrait photos per actor (up to 27 for famous actors). Averaging CLIP embeddings from multiple photos produces a more stable, representative vector.

## Approach

New CLI command `fetch-images` that:

1. Fetches up to 5 photos per actor (top 5 by vote_average) from TMDB
2. Downloads them locally at w500 resolution
3. Generates CLIP embeddings for each in memory
4. Averages and L2-normalizes into a single embedding per actor
5. Saves only the averaged embedding to disk
6. Similarity index prefers averaged embeddings, falls back to single-photo

## Storage impact

- Averaged embeddings: ~2KB each, ~130MB for 65k actors (same size as current single-photo embeddings)
- Downloaded photos: local only, not deployed. ~5 photos × ~30KB × 65k actors = ~10GB locally
- Railway deployment stays under 5GB (display headshots + averaged embeddings + SQLite DB)

## Files to modify

### `config.py` — add new paths and settings

- `headshots_multi_dir: Path = Path("data/headshots_multi")`
- `embeddings_avg_dir: Path = Path("data/embeddings_avg")`
- `multi_image_size: str = "w500"`
- `min_image_width: int = 500`
- `max_photos_per_actor: int = 5`

### `models.py` — add field

- `clip_avg_embedding_path: str | None = None` on Actor

### `database.py` — add migration

- Same pattern as existing `clip_embedding_path` migration, add `clip_avg_embedding_path` column

### `tmdb.py` — add two methods to TMDBClient

- `fetch_person_images(tmdb_id, client)` — hits `/person/{id}/images`, returns profiles list
- `download_multi_headshots(session)` — for each actor without `clip_avg_embedding_path`:
  - Fetch image list, filter by `width >= min_image_width`, take top 5 by `vote_average`
  - Download to `data/headshots_multi/{tmdb_id}/{index}.jpg`
  - Log progress

### `embeddings.py` — add averaging function

- `process_multi_photo_embeddings(session)` — for each actor with photos in `headshots_multi/{tmdb_id}/`:
  - Generate CLIP embedding for each photo (in memory, not saved individually)
  - L2-normalize each, compute mean, L2-normalize the result
  - Save averaged embedding to `data/embeddings_avg/{tmdb_id}.npy`
  - Update `actor.clip_avg_embedding_path`

### `similarity.py` — prefer averaged embeddings

- Load query: `WHERE clip_avg_embedding_path IS NOT NULL OR clip_embedding_path IS NOT NULL`
- Loading loop: `embedding_path = actor.clip_avg_embedding_path or actor.clip_embedding_path`
- Fully backwards compatible — actors without averaged embeddings still work

### `routes/search.py` — update cast page check

- `in_database` check includes `clip_avg_embedding_path`

### `cli.py` — new command

- `fetch-images [batch_size]` — orchestrates download + averaging
- Calls `download_multi_headshots()` then `process_multi_photo_embeddings()`

### Documentation

- Update README.md CLI commands table
- Update PLAN.md progress checklist

## Averaging math

```python
normalized = [emb / np.linalg.norm(emb) for emb in embeddings]
avg = np.mean(normalized, axis=0)
avg = avg / np.linalg.norm(avg)
```

Normalize before averaging so each photo contributes equally. Re-normalize after so the result is on the unit sphere for cosine similarity.

## Verification

1. `uv run pytest tests/ -v` — all existing tests pass
2. `uv run familiar-actors fetch-images 10` — process 10 actors, verify .npy files in `data/embeddings_avg/`
3. `uv run familiar-actors serve` — search and confirm results use averaged embeddings
4. Check that actors with only single-photo embeddings still appear in results
5. New tests: averaging math, fallback behavior, skip-if-processed
