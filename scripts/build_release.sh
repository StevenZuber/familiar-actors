#!/usr/bin/env bash
# Build the deployment data tarball for a GitHub release.
#
# Packages everything the deployed app needs at boot, for one embedding space:
#   - familiar_actors.db        (actor metadata)
#   - <space> consolidated index + ids  (embeddings_index.npy/embeddings_ids.json
#                                         for clip; facecrop_index.npy/
#                                         facecrop_ids.json for facecrop)
#   - clip_image_encoder.onnx   (ONNX CLIP encoder for /upload photo search;
#                                 used by BOTH spaces — the facecrop query path
#                                 runs CLIP on the cropped face)
#
# The YuNet face detector ships in the repo (familiar_actors/models/), not the
# tarball. Only the live space's index is included (each is ~800MB; shipping
# both would bloat the tarball), so build with the space you'll deploy.
#
# Files are stored flat — the app extracts the tarball directly into data/.
# Run scripts/consolidate_index.py --space <space> first if embeddings changed,
# and scripts/export_onnx.py if the CLIP model config changed.
#
# Usage: scripts/build_release.sh [space] [output-path]
#        space defaults to $EMBEDDING_SPACE or "clip"; output to /tmp/data.tar.gz

set -euo pipefail
cd "$(dirname "$0")/.."

space="${1:-${EMBEDDING_SPACE:-clip}}"
out="${2:-/tmp/data.tar.gz}"

# Mirror settings.consolidated_index_paths() naming: clip keeps legacy names.
if [[ "$space" == "clip" ]]; then
    stem="embeddings"
else
    stem="$space"
fi

files=(
    familiar_actors.db
    "${stem}_index.npy"
    "${stem}_ids.json"
    clip_image_encoder.onnx
)

echo "Building release tarball for embedding space: $space"
for f in "${files[@]}"; do
    if [[ ! -f "data/$f" ]]; then
        echo "Missing data/$f — cannot build release tarball." >&2
        echo "(Did you run: uv run python scripts/consolidate_index.py --space $space ?)" >&2
        exit 1
    fi
done

# COPYFILE_DISABLE / --no-xattrs keep macOS from adding ._* AppleDouble
# entries, which would litter the server's data dir on extraction.
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$out" -C data "${files[@]}"

echo "Built $out:"
ls -lh "$out"
tar -tzf "$out"
