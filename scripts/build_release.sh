#!/usr/bin/env bash
# Build the deployment data tarball for a GitHub release.
#
# Packages everything the deployed app needs at boot:
#   - familiar_actors.db        (actor metadata)
#   - embeddings_index.npy      (consolidated embedding matrix)
#   - embeddings_ids.json       (row -> actor id mapping)
#   - clip_image_encoder.onnx   (ONNX encoder for /upload photo search)
#
# Files are stored flat (no directory prefix) because the app extracts
# the tarball directly into settings.data_dir.
#
# Run scripts/consolidate_index.py first if embeddings changed, and
# scripts/export_onnx.py if the CLIP model config changed.
#
# Usage: scripts/build_release.sh [output-path]   (default /tmp/data.tar.gz)

set -euo pipefail
cd "$(dirname "$0")/.."

files=(
    familiar_actors.db
    embeddings_index.npy
    embeddings_ids.json
    clip_image_encoder.onnx
)

for f in "${files[@]}"; do
    if [[ ! -f "data/$f" ]]; then
        echo "Missing data/$f — cannot build release tarball." >&2
        exit 1
    fi
done

out="${1:-/tmp/data.tar.gz}"

# COPYFILE_DISABLE / --no-xattrs keep macOS from adding ._* AppleDouble
# entries, which would litter the server's data dir on extraction.
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$out" -C data "${files[@]}"

echo "Built $out:"
ls -lh "$out"
tar -tzf "$out"
