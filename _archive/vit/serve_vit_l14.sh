#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
WORK="${WORK:-$REPO_ROOT/data/clip_corpus}"
PORT="${PORT:-1234}"
MODEL="${MODEL:-ViT-L/14}"
INDICES_PATHS="${INDICES_PATHS:-$WORK/indices_paths.json}"

if [[ ! -f "$INDICES_PATHS" ]]; then
  echo "[error] missing clip-retrieval configuration: $INDICES_PATHS" >&2
  exit 1
fi
if [[ ! -d "$WORK/index" || ! -d "$WORK/embeddings/metadata" ]]; then
  echo "[error] incomplete ViT corpus under: $WORK" >&2
  exit 1
fi

echo "Starting $MODEL backend on port $PORT"
echo "indices: $INDICES_PATHS"
exec clip-retrieval back \
  --port "$PORT" \
  --indices-paths "$INDICES_PATHS" \
  --enable_faiss_memory_mapping True \
  --clip_model "$MODEL"
