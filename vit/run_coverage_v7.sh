#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

PLAN_PATH="${PLAN_PATH:-data/options/option_plans.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-data/images_vit_cov_c}"
OUTPUT="${OUTPUT:-$IMAGE_ROOT/collection_log.jsonl}"
CLIENT_URL="${CLIENT_URL:-http://127.0.0.1:1234/knn-service}"
COVERAGE_URL="${COVERAGE_URL:-http://127.0.0.1:1235/patch-coverage}"
COLOR_COVERAGE_URL="${COLOR_COVERAGE_URL:-http://127.0.0.1:1235/patch-color-coverage}"
VLM_URLS="${VLM_URLS:-}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
LIMIT="${LIMIT:-0}"
TOP_K="${TOP_K:-12}"
GRID="${GRID:-5}"
WORKERS="${WORKERS:-4}"
COLOR_COV="${COLOR_COV:-color}"
FORCE="${FORCE:-0}"

ARGS=(
  --plan_path "$PLAN_PATH"
  --client-url "$CLIENT_URL"
  --coverage-url "$COVERAGE_URL"
  --color-coverage-url "$COLOR_COVERAGE_URL"
  --vlm-urls "$VLM_URLS"
  --vlm-model "$VLM_MODEL"
  --image_root "$IMAGE_ROOT"
  --output "$OUTPUT"
  --limit "$LIMIT"
  --top_k "$TOP_K"
  --grid "$GRID"
  --workers "$WORKERS"
  --color-cov "$COLOR_COV"
)

if [[ "$FORCE" == "1" ]]; then
  ARGS+=(--force)
fi

exec python vit/collect_images_vit_coverage_v7.py "${ARGS[@]}" "$@"

