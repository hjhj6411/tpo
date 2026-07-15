#!/usr/bin/env bash
set -euo pipefail

# Split option plans across the requested independent single-GPU collectors, then merge
# their top-1 benchmark crops and JSONL logs. FSigLIP and VLM services must
# already be running; this launcher does not create or modify environments.

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

CONDA_BASE="${CONDA_BASE:-/home1/hjhj6411/miniconda3}"
PLAN_PATH="${PLAN_PATH:-data/options/option_plans.jsonl}"
OUT_ROOT="${OUT_ROOT:-data/retrieval/sam3vlmfix_benchmark_all_4gpu}"

GPUS="${GPUS:-0,1,2,3}"
FSIGLIP_HOST="${FSIGLIP_HOST:-127.0.0.1}"
# One port is shared by every worker. Alternatively provide one port per GPU,
# for example FSIGLIP_PORTS=1235,1236,1237,1238.
FSIGLIP_PORTS="${FSIGLIP_PORTS:-1235}"
VLM_URL="${VLM_URL:-http://127.0.0.1:8001/v1/chat/completions}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"

OFFSET="${OFFSET:-0}"
TOTAL_LIMIT="${TOTAL_LIMIT:-0}"  # 0 means every plan after OFFSET.
OPTIONS="${OPTIONS:-A,B,C,D}"
QUERY_MODE="${QUERY_MODE:-original}"
INDEX_NAME="${INDEX_NAME:-pod_fashion}"
RETRIEVAL_K="${RETRIEVAL_K:-50}"
SCORE_TOP_N="${SCORE_TOP_N:-20}"
PATTERN_SHORT_SIDE_TILES="${PATTERN_SHORT_SIDE_TILES:-5}"
COLOR_SHORT_SIDE_TILES="${COLOR_SHORT_SIDE_TILES:-15}"
PATCH_MASK_FILL="${PATCH_MASK_FILL:-local_mean}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
PRINT_RANKING_N="${PRINT_RANKING_N:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -f "$PLAN_PATH" ]]; then
  echo "[error] plan file not found: $PLAN_PATH" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
IFS=',' read -r -a PORT_ARR <<< "$FSIGLIP_PORTS"
NUM_WORKERS="${#GPU_ARR[@]}"

if [[ "$NUM_WORKERS" -lt 1 ]]; then
  echo "[error] provide at least one GPU id via GPUS" >&2
  exit 1
fi
if [[ "${#PORT_ARR[@]}" -eq 1 ]]; then
  shared_port="${PORT_ARR[0]}"
  PORT_ARR=()
  for _ in "${GPU_ARR[@]}"; do
    PORT_ARR+=("$shared_port")
  done
elif [[ "${#PORT_ARR[@]}" -ne "$NUM_WORKERS" ]]; then
  echo "[error] FSIGLIP_PORTS must contain one shared port or one port per GPU" >&2
  exit 1
fi

PLAN_COUNT="$(awk 'NF {count++} END {print count+0}' "$PLAN_PATH")"
if (( OFFSET < 0 || OFFSET >= PLAN_COUNT )); then
  echo "[error] OFFSET=$OFFSET is outside plan count $PLAN_COUNT" >&2
  exit 1
fi

AVAILABLE=$((PLAN_COUNT - OFFSET))
if (( TOTAL_LIMIT <= 0 || TOTAL_LIMIT > AVAILABLE )); then
  NUM_PLANS="$AVAILABLE"
else
  NUM_PLANS="$TOTAL_LIMIT"
fi
CHUNK=$(((NUM_PLANS + NUM_WORKERS - 1) / NUM_WORKERS))

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry run] plans=$NUM_PLANS/$PLAN_COUNT offset=$OFFSET chunk=$CHUNK"
  for i in "${!GPU_ARR[@]}"; do
    shard_start=$((i * CHUNK))
    remain=$((NUM_PLANS - shard_start))
    if (( remain <= 0 )); then
      continue
    fi
    shard_limit="$CHUNK"
    if (( remain < CHUNK )); then
      shard_limit="$remain"
    fi
    echo "[worker $i] gpu=${GPU_ARR[$i]} port=${PORT_ARR[$i]} offset=$((OFFSET + shard_start)) limit=$shard_limit"
  done
  exit 0
fi

if [[ -d "$OUT_ROOT" ]] && [[ -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$OUT_ROOT"
  else
    echo "[error] output directory is not empty: $OUT_ROOT" >&2
    echo "        Use a new OUT_ROOT or set FORCE=1 to replace it." >&2
    exit 1
  fi
fi

LOG_DIR="$OUT_ROOT/_logs"
SHARD_DIR="$OUT_ROOT/_shards"
mkdir -p "$LOG_DIR" "$SHARD_DIR"

cat > "$OUT_ROOT/launcher_config.txt" <<EOF
plan_path=$PLAN_PATH
plan_count=$PLAN_COUNT
offset=$OFFSET
num_plans=$NUM_PLANS
gpus=$GPUS
fsiglip_host=$FSIGLIP_HOST
fsiglip_ports=$FSIGLIP_PORTS
vlm_url=$VLM_URL
vlm_model=$VLM_MODEL
options=$OPTIONS
query_mode=$QUERY_MODE
retrieval_k=$RETRIEVAL_K
score_top_n=$SCORE_TOP_N
pattern_short_side_tiles=$PATTERN_SHORT_SIDE_TILES
color_short_side_tiles=$COLOR_SHORT_SIDE_TILES
patch_mask_fill=$PATCH_MASK_FILL
EOF

echo "================================================================"
echo "$NUM_WORKERS-GPU benchmark crop collection"
echo "================================================================"
echo "plans:          $NUM_PLANS / $PLAN_COUNT (offset=$OFFSET)"
echo "gpus:           $GPUS"
echo "fsiglip ports:  ${PORT_ARR[*]}"
echo "vlm:            $VLM_URL"
echo "output:         $OUT_ROOT"
echo "plans/worker:   up to $CHUNK"
echo "================================================================"

WORKER_PIDS=()
cleanup() {
  for pid in "${WORKER_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for i in "${!GPU_ARR[@]}"; do
  shard_start=$((i * CHUNK))
  remain=$((NUM_PLANS - shard_start))
  if (( remain <= 0 )); then
    continue
  fi

  shard_limit="$CHUNK"
  if (( remain < CHUNK )); then
    shard_limit="$remain"
  fi

  gpu="${GPU_ARR[$i]}"
  port="${PORT_ARR[$i]}"
  shard_offset=$((OFFSET + shard_start))
  shard_root="$SHARD_DIR/shard_${i}_gpu${gpu}_offset${shard_offset}_limit${shard_limit}"
  log="$LOG_DIR/worker_${i}_gpu${gpu}_offset${shard_offset}_limit${shard_limit}.log"

  echo "[worker $i] gpu=$gpu port=$port offset=$shard_offset limit=$shard_limit"
  # shellcheck disable=SC2206
  EXTRA_ARRAY=( $EXTRA_ARGS )

  (
    source "$CONDA_BASE/bin/activate"
    conda activate sam3
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/external/sam3:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="$gpu"

    python fsiglip/collect_topk_sam3_fsiglip_patch_rank_vlm_garment_axis_patches_benchmark_crop.py \
      --plan-path "$PLAN_PATH" \
      --client-url "http://${FSIGLIP_HOST}:${port}/knn-service" \
      --score-url "http://${FSIGLIP_HOST}:${port}/score-image-files" \
      --vlm-url "$VLM_URL" \
      --vlm-model "$VLM_MODEL" \
      --index-name "$INDEX_NAME" \
      --output-root "$shard_root" \
      --query-mode "$QUERY_MODE" \
      --retrieval-k "$RETRIEVAL_K" \
      --score-top-n "$SCORE_TOP_N" \
      --offset "$shard_offset" \
      --limit "$shard_limit" \
      --options "$OPTIONS" \
      --device cuda \
      --pattern-short-side-tiles "$PATTERN_SHORT_SIDE_TILES" \
      --color-short-side-tiles "$COLOR_SHORT_SIDE_TILES" \
      --patch-mask-fill "$PATCH_MASK_FILL" \
      --print-ranking-n "$PRINT_RANKING_N" \
      --progress-every "$PROGRESS_EVERY" \
      "${EXTRA_ARRAY[@]}"
  ) > "$log" 2>&1 &

  WORKER_PIDS+=("$!")
done

echo "Waiting for ${#WORKER_PIDS[@]} workers..."
FAIL=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done
trap - INT TERM

if [[ "$FAIL" -ne 0 ]]; then
  echo "[error] one or more workers failed; shard outputs were kept for diagnosis" >&2
  echo "        logs: $LOG_DIR" >&2
  exit 1
fi

echo "Merging crops and logs..."
python - "$OUT_ROOT" <<'PY'
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
shards = sorted((root / "_shards").glob("shard_*"))
crop_root = root / "benchmark_crops"
crop_root.mkdir(parents=True, exist_ok=True)

num_crops = 0
for shard in shards:
    shard_crops = shard / "benchmark_crops"
    if shard_crops.exists():
        for src in sorted(shard_crops.glob("*/*")):
            rel = src.relative_to(shard_crops)
            dst = crop_root / rel
            if dst.exists():
                raise SystemExit(f"duplicate benchmark crop: {dst}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            num_crops += 1
        shutil.rmtree(shard_crops)

for log_name in ("raw_topk_results.jsonl", "scored_results.jsonl"):
    dst = root / log_name
    with dst.open("w", encoding="utf-8") as out:
        for shard in shards:
            src = shard / log_name
            if not src.exists():
                continue
            for line in src.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                if log_name == "scored_results.jsonl":
                    row = json.loads(line)
                    old_crop = row.get("benchmark_crop_path")
                    if old_crop:
                        old_crop = Path(old_crop)
                        row["benchmark_crop_path"] = str(
                            crop_root / old_crop.parent.name / old_crop.name
                        )
                    line = json.dumps(row, ensure_ascii=False, default=str)
                out.write(line + "\n")

print(f"merged {num_crops} crops from {len(shards)} shards into {crop_root}")
PY

echo
echo "Done."
echo "Crops:  $OUT_ROOT/benchmark_crops"
echo "Scores: $OUT_ROOT/scored_results.jsonl"
echo "Logs:   $LOG_DIR"
