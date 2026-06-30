#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

CONDA_BASE="${CONDA_BASE:-/home1/hjhj6411/miniconda3}"
FS_CORPUS="${FS_CORPUS:-/home1/hjhj6411/pod_bench/data/fashionsiglip_corpus}"

GPUS="${GPUS:-0,1,2,3}"
BASE_PORT="${BASE_PORT:-1235}"
START_SERVERS="${START_SERVERS:-1}"

PLAN_PATH="${PLAN_PATH:-data/options/option_plans.jsonl}"
OUT_ROOT="${OUT_ROOT:-data/retrieval/sam3_fsiglip_patch_rank_4gpu}"

OFFSET="${OFFSET:-0}"
TOTAL_LIMIT="${TOTAL_LIMIT:-20}"
OPTIONS="${OPTIONS:-C}"

QUERY_MODE="${QUERY_MODE:-original}"
INDEX_NAME="${INDEX_NAME:-pod_fashion}"

RETRIEVAL_K="${RETRIEVAL_K:-20}"
SCORE_TOP_N="${SCORE_TOP_N:-20}"

SHORT_SIDE_TILES="${SHORT_SIDE_TILES:-8}"
MIN_COVERAGE="${MIN_COVERAGE:-0.20}"
PATTERN_MARGIN="${PATTERN_MARGIN:-0.0}"
COLOR_MARGIN="${COLOR_MARGIN:-0.005}"
COLOR_MODE="${COLOR_MODE:-patch}"

PRINT_RANKING_N="${PRINT_RANKING_N:-5}"
PROGRESS_EVERY="${PROGRESS_EVERY:-5}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_GPUS="${#GPU_ARR[@]}"

if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "No GPUs specified."
  exit 1
fi

CHUNK=$(( (TOTAL_LIMIT + NUM_GPUS - 1) / NUM_GPUS ))

LOG_DIR="$OUT_ROOT/_logs"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "4-GPU SAM3 + FashionSigLIP patch ranking launcher"
echo "================================================================"
echo "repo:          $REPO_ROOT"
echo "gpus:          $GPUS"
echo "base_port:     $BASE_PORT"
echo "start_servers: $START_SERVERS"
echo "plan_path:     $PLAN_PATH"
echo "out_root:      $OUT_ROOT"
echo "offset:        $OFFSET"
echo "total_limit:   $TOTAL_LIMIT"
echo "chunk:         $CHUNK"
echo "options:       $OPTIONS"
echo "retrieval_k:   $RETRIEVAL_K"
echo "score_top_n:   $SCORE_TOP_N"
echo "================================================================"

SERVER_PIDS=()
WORKER_PIDS=()

cleanup() {
  if [[ "$START_SERVERS" == "1" ]]; then
    echo
    echo "[cleanup] stopping FSigLIP servers..."
    for pid in "${SERVER_PIDS[@]:-}"; do
      kill "$pid" 2>/dev/null || true
    done
  fi
}
trap cleanup EXIT

wait_health() {
  local port="$1"
  local name="$2"

  for i in $(seq 1 120); do
    if curl -fs "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[ready] $name http://127.0.0.1:${port}/health"
      return 0
    fi
    sleep 2
  done

  echo "[error] $name did not become healthy on port $port"
  return 1
}

if [[ "$START_SERVERS" == "1" ]]; then
  echo
  echo "Starting FSigLIP servers..."

  for i in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$i]}"
    port=$((BASE_PORT + i))
    log="$LOG_DIR/server_gpu${gpu}_port${port}.log"

    echo "[server $i] gpu=$gpu port=$port log=$log"

    (
      source "$CONDA_BASE/bin/activate"
      conda activate fsiglip
      export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
      export CUDA_VISIBLE_DEVICES="$gpu"

      python fsiglip/serve_fsiglip_knn.py \
        --port "$port" \
        --gpu 0 \
        --corpus "$FS_CORPUS"
    ) > "$log" 2>&1 &

    SERVER_PIDS+=("$!")
  done
fi

echo
echo "Checking FSigLIP server health..."
for i in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$i]}"
  port=$((BASE_PORT + i))
  wait_health "$port" "server_gpu${gpu}"
done

echo
echo "Starting collector workers..."

for i in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$i]}"
  port=$((BASE_PORT + i))

  shard_start=$((i * CHUNK))
  remain=$((TOTAL_LIMIT - shard_start))

  if [[ "$remain" -le 0 ]]; then
    echo "[worker $i] skip: no remaining plans"
    continue
  fi

  shard_limit="$CHUNK"
  if [[ "$remain" -lt "$CHUNK" ]]; then
    shard_limit="$remain"
  fi

  shard_offset=$((OFFSET + shard_start))
  shard_root="$OUT_ROOT/shard_${i}_gpu${gpu}_offset${shard_offset}_limit${shard_limit}"
  log="$LOG_DIR/worker_gpu${gpu}_offset${shard_offset}_limit${shard_limit}.log"

  echo "[worker $i] gpu=$gpu port=$port offset=$shard_offset limit=$shard_limit out=$shard_root"

  # shellcheck disable=SC2206
  EXTRA_ARRAY=( $EXTRA_ARGS )

  (
    source "$CONDA_BASE/bin/activate"
    conda activate sam3
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/external/sam3:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="$gpu"

    python fsiglip/collect_topk_sam3_fsiglip_patch_rank.py \
      --plan-path "$PLAN_PATH" \
      --client-url "http://127.0.0.1:${port}/knn-service" \
      --score-url "http://127.0.0.1:${port}/score-image-files" \
      --index-name "$INDEX_NAME" \
      --output-root "$shard_root" \
      --query-mode "$QUERY_MODE" \
      --retrieval-k "$RETRIEVAL_K" \
      --score-top-n "$SCORE_TOP_N" \
      --offset "$shard_offset" \
      --limit "$shard_limit" \
      --options "$OPTIONS" \
      --device cuda \
      --short-side-tiles "$SHORT_SIDE_TILES" \
      --min-coverage "$MIN_COVERAGE" \
      --pattern-margin "$PATTERN_MARGIN" \
      --color-margin "$COLOR_MARGIN" \
      --color-mode "$COLOR_MODE" \
      --print-ranking-n "$PRINT_RANKING_N" \
      --progress-every "$PROGRESS_EVERY" \
      --force \
      "${EXTRA_ARRAY[@]}"
  ) > "$log" 2>&1 &

  WORKER_PIDS+=("$!")
done

echo
echo "Waiting for workers..."
FAIL=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo
  echo "[error] one or more workers failed. Check logs:"
  echo "$LOG_DIR"
  exit 1
fi

echo
echo "All workers finished."

echo
echo "Writing summary..."
python - "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = root / "_summary_top1.tsv"

rows = []
for tsv in sorted(root.glob("shard_*/ranked_images/*/*/_ranking.tsv")):
    lines = tsv.read_text(encoding="utf-8").splitlines()

    meta = {}
    for line in lines:
        if "\t" in line and not line.startswith("rank\t"):
            k, v = line.split("\t", 1)
            if k in {"query_id", "option", "option_text", "target"}:
                meta[k] = v

    data = [x for x in lines if x.startswith("001\t")]
    if not data:
        continue

    parts = data[0].split("\t")
    rows.append([
        meta.get("query_id", ""),
        meta.get("option", ""),
        meta.get("option_text", ""),
        meta.get("target", ""),
        *parts,
        str(tsv),
    ])

header = [
    "query_id", "option", "option_text", "target",
    "rank", "orig_rank", "combined", "pattern", "color", "garment",
    "similarity", "sam3_prompt", "skip_reason", "ranking_tsv",
]

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    f.write("\t".join(header) + "\n")
    for r in rows:
        f.write("\t".join(r) + "\n")

print(out)
PY

echo
echo "Done."
echo "Output root: $OUT_ROOT"
echo "Logs:        $LOG_DIR"
echo "Summary:     $OUT_ROOT/_summary_top1.tsv"
echo
echo "Ranking files:"
find "$OUT_ROOT" -name "_ranking.tsv" | sort
