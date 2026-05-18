#!/bin/bash
# POD-Bench One-Cycle Pipeline
#
# Prerequisites:
#   export OPENAI_API_KEY=sk-...           # GPT-5-mini (paid)
#   # Optional fallback for image search:
#   export GOOGLE_API_KEY=...
#   export GOOGLE_CSE_ID=...
#   # Optional Amazon catalog path:
#   export AMAZON_META_DIR=/path/to/amazon
#
# Local vLLM servers (see docs/SETUP.md):
#   Steps 5 (label_verifier): Qwen2.5-7B at :8000, Llama-3.1-8B at :8001
#   Steps 6-7 (audit, eval):  Qwen2.5-VL-7B at :8000

set -e
cd "$(dirname "$0")/.."

N_USERS=${N_USERS:-50}
N_QUERIES=${N_QUERIES:-20}
LIMIT=${LIMIT:-0}

echo "========================================"
echo " POD-Bench One-Cycle Pipeline (English)"
echo " Domain: fashion only"
echo "========================================"

echo ""
echo "Step 1/6: Profile Generation ($N_USERS users)"
python -m src.profile_generator --n_users $N_USERS

echo ""
echo "Step 2/6: Query Generation ($N_QUERIES per user)"
python -m src.query_generator --n_per_user $N_QUERIES

echo ""
echo "Step 3/6: Option Planning"
python -m src.option_planner --limit $LIMIT

echo ""
echo "Step 4/6: Image Collection"
python -m src.image_collector --limit $LIMIT

echo ""
echo "Step 5/6: Label Verification (rule + 3-judge ensemble)"
python -m src.label_verifier --limit $LIMIT

echo ""
echo "Step 6/6: Quality Audit + Final Assembly"
python -m src.quality_audit --n_audit 30

echo ""
echo "========================================"
echo " Pipeline Complete"
echo "========================================"
echo "Final benchmark:  data/final/pod_bench.jsonl"
echo "Audit metrics:    data/labels/audit_metrics.json"
echo "Reliability:      data/labels/reliability_metrics.json"
echo ""
echo "Run evaluation:"
echo "  python -m src.evaluator --model vlm_evaluator --profile_variant combined"
echo "  python -m src.evaluator --model vlm_evaluator --profile_variant narrative_only"
echo "  python -m src.evaluator --model vlm_evaluator --profile_variant keyword_only"
