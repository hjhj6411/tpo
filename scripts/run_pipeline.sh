#!/bin/bash
# POD-Bench v2 One-Cycle Pipeline (One Axis Per Instance)
#
# (A) ALL-FREE testing: keep PROVIDERS defaults (vllm everywhere), then
#       bash scripts/run_pipeline.sh
#
# (B) Mixed paid+free: edit configs/config.py PROVIDERS to set selected
#     stages to "gpt5_mini", then:
#       export OPENAI_API_KEY=sk-...
#       bash scripts/run_pipeline.sh
#
# CLI override at runtime: PROVIDER=gpt5_mini bash scripts/run_pipeline.sh

set -e
cd "$(dirname "$0")/.."

N_USERS=${N_USERS:-15}
N_PER_USER=${N_PER_USER:-18}
LIMIT=${LIMIT:-0}
PROVIDER=${PROVIDER:-}

PROVIDER_FLAG=""
if [ -n "$PROVIDER" ]; then
  PROVIDER_FLAG="--provider $PROVIDER"
fi

echo "============================================="
echo " POD-Bench v2 Pipeline (One Axis Per Instance)"
echo "   Users:  $N_USERS"
echo "   Inst/u: $N_PER_USER"
echo "   Override provider: ${PROVIDER:-(use config defaults)}"
echo "============================================="

echo ""
echo "Step 1/6: Profile Generation"
python -m src.profile_generator --n_users $N_USERS $PROVIDER_FLAG

echo ""
echo "Step 2/6: Query+Axis Generation"
python -m src.query_generator --n_per_user $N_PER_USER $PROVIDER_FLAG

echo ""
echo "Step 3/6: Option Planning (deterministic)"
python -m src.option_planner --limit $LIMIT

echo ""
echo "Step 4/6: Image Collection (parent-ASIN variant + fallback)"
python -m src.image_collector --limit $LIMIT

echo ""
echo "Step 5/6: Label Verification (rule + 3-judge ensemble)"
python -m src.label_verifier --limit $LIMIT

echo ""
echo "Step 6/6: Quality Audit + Final Assembly"
python -m src.quality_audit --n_audit 30

echo ""
echo "============================================="
echo " Pipeline complete"
echo "============================================="
echo "Final:        data/final/pod_bench.jsonl"
echo "Reliability:  data/labels/reliability_metrics.json"
echo "Audit:        data/labels/audit_metrics.json"
echo ""
echo "Evaluation:"
echo "  python -m src.evaluator --profile_variant combined"
echo "  python -m src.evaluator --profile_variant keyword_only"
echo "  python -m src.evaluator --profile_variant narrative_only"
