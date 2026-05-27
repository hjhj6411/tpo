#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  POD-Bench v2 — Full Pipeline Runner
#  Usage: bash scripts/run_pipeline.sh [--stage STAGE] [--limit N]
# ═══════════════════════════════════════════════════════════
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ──────────────────────────────────────────────
STAGE="all"
LIMIT=0
FORCE=""
PROVIDER=""
N_USERS=20

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)     STAGE="$2"; shift 2 ;;
        --limit)     LIMIT="$2"; shift 2 ;;
        --force)     FORCE="--force"; shift ;;
        --provider)  PROVIDER="--provider $2"; shift 2 ;;
        --n_users)   N_USERS="$2"; shift 2 ;;
        *)           echo "Unknown: $1"; exit 1 ;;
    esac
done

run_stage() {
    local name="$1"
    shift
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  STAGE: $name"
    echo "══════════════════════════════════════════════════"
    python -m src."$@"
}

# ── Stage 1: Profile Generation ──────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "profiles" ]]; then
    run_stage "Profile Generation" profile_generator \
        --n_users "$N_USERS" \
        --output data/profiles/profiles.jsonl \
        $FORCE $PROVIDER
fi

# ── Stage 2: Query + Scenario Matching ───────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "queries" ]]; then
    LIMIT_ARG=""
    [[ $LIMIT -gt 0 ]] && LIMIT_ARG="--limit $LIMIT"
    run_stage "Query Generation" query_generator \
        --profile_path data/profiles/profiles.jsonl \
        --output data/queries/queries.jsonl \
        $FORCE $PROVIDER $LIMIT_ARG
fi

# ── Stage 3: Option Planning ─────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "options" ]]; then
    LIMIT_ARG=""
    [[ $LIMIT -gt 0 ]] && LIMIT_ARG="--limit $LIMIT"
    run_stage "Option Planning" option_planner \
        --profile_path data/profiles/profiles.jsonl \
        --query_path data/queries/queries.jsonl \
        --output data/options/option_plans.jsonl \
        $FORCE $LIMIT_ARG
fi

# ── Stage 4: Image Collection ────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "images" ]]; then
    LIMIT_ARG=""
    [[ $LIMIT -gt 0 ]] && LIMIT_ARG="--limit $LIMIT"
    run_stage "Image Collection" image_collector \
        --plan_path data/options/option_plans.jsonl \
        --output data/images/collection_log.jsonl \
        --image_root data/images \
        $FORCE $LIMIT_ARG
fi

# ── Stage 5: Label Verification ──────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "labels" ]]; then
    LIMIT_ARG=""
    [[ $LIMIT -gt 0 ]] && LIMIT_ARG="--limit $LIMIT"
    run_stage "Label Verification" label_verifier \
        --plan_path data/options/option_plans.jsonl \
        --profile_path data/profiles/profiles.jsonl \
        --query_path data/queries/queries.jsonl \
        --output data/labels/labels.jsonl \
        $LIMIT_ARG
fi

# ── Stage 6: Quality Audit + Assembly ────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "audit" ]]; then
    LIMIT_ARG=""
    [[ $LIMIT -gt 0 ]] && LIMIT_ARG="--limit $LIMIT"
    run_stage "Quality Audit" quality_audit \
        --label_path data/labels/labels.jsonl \
        --plan_path data/options/option_plans.jsonl \
        --query_path data/queries/queries.jsonl \
        --profile_path data/profiles/profiles.jsonl \
        --image_path data/images/collection_log.jsonl \
        --output data/final/benchmark.jsonl \
        --essentiality_output data/final/essentiality.json \
        $LIMIT_ARG
fi

# ── Stage 7: Evaluation ─────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "eval" ]]; then
    LIMIT_ARG=""
    [[ $LIMIT -gt 0 ]] && LIMIT_ARG="--limit $LIMIT"
    run_stage "Evaluation" evaluator \
        --benchmark_path data/final/benchmark.jsonl \
        --output data/final/eval_results.jsonl \
        --metrics data/final/eval_metrics.json \
        $PROVIDER $LIMIT_ARG
fi

echo ""
echo "══════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "══════════════════════════════════════════════════"
