#!/bin/bash
# POD-Bench One-Cycle Pipeline 실행 스크립트
#
# 실행 전 준비:
#   1. export OPENAI_API_KEY=sk-...  (GPT-5-mini용)
#   2. 로컬 vLLM 서버 띄우기:
#      터미널 1: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
#      터미널 2: vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001
#      터미널 3: vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 \
#                          --limit-mm-per-prompt image=5
#      (8000 포트는 텍스트와 VLM이 공유 — 단계별로 다른 모델을 띄움)
#   3. (선택) Google Custom Search API 설정:
#      export GOOGLE_API_KEY=...
#      export GOOGLE_CSE_ID=...
#   4. (선택) Amazon Reviews 2023 메타데이터 경로:
#      export AMAZON_META_DIR=/home/hjhj6411/fashion/data/amazon

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo " POD-Bench One-Cycle Pipeline"
echo "========================================"

# Phase 1 — 작은 수로 빠르게 검증
N_USERS=${N_USERS:-50}
N_QUERIES=${N_QUERIES:-20}
LIMIT=${LIMIT:-0}

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
echo "Step 5/6: Label Verification (rule + judge ensemble)"
python -m src.label_verifier --limit $LIMIT

echo ""
echo "Step 6/6: Quality Audit & Final Benchmark Assembly"
python -m src.quality_audit --n_audit 30

echo ""
echo "========================================"
echo " Pipeline Complete"
echo "========================================"
echo "Final benchmark: data/final/pod_bench.jsonl"
echo "Audit metrics:   data/labels/audit_metrics.json"
echo "Reliability:     data/labels/reliability_metrics.json"
echo ""
echo "Next: python -m src.evaluator --model vlm_evaluator"
