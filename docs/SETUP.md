# Setup Guide

## Environment Variables

```bash
# Required (paid: GPT-5-mini)
export OPENAI_API_KEY=sk-...

# Optional (free image fallback)
export GOOGLE_API_KEY=...
export GOOGLE_CSE_ID=...

# Optional (Amazon Reviews 2023 metadata path)
export AMAZON_META_DIR=/home/hjhj6411/fashion/data/amazon
```

Google Custom Search setup:
1. https://console.cloud.google.com — create project, enable Custom Search API
2. https://programmablesearchengine.google.com — create engine, enable
   "Image search" and "Search the entire web"
3. Free quota: 100 queries/day

## Install

```bash
pip install -r requirements.txt
pip install vllm  # GPU machine only
```

## Local vLLM Servers

POD-Bench uses different models at different stages. With 4× RTX 6000 Ada you
can run all servers in parallel:

```bash
# GPU 0,1: Qwen2.5-VL-7B (multimodal — captioner + VLM evaluator)
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --port 8000 --tensor-parallel-size 2 \
    --limit-mm-per-prompt image=5

# GPU 2: Qwen2.5-7B (text judge)
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8001

# GPU 3: Llama-3.1-8B (text judge)
CUDA_VISIBLE_DEVICES=3 vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8002
```

If using the parallel setup, update `configs/config.py`:

```python
FREE_MODELS = {
    "local_judge":     {"api_base": "http://localhost:8001/v1",
                        "model_name": "Qwen/Qwen2.5-7B-Instruct"},
    "local_judge_alt": {"api_base": "http://localhost:8002/v1",
                        "model_name": "meta-llama/Llama-3.1-8B-Instruct"},
    "vlm_evaluator":   {"api_base": "http://localhost:8000/v1",
                        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct"},
}
```

## Running

```bash
# Full pipeline
bash scripts/run_pipeline.sh

# Partial / debug
python -m src.profile_generator --n_users 5
python -m src.query_generator --n_per_user 5
python -m src.option_planner --limit 20
python -m src.image_collector --limit 20
python -m src.label_verifier --limit 20
python -m src.quality_audit --n_audit 10

# Evaluation (3 profile variants)
python -m src.evaluator --model vlm_evaluator --profile_variant combined
python -m src.evaluator --model vlm_evaluator --profile_variant keyword_only
python -m src.evaluator --model vlm_evaluator --profile_variant narrative_only
```

## Resumability

Each step is idempotent — re-running picks up where it left off. Use `--force`
to regenerate from scratch.
