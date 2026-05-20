# Setup Guide (v2)

## Environment Variables

```bash
# Required only if any PROVIDERS stage uses 'gpt5_mini'
export OPENAI_API_KEY=sk-...

# Optional (Google image fallback, free 100/day)
export GOOGLE_API_KEY=...
export GOOGLE_CSE_ID=...

# Optional (Amazon Reviews 2023 metadata path)
export AMAZON_META_DIR=/home/hjhj6411/fashion/data/amazon
```

## Install

```bash
pip install -r requirements.txt
pip install vllm  # GPU machine only
```

## Local vLLM Servers

The v2 pipeline expects three independent endpoints by default
(see `configs/config.py` → `PROVIDER_ENDPOINTS`):

| Provider name | Port | Model |
|---|---|---|
| `vllm`     | 8000 | Qwen/Qwen2.5-7B-Instruct          |
| `vllm_alt` | 8001 | meta-llama/Llama-3.1-8B-Instruct  |
| `vllm_vlm` | 8002 | Qwen/Qwen2.5-VL-7B-Instruct       |

With 4× RTX 6000 Ada you can run all three in parallel:

```bash
# GPU 0,1: VLM (multimodal)
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --port 8002 --tensor-parallel-size 2 \
    --limit-mm-per-prompt image=5

# GPU 2: Qwen2.5-7B text
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# GPU 3: Llama-3.1-8B text
CUDA_VISIBLE_DEVICES=3 vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001
```

## Switching Stages Between vllm and gpt5_mini

The central switch is `configs/config.py` → `PROVIDERS`:

```python
PROVIDERS = {
    "profile_generation":    {"provider": "vllm"},       # ← change to "gpt5_mini" anytime
    "query_generation":      {"provider": "vllm"},
    "option_planning":       {"provider": "vllm"},
    "label_judge_primary":   {"provider": "vllm"},
    "label_judge_secondary": {"provider": "vllm_alt"},
    "label_judge_tertiary":  {"provider": "gpt5_mini"},  # ← already set to paid for ensemble diversity
    "blind_solver":          {"provider": "vllm"},
    "captioner":             {"provider": "vllm_vlm"},
    "vlm_evaluator":         {"provider": "vllm_vlm"},
}
```

Recommended testing workflow:

```bash
# (1) All-vllm dry run to validate the pipeline end-to-end
#     (No paid calls; PROVIDERS as shipped except label_judge_tertiary if you
#      don't have OPENAI_API_KEY — set it to "vllm" too)
bash scripts/run_pipeline.sh

# (2) Once pipeline works, flip stages to gpt5_mini for final runs
#     Edit configs/config.py, then:
export OPENAI_API_KEY=sk-...
bash scripts/run_pipeline.sh
```

## Per-Step Partial Runs

```bash
python -m src.profile_generator --n_users 5
python -m src.query_generator --n_per_user 6
python -m src.option_planner --limit 30        # deterministic, fast
python -m src.image_collector --limit 30
python -m src.label_verifier --limit 30
python -m src.quality_audit --n_audit 10

# Evaluation (3 profile variants)
python -m src.evaluator --profile_variant combined
python -m src.evaluator --profile_variant keyword_only
python -m src.evaluator --profile_variant narrative_only
```

## CLI Provider Override

To run a single stage with a specific provider without editing config:

```bash
python -m src.profile_generator --provider gpt5_mini --n_users 5
python -m src.evaluator --provider vllm_vlm --profile_variant combined
```

## Resumability

Every step is idempotent — re-running picks up where it left off.
Use `--force` to regenerate from scratch.
