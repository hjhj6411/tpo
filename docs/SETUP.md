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

There is no single pipeline driver script — run the stages in order. Stages 1–3
are deterministic and take seconds; Stage 4 (image collection) is the long one.

```bash
export POD_VARIANT=wacv_scenario_v3

# Stages 1-3: deterministic construction (no LLM calls)
python -m construction.profile_generator --n_users 24 --force
python -m construction.query_generator   --force
python -m construction.option_planner    --force

# Validate what you just built (must report 0 failures)
python -m scripts.validate_options

# Reproducibility check: rebuild, compare SHA256, validate, mutation-test
bash scripts/verify_release.sh
```

To use paid providers for the LLM stages, edit `PROVIDERS` in `configs/config.py`
and `export OPENAI_API_KEY=sk-...` first.

## Per-Step Partial Runs

```bash
python -m construction.profile_generator --n_users 5
python -m construction.query_generator --n_per_user 6
python -m construction.option_planner --limit 30        # deterministic, fast

# Stage 4 — image collection (current collector; see docs/SETUP_FSIGLIP.md)
python fsiglip/collector_sam3.py --limit 30

# Stage 5 — label verification (promoted from src/ on 2026-07-27)
python -m scripts.label_verifier --limit 30

# Evaluation — text-only baseline and the image MCQ, per profile mode
python -m scripts.text_only_eval --profile-mode narrative --limit 50
python scripts/multimodal_eval.py --plans ... --image-root ... --model ...
```

`src.quality_audit` and `src.evaluator` appear in older notes but do not exist in
this repo; the evaluators above replace them.

## CLI Provider Override

To run a single stage with a specific provider without editing config:

```bash
python -m construction.profile_generator --provider gpt5_mini --n_users 5
python -m scripts.text_only_eval --provider gpt5_mini --profile-mode narrative
```

## Resumability

Every step is idempotent — re-running picks up where it left off.
Use `--force` to regenerate from scratch.
