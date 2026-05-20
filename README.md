# POD-Bench v2: Preference Origin Disentanglement Benchmark

A vision-essential benchmark diagnosing whether VLM personalization methods
disentangle **intrinsic preference** from **situational TPO** context.

## What's New in v2

| Change | Reason |
|---|---|
| **One Axis Per Instance** (active_axis + fixed_attrs) | Clean per-axis diagnostic signal |
| **Phase 1 = 3 axes** (color, pattern, garment_category) | material/fit/sleeve are mixed (hedonic ↔ utilitarian) |
| **Per-axis OW indices** (OW_TPO, OW_Pref, OW_Neither) | Diagnoses *which* axis the model is failing on |
| **Parent-ASIN variant search** for color/pattern | Same product, different color = structure preserved, no diffusion |
| **Color-as-safety TPO exclusions** | mountain_hike, skiing, etc. excluded when active_axis=color |
| **Provider abstraction** (PROVIDERS dict) | Flip stages from free vllm → paid gpt-5-mini per stage |
| **Deterministic option planner** | Rule-based 4-option construction; no LLM hallucination at labels |
| **15 users × 18 instances ≈ 270** | Statistical power vs original 5-user plan |

## Core Mechanism

4-option image MCQ on the `active_axis`:

```
                    matches preference (on active_axis)?
                            YES        NO
TPO match?  YES              A          B          (A is correct)
            NO               C          D
```

Diagnostic signals (per active_axis):
- B-bias (high `OW_tpo_only`) → TPO over-weighting (PCogAlign-style)
- C-bias (high `OW_preference_only`) → preference over-weighting / mechanical FAE
- D-bias (high `OW_neither`) → cognitive collapse on this axis

## Pipeline

```
1. profile_generator → English MMPB-style profiles (PHASE1_AXES only)
2. query_generator   → (query, active_axis, fixed_attrs, tpo_scenario)
3. option_planner    → deterministic 4-option construction
4. image_collector   → parent-ASIN variant + title + Google fallback
5. label_verifier    → rule + 3-judge ensemble + α/κ
6. quality_audit     → vision-essentiality + final assembly
7. evaluator         → per-axis OW + position shuffle + variant ablation
```

## Provider Abstraction

Every LLM/VLM call dispatches through `call_llm(stage=...)` / `call_vlm(stage=...)`.
The `PROVIDERS` dict in `configs/config.py` maps each stage to a provider:

```python
PROVIDERS = {
    "profile_generation":    {"provider": "vllm"},      # free
    "query_generation":      {"provider": "vllm"},      # free
    "option_planning":       {"provider": "vllm"},      # free (unused; deterministic)
    "label_judge_primary":   {"provider": "vllm"},      # free
    "label_judge_secondary": {"provider": "vllm_alt"},  # free
    "label_judge_tertiary":  {"provider": "gpt5_mini"}, # paid (ensemble diversity)
    "blind_solver":          {"provider": "vllm"},      # free
    "captioner":             {"provider": "vllm_vlm"},  # free
    "vlm_evaluator":         {"provider": "vllm_vlm"},  # free
}
```

**Recommended workflow:**
1. Set all stages to `vllm` and run the full pipeline (no API cost)
2. Once happy, flip individual stages to `gpt5_mini` for final results

CLI override:
```bash
python -m src.profile_generator --provider gpt5_mini --n_users 5
PROVIDER=gpt5_mini bash scripts/run_pipeline.sh
```

## GPT-5-mini API Quirks (Handled)

1. Uses `max_completion_tokens` (not `max_tokens`)
2. Reasoning tokens count toward the cap; default 4096
3. `temperature` is NOT supported (gpt-5 family); we omit it
4. Defensive content extraction handles empty content / refusals / list-form content

## Quick Start

```bash
unzip pod_bench.zip && cd pod_bench
pip install -r requirements.txt

# Start local vLLM servers (see docs/SETUP.md)
bash scripts/run_pipeline.sh
```

## Output Structure

```
data/
├── profiles/profiles.jsonl
├── queries/queries.jsonl              # with active_axis + fixed_attrs
├── options/option_plans.jsonl
├── images/<query_id>/{A,B,C,D}.jpg
├── labels/labels.jsonl
├── labels/reliability_metrics.json
├── labels/audit_metrics.json
├── final/pod_bench.jsonl              # ← the benchmark
└── evaluation/<provider>__<variant>/metrics_overall.json
```
