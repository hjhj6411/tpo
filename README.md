# POD-Bench v2: Personalized Outfit Decision Benchmark

A VLM personalization benchmark that tests whether vision-language models can recommend fashion items by jointly considering **user preferences** (likes/dislikes) and **TPO constraints** (Time, Place, Occasion).

## Key Design: Canonical Extreme Scenarios

Unlike v1 (random TPO sampling → ambiguous contrasts), v2 uses **60 curated canonical extreme scenarios** across 8 archetypes where TPO-compatible vs TPO-incompatible distinctions are **indisputable**:

| Archetype | # | Example |
|---|---|---|
| Extreme Cold | 8 | Blizzard outdoor → parka ✓, shorts ✗ |
| Extreme Heat | 8 | Scorching beach → tank top ✓, parka ✗ |
| Ultra-Formal | 10 | Board meeting → suit ✓, hoodie ✗ |
| Mourning/Somber | 5 | Funeral → black suit ✓, orange tank top ✗ |
| Athletic | 8 | Gym training → t-shirt ✓, blazer ✗ |
| Weather Extreme | 5 | Typhoon → windbreaker ✓, dress ✗ |
| Semi-Formal Social | 10 | Gallery opening → blazer ✓, shorts ✗ |
| Casual Outdoor | 6 | Park picnic → jeans ✓, suit jacket ✗ |

## Backward-Designed Profiles

7 preference archetypes × 3 variants = 21 users (20 used), each designed so likes/dislikes span multiple garment functional groups → maximizes scenario compatibility.

## 4-Option Structure

Each instance has 4 options along one active axis:
- **A** (tpo_and_preference): liked value + TPO-compatible
- **B** (tpo_only): non-preferred value + TPO-compatible
- **C** (preference_only): liked value + TPO-violated
- **D** (neither): non-preferred + TPO-violated

## Pipeline

```
Stage 1: Profile Generation    (archetype → narrative LLM)
Stage 2: Query Generation       (scenario × user compatibility matching)
Stage 3: Option Planning        (deterministic A/B/C/D from scenario constraints)
Stage 4: Image Collection       (Amazon catalog + Google fallback + VLM verification)
Stage 5: Label Verification     (rule-based + 3-judge LLM ensemble)
Stage 6: Quality Audit          (assembly + vision-essentiality gate)
Stage 7: Evaluation             (VLM accuracy per-axis, per-archetype)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start vLLM server(s) — see docs/SETUP.md

# Run full pipeline
bash scripts/run_pipeline.sh

# Run specific stage
bash scripts/run_pipeline.sh --stage profiles
bash scripts/run_pipeline.sh --stage queries
bash scripts/run_pipeline.sh --stage options --limit 100
```

## Expected Scale

- 60 scenarios × ~102 axis-slots/user × 20 users × ~65-75% compatibility
- → **~960–1,050 final instances**
- Comparable to: MMPB (~500), NaturalBench (~900), BLINK (~3.8k)

## Project Structure

```
pod_bench_v2/
├── configs/
│   ├── config.py          # Central configuration
│   ├── scenarios.py       # 60 canonical extreme scenarios
│   └── profiles.py        # 7 preference archetypes × 3 variants
├── src/
│   ├── utils.py           # Provider abstraction (vLLM, OpenAI)
│   ├── compatibility.py   # User × scenario compatibility matrix
│   ├── profile_generator.py
│   ├── query_generator.py
│   ├── option_planner.py
│   ├── image_collector.py
│   ├── label_verifier.py
│   ├── quality_audit.py
│   └── evaluator.py
├── scripts/
│   └── run_pipeline.sh
├── docs/
│   └── SETUP.md
└── data/                  # Generated at runtime
    ├── profiles/
    ├── queries/
    ├── options/
    ├── images/
    ├── labels/
    └── final/
```

## Citation

```
@misc{podbench2026,
  title={POD-Bench: Personalized Outfit Decision Benchmark for Vision-Language Models},
  author={VisAGI Lab},
  year={2026}
}
```
