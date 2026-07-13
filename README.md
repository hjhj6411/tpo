# POD-Bench v2: Personalized Outfit Decision Benchmark

A VLM personalization benchmark that tests whether vision-language models can recommend fashion items by jointly considering **user preferences** (likes/dislikes) and **TPO constraints** (Time, Place, Occasion).

## Key Design: Canonical Extreme Scenarios

Unlike v1 (random TPO sampling → ambiguous contrasts), v2 uses **53 curated canonical extreme scenarios** across **15 archetypes** where TPO-compatible vs TPO-incompatible distinctions are **indisputable** — the incompatible garments (and, for dress-coded archetypes, colors/patterns) are wrong on grounds of physical danger, functional impossibility, or near-universal social norm.

The 15 archetypes fall into two families:

**PHYSICAL** (garment is the only TPO-constrained axis; color/pattern are pure preference axes via the relaxed compatibility check):

| Archetype | # | Example |
|---|---|---|
| Extreme Cold | 4 | Blizzard outdoor → fleece ✓, shorts ✗ |
| Extreme Heat | 4 | Scorching beach → tank top ✓, fleece ✗ |
| Aquatic / Water | 3 | Lap swimming → t-shirt ✓, blazer ✗ |
| Athletic Indoor | 3 | Gym weight training → t-shirt ✓, suit jacket ✗ |
| Athletic Outdoor | 3 | Road run → shorts ✓, coat ✗ |
| Rugged Outdoor | 3 | Mountain hike → jacket ✓, dress ✗ |
| Severe Weather | 4 | Typhoon errands → windbreaker ✓, dress ✗ |
| Casual Leisure | 4 | Park picnic → jeans ✓, suit jacket ✗ |

**DRESS-CODED** (garment AND color AND/OR pattern carry TPO meaning):

| Archetype | # | Example |
|---|---|---|
| Business / Professional | 4 | Board meeting → suit ✓, hoodie ✗ |
| Ultra-Formal / Ceremonial | 4 | Black-tie gala → dark suit ✓, graphic-print tee ✗ |
| Judicial / Civic / Official | 3 | Court appearance → dark blazer ✓, shorts ✗ |
| Mourning / Somber | 4 | Funeral → black solid suit ✓, orange floral tank top ✗ |
| Religious / Modest | 3 | Temple ceremony → covered modest dress ✓, tank top ✗ |
| Semi-Formal Social | 4 | Gallery opening → blazer ✓, shorts ✗ |
| Wedding / Celebration | 3 | Wedding reception → dress ✓, camo hoodie ✗ |

**Why 15 archetypes now all produce instances.** In the clean 2×2 the active axis is color/pattern (PREFERENCE) and garment is the TPO axis. A physical scenario (e.g. blizzard) does not constrain color, so any liked color is situation-appropriate — color is then a *pure* preference probe while the garment alone carries TPO. The relaxed `check_axis_compatibility` treats an unconstrained active axis as "all values TPO-compatible", which is what makes the physical archetypes contribute (in v1 they produced 0 because color/pattern were unconstrained). Dress-coded archetypes keep their color/pattern constraints, so A/B are restricted to compatible values.

## Backward-Designed Profiles

**8 preference archetypes × 3 variants = 24 users** (`configs/profiles.py`), each hardcoded so likes/dislikes span multiple garment functional groups → maximizes scenario compatibility. Two design goals are baked in:

1. **Maximize compatibility** across the 15 scenario archetypes — each variant's ~3 garment likes / ~3 dislikes leave ~12 neutral garments, so almost every scenario has a clean neutral TPO-compatible vs TPO-incompatible garment pair for the planner.
2. **Eliminate the `solid`-monoculture confound** at the source. Liked patterns are spread across the whole vocabulary (`solid` is a liked pattern for only 7 of 24 users; population tally over 48 slots: striped 9, checkered 8, solid 7, plaid 5, floral 5, graphic_print 5, polka_dot 4, camouflage 3, animal_print 2). A new `bold_expressive` archetype deliberately likes bright colors + loud patterns to balance the dataset away from a dark/solid monoculture. Combined with the counterbalanced option planner, the preference-blind value-prior collapses to ≈0.50.

The 8 archetypes: `classic_formal`, `casual_sporty`, `minimalist`, `adventurous_outdoor`, `elegant`, `streetwear`, `relaxed_neutral`, `bold_expressive`.

## 4-Option Structure

Each instance has 4 options along one active axis (`active_axis` ∈ {color, pattern}):
- **A** (tpo_and_preference): liked value + TPO-compatible garment
- **B** (tpo_only): non-preferred value + TPO-compatible garment
- **C** (preference_only): liked value + TPO-violated garment
- **D** (neither): non-preferred value + TPO-violated garment

Three-way scoring decomposes the result rather than producing a single leaderboard number:
- **Strict** = chose A
- **TPO** = chose A or B (situation satisfied)
- **Preference** = chose A or C (taste satisfied)

## Pipeline

```
Stage 1: Profile Generation    (24 hardcoded archetype variants → narrative)
Stage 2: Query Generation       (scenario × user compatibility matching, clean 2×2)
Stage 3: Option Planning        (deterministic A/B/C/D + GLOBAL counterbalancing)
Stage 4: Image Collection       (FashionSigLIP/ViT KNN + patch-coverage re-rank + VLM gate)
Stage 5: Label Verification     (rule-based + 3-judge LLM ensemble)
Stage 6: Quality Audit          (assembly + vision-essentiality gate)
Stage 7: Evaluation             (multimodal + text-only MCQ, 3 scores, per-axis/archetype)
```

**Image collection (Stage 4) is the engineered core.** Retrieval is frozen on **Marqo-FashionSigLIP** (served by `fsiglip/serve_fsiglip_knn.py` on `:1235`; a ViT-L/14 clip-retrieval backend on `:1234` is the legacy twin). On top of top-1 retrieval, single-purpose deterministic gates re-rank candidates:
- **Pattern coverage** (`/patch-coverage`): the image is tiled on an N×N grid, each non-background tile is argmax-classified `{pattern} fabric` vs `plain`, and coverage = patterned tiles / valid tiles. This separates whole-garment patterns from localized ones, which a single average-pooled SigLIP embedding cannot (FashionVLP, CVPR'22).
- **Color coverage** (`/patch-color-coverage`): a 13-way per-tile color argmax (pattern held constant in the anchor) that catches the Navy↔Blue / Beige↔Brown confusions a binary anchor passes.
- **Gender consistency + intra-set URL dedup**: the four options of one query share a model gender and are four distinct images.

All gates are **fail-open** (a flaky VLM degrades to plain top-1). Re-rank semantics: pattern-axis non-solid uses (pcov, sim) or, with color coverage on, a `pcov × ccov` product; color axis uses (ccov, sim); everything else uses retrieval order.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start vLLM server(s) — see docs/SETUP.md and docs/SETUP_FSIGLIP.md

# Run full pipeline
bash scripts/run_pipeline.sh

# Run specific stage
bash scripts/run_pipeline.sh --stage profiles
bash scripts/run_pipeline.sh --stage queries
bash scripts/run_pipeline.sh --stage options --limit 100
```

## Expected Scale

- 53 scenarios × ~2 axis-slots/user × 24 users × compatibility rate
- → low-thousands of compatible instances before collection; a 70–85% complete-collection rate yields the final usable set
- The downstream report should headline the **counterbalanced subset** (`validation_report.counterbalanced_ids.json`), on which the preference-blind value-prior is ≈0.50 by construction
- Comparable in scale to: MMPB (~500), NaturalBench (~900), BLINK (~3.8k)

## Project Structure

```
pod_bench_v2/
├── configs/
│   ├── config.py          # Central configuration (variant-aware data paths)
│   ├── scenarios.py        # 53 canonical scenarios across 15 archetypes
│   └── profiles.py         # 8 preference archetypes × 3 variants = 24 users
├── src/
│   ├── utils.py            # Provider abstraction (vLLM, OpenAI)
│   ├── compatibility.py    # User × scenario compatibility (relaxed 2×2)
│   ├── profile_generator.py
│   ├── query_generator.py
│   ├── option_planner.py   # clean 2×2 + global counterbalancing
│   ├── image_collector.py            # local Amazon FAISS backend
│   ├── collect_images_clip_retrieval.py   # ViT clip-retrieval + agent/ReAct
│   ├── collect_images_siglip_coverage_v6.py  # FashionSigLIP + pattern coverage
│   ├── collect_images_vit_coverage_v7.py     # + color coverage re-rank
│   ├── label_verifier.py
│   ├── quality_audit.py
│   └── evaluator.py
├── fsiglip/
│   ├── extract_fsiglip.py        # embed Amazon corpus with FashionSigLIP
│   ├── build_faiss_fsiglip.py    # build IndexFlatIP + ids table
│   └── serve_fsiglip_knn.py      # KNN + /patch-coverage + /patch-color-coverage
├── scripts/
│   ├── build_faiss_index.py      # per-garment FAISS sub-indices
│   ├── multimodal_eval.py        # image-MCQ eval (3 scores, shuffle, thinking)
│   ├── text_only_eval.py         # text-attribute baseline (no/narrative/all)
│   ├── validate_options.py       # construction validity + confound audit
│   ├── grade_options.py / eval_pv.py   # independent VLM-judge grading
│   └── run_pipeline.sh
├── docs/
│   ├── SETUP.md
│   ├── SETUP_CLIP_ENV.md          # legacy ViT clip-retrieval backend
│   └── SETUP_FSIGLIP.md           # FashionSigLIP extract → index → serve → collect
└── data/                  # Generated at runtime (POD_VARIANT redirects to data_<tag>/)
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