# POD-Bench v2: Personalized Outfit Decision Benchmark

A VLM personalization benchmark that tests whether vision-language models can recommend fashion items by jointly considering **user preferences** (likes/dislikes) and **TPO constraints** (Time, Place, Occasion).

## Key Design: Canonical Extreme Scenarios

Unlike v1 (random TPO sampling → ambiguous contrasts), v2 uses **60 curated canonical extreme scenarios** across **16 archetypes** (`configs/scenarios.py`, currently v9) where TPO-compatible vs TPO-incompatible distinctions are **indisputable** — the incompatible garments (and, for dress-coded archetypes, colors/patterns) are wrong on grounds of physical danger, functional impossibility, or near-universal social norm. Dress-code judgments are explicitly scoped to contemporary Western / international conventions (`EVAL_FRAME_CLAUSE`).

The 16 archetypes fall into two families:

**PHYSICAL — 9 archetypes, 32 scenarios** (garment is the only TPO-constrained axis; color/pattern are pure preference axes via the relaxed compatibility check):

| Archetype | # | Example |
|---|---|---|
| Extreme Cold | 4 | Blizzard outdoor → fleece jacket ✓, shorts ✗ |
| Extreme Heat | 4 | Scorching beach → tank top ✓, fleece jacket ✗ |
| Aquatic / Water | 3 | Pool visit & lounging → t-shirt ✓, blazer ✗ |
| Athletic Indoor | 4 | Gym weight training → t-shirt ✓, blazer ✗ |
| Athletic Outdoor | 3 | Road run → shorts ✓, trench coat ✗ |
| Rugged Outdoor | 3 | Mountain hike → windbreaker ✓, dress ✗ |
| Severe Weather | 4 | Typhoon errands → windbreaker ✓, mini skirt ✗ |
| Casual Leisure | 4 | Park picnic → jeans ✓, blazer ✗ |
| Practical Work | 3 | Moving day → sweatshirt ✓, slacks ✗ |

**DRESS-CODED — 7 archetypes, 28 scenarios** (garment AND color AND/OR pattern carry TPO meaning):

| Archetype | # | Example |
|---|---|---|
| Business / Professional | 4 | Board meeting → blazer ✓, hoodie ✗; orange/pink ✗ |
| Ultra-Formal / Ceremonial | 4 | Black-tie gala → dark formal ✓, t-shirt ✗; floral ✗ |
| Judicial / Civic / Official | 4 | Court appearance → dark blazer ✓, shorts ✗; red ✗ |
| Mourning / Somber | 4 | Funeral → black solid ✓, hoodie ✗; green/pink ✗ |
| Religious / Modest | 4 | Temple ceremony → long skirt ✓, tank top ✗; leopard ✗ |
| Semi-Formal Social | 4 | Gallery opening → blazer ✓, shorts ✗; polka dot ✗ |
| Wedding / Celebration | 4 | Wedding reception → dark dress ✓, t-shirt ✗; white ✗, floral ✗ |

**Why all 16 archetypes produce instances.** In the clean 2×2 the active axis is color/pattern (PREFERENCE) and garment is the TPO axis. A physical scenario (e.g. blizzard) does not constrain color, so any liked color is situation-appropriate — color is then a *pure* preference probe while the garment alone carries TPO. The relaxed `check_axis_compatibility` treats an unconstrained active axis as "all values TPO-compatible", which is what makes the physical archetypes contribute (in v1 they produced 0 because color/pattern were unconstrained). Dress-coded archetypes keep their color/pattern constraints, so A/B are restricted to compatible values.

**Scenario revision history** (details in the `configs/scenarios.py` docstring):
- **v5–v7**: constraint-set fixes (mourning pattern pool, heat/aquatic outerwear, casual-leisure formal garments, wedding color/pattern tightening) and 7 new scenarios (practical_work ×3, citizenship oath, baptism, graduation, climbing gym) bringing 53 → 60.
- **v8**: hemisphere-ambiguity fix — month-name seasonal cues replaced with season words ("in January" → "in the middle of winter"), since month names flip meaning in the Southern Hemisphere while season words travel with the asker.
- **v9**: explicit/implicit sharpening after a full 240-seed audit — implicit seeds never state the constraint (8 leaky seeds rewritten), always license the inference (season cues added to 2 under-determined heat seeds), and dress-coded explicit seeds always state the dress expectation (19 seeds strengthened). `severe_weather` implicit seeds necessarily mention the weather (it IS the situation) and should be treated as weak-implicit in explicit-vs-implicit analyses.

## Canonical Vocabulary

`configs/config.py` defines the benchmark labels (not a full fashion taxonomy): 13 colors, 6 patterns (`solid, striped, checkered, floral, polka_dot, leopard` — camouflage/argyle/plaid removed for retrieval stability), 20 garments. The heavy-outerwear label is **`fleece_jacket`** (renamed from `fleece`, which retrieved neck gaiters); rendering aliases ("fleece jacket", "leopard print", "formal shirt") live only in the retrieval/rendering layer, never in the config.

## Backward-Designed Profiles

**8 preference archetypes × 3 variants = 24 users** (`configs/profiles.py`), each hardcoded so likes/dislikes span multiple garment functional groups. Design rules baked in:

1. **Maximize compatibility** across the 16 scenario archetypes — each variant's 3 garment likes / 3 dislikes leave ~14 neutral garments, so almost every scenario has a clean neutral TPO-compatible vs TPO-incompatible garment pair for the planner. Current compatibility: **2,760 / 2,880 slots = 95.8%** (the residual holes are users with no liked color inside the mourning/judicial `{black, navy, gray}` pools — intentional).
2. **Strict 2×2 with real dislikes.** B/D always use a profile-disliked value, never a neutral fallback. Every variant has one liked and one disliked value within the dress-code-safe pattern set `{solid, striped}`, so pattern stays a strict like/dislike axis even in formal/mourning/wedding scenarios.
3. **No value monoculture.** Pattern likes across the pool: solid 13, checkered 12, striped 11, leopard 5, floral 4, polka dot 3 — and every color appears on both the like and dislike side somewhere. `leopard` likes go to bold_expressive/streetwear (plus one adventurous_outdoor variant) where CLIP retrieval is stable; conservative profiles carry it as a dislike.

The 8 archetypes: `classic_formal`, `casual_sporty`, `minimalist`, `adventurous_outdoor`, `elegant`, `streetwear`, `relaxed_neutral`, `bold_expressive`.

## 4-Option Structure

Each instance has 4 options along one active axis (`active_axis` ∈ {color, pattern}):
- **A** (tpo_and_preference): liked value + TPO-compatible garment
- **B** (tpo_only): disliked value + TPO-compatible garment
- **C** (preference_only): liked value + TPO-violated garment
- **D** (neither): disliked value + TPO-violated garment

The garment pair (A/B vs C/D) is always **preference-neutral** for that user, so garment carries only TPO. Non-active axes are fixed to a preference-neutral, TPO-safe value when one exists, else left unfixed.

The planner (`construction/option_planner.py`) assigns values with three global objectives:
- **Counterbalance**: each value appears ~equally as A (liked) and B (disliked) across the dataset, so a preference-blind value-prior collapses to ≈0.50.
- **Confusability avoidance**: visually confusable (A, B) color pairs (black↔navy, gray↔navy, …) and (compat, incompat) garment pairs (jeans↔slacks, fleece jacket↔hoodie, …) carry a large soft penalty. After the fix, confusable color pairs dropped 45% → 19%, and the remainder are forced by dress-code pools (`{black, navy, gray}`) — flag those in judging.
- **Diversity**: per-user and per-scenario repetition of the same (a, b, garment-pair) signature is penalized; identical signatures within a user×scenario: 0.

Three-way scoring decomposes the result rather than producing a single leaderboard number:
- **Strict** = chose A
- **TPO** = chose A or B (situation satisfied)
- **Preference** = chose A or C (taste satisfied)

## Pipeline

```
Stage 1: Profile Generation    construction/profile_generator.py  (24 deterministic variants → narrative)
Stage 2: Query Generation      construction/query_generator.py    (scenario × user compatibility, clean 2×2)
Stage 3: Option Planning       construction/option_planner.py     (A/B/C/D + counterbalance + confusability)
Stage 4: Image Collection      fsiglip/ (FashionSigLIP KNN + SAM3 mask + patch re-rank + VLM garment judge)
Stage 5: Label Verification    (rule-based + multi-judge LLM ensemble)
Stage 6: Quality Audit         (assembly + vision-essentiality gate)
Stage 7: Evaluation            (multimodal + text-only MCQ, 3 scores, per-axis/archetype)
```

Stages 1–3 are deterministic (no LLM calls) and fully reproducible from `configs/` + `construction/` with a fixed seed.

**Image collection (Stage 4) is the engineered core.** Retrieval is frozen on **Marqo-FashionSigLIP** (`fsiglip/serve_fsiglip_knn.py`, 651k Amazon-fashion corpus; `fsiglip2/` is the second-generation embedding twin). The current collector is `fsiglip/collect_topk_sam3_fsiglip_patch_rank_vlm_garment_axis_patches.py`:
- **Garment axis — VLM judge.** A closed-vocabulary VLM classification of the worn garment (SAM3-mask *scoring* was rejected: as a localizer it cannot discriminate dress ↔ tank top). SAM3 text-prompted masks localize the garment for patch extraction.
- **Pattern axis — patch-based with family verification.** Per-tile pattern classification over the garment mask, with a decision-aware pattern-family hit rule (rescues near-miss family matches without admitting errors).
- **Color axis — per-tile color argmax** that catches Navy↔Blue / Beige↔Brown confusions a single pooled embedding passes.
- Gates are fail-open; gender consistency and intra-set URL dedup hold within each 4-option set.

## Quick Start

```bash
pip install -r requirements.txt

# Stages 1–3 (deterministic, no servers needed)
python -m construction.profile_generator --force
python -m construction.query_generator  --force
python -m construction.option_planner   --force

# Stage 4: serve the FashionSigLIP index, then collect
#   see docs/SETUP_FSIGLIP.md (extract → build_faiss → serve → collect)

# Audits
python -m configs.scenarios          # scenario/archetype/axis-slot counts
python -m configs.profiles           # like/dislike tallies + overlap check
python scripts/validate_options.py   # construction validity + confound audit
```

Set `POD_VARIANT=<tag>` to redirect all data paths to `data_<tag>/`.

## Expected Scale

- 24 users × 60 scenarios × 2 active axes = 2,880 slots → **2,760 compatible instances (95.8%)** = 2,760 queries / option plans
- A 70–85% complete-collection rate after Stage 4 yields the final usable set
- The downstream report should headline the **counterbalanced subset** (`validation_report.counterbalanced_ids.json`), on which the preference-blind value-prior is ≈0.50 by construction
- Comparable in scale to: MMPB (~500), NaturalBench (~900), BLINK (~3.8k)

## Project Structure

```
pod_bench/
├── configs/
│   ├── config.py           # canonical vocabulary, paths, providers (variant-aware)
│   ├── scenarios.py        # 60 canonical scenarios / 16 archetypes (v9)
│   └── profiles.py         # 8 preference archetypes × 3 variants = 24 users
├── construction/           # Stages 1–3 (deterministic pipeline)
│   ├── compatibility.py    # user × scenario relaxed 2×2 compatibility
│   ├── profile_generator.py
│   ├── query_generator.py
│   └── option_planner.py   # counterbalance + confusability + diversity
├── fsiglip/                # Stage 4 (frozen FashionSigLIP backend)
│   ├── extract_fsiglip.py / build_faiss_fsiglip.py / serve_fsiglip_knn.py
│   └── collect_topk_sam3_fsiglip_patch_rank_vlm_garment_axis_patches.py  # current collector
├── fsiglip2/               # second-generation embedding backend
├── scripts/                # eval + validation (multimodal_eval, text_only_eval, validate_options, …)
├── docs/                   # SETUP.md, SETUP_FSIGLIP.md, GARMENT_TAXONOMY_REDESIGN.md, …
├── src/                    # legacy collectors (not used by the current pipeline)
└── data/                   # generated at runtime (POD_VARIANT redirects to data_<tag>/)
    ├── profiles/ queries/ options/ images/ labels/ final/
    └── retrieval/          # collector outputs (e.g. sam3vlmfix_axis_patches/)
```

## Citation

```
@misc{podbench2026,
  title={POD-Bench: Personalized Outfit Decision Benchmark for Vision-Language Models},
  author={VisAGI Lab},
  year={2026}
}
```
