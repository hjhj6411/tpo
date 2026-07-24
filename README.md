# POD-Bench v2: Personalized Outfit Decision Benchmark

A VLM personalization benchmark that tests whether vision-language models can recommend fashion items by jointly considering **user preferences** (likes/dislikes) and **TPO constraints** (Time, Place, Occasion).

## Key Design: Canonical Extreme Scenarios

Unlike v1 (random TPO sampling → ambiguous contrasts), v2 uses **59 curated canonical extreme scenarios** across **20 archetypes** (`configs/scenarios.py`) where TPO-compatible vs TPO-incompatible distinctions are **indisputable** — the incompatible garments (and, for dress-coded archetypes, colors/patterns) are wrong on grounds of physical danger, functional impossibility, or near-universal social norm. Dress-code judgments are explicitly scoped to contemporary Western / international conventions (`EVAL_FRAME_CLAUSE`).

**Physical and Dress-code are two separate datasets.** They are constructed, reported, and scored independently; sample sizes and axis shares are never compared across them. See `docs/redesign_v2_plan.md` §14–§15 for the rationale and the current construction statistics.

**PHYSICAL — 7 archetypes, 24 scenarios** (garment is the only TPO-constrained axis; color/pattern are pure preference axes via the relaxed compatibility check):

| Archetype | # | Example |
|---|---|---|
| Extreme Cold / Winter | 4 | Blizzard outdoor → fleece jacket ✓, shorts ✗ |
| Extreme Heat / Summer | 4 | Scorching beach → tank top ✓, fleece jacket ✗ |
| Indoor Athletic / Gym | 4 | Gym weight training → t-shirt ✓, blazer ✗ |
| Water & Swimming | 3 | Poolside cover-up → t-shirt ✓, blazer ✗ |
| Outdoor Athletic / Field Sports | 3 | Road run → shorts ✓, trench coat ✗ |
| Rugged Outdoor / Hiking & Camping | 3 | Mountain hike → windbreaker ✓, dress ✗ |
| Manual / Practical Work | 3 | Moving day → sweatshirt ✓, slacks ✗ |

**DRESS-CODED — 13 archetypes, 35 scenarios** (garment AND color AND/OR pattern carry TPO meaning). Each scenario constrains garment plus **at least one** of color/pattern — not necessarily both; a scenario is only given a constraint that can actually be justified:

| Archetype | # | Constrained axes | Example |
|---|---|---|---|
| Business / Professional | 4 | garment+color+pattern | Board meeting → blazer ✓, hoodie ✗; orange ✗ |
| Religious / Sacred / Modest | 4 | garment+pattern | Temple ceremony → long skirt ✓, tank top ✗; leopard ✗ |
| Semi-Formal Social | 4 | garment+pattern | Gallery opening → blazer ✓, shorts ✗; polka dot ✗ |
| Festive / Bright-Color Dress Code | 3 | garment+color | Bright-theme party → dress ✓, hoodie ✗; black/navy/gray ✗ |
| Field Stealth / Wildlife | 3 | garment+color+pattern | Wildlife hide → windbreaker ✓, dress ✗; orange ✗ |
| Garden / Daytime Floral Social | 3 | garment+pattern | Garden party → dress ✓, sweatshirt ✗; leopard ✗ |
| Night Visibility / Road Safety | 3 | garment+color | Night road run → shorts ✓, long skirt ✗; black/navy ✗ |
| Club / Institutional Athletic Code | 2 | garment+color+pattern | Golf round → slacks ✓, sweatshirt ✗ |
| Mourning / Somber | 2 | garment+color+pattern | Funeral → black solid ✓, hoodie ✗; pink ✗ |
| Stage & Media Production | 2 | garment+color+pattern | Green-screen shoot → blazer ✓, hoodie ✗; green ✗ |
| Ultra-Formal / Ceremonial | 2 | garment+color+pattern | Black-tie gala → dark formal ✓, t-shirt ✗; floral ✗ |
| Wedding / Celebration | 2 | garment+color+pattern | Wedding reception → dark dress ✓, t-shirt ✗; white ✗ |
| Judicial / Civic / Official | 1 | garment+color+pattern | Court appearance → dark blazer ✓, shorts ✗; red ✗ |

Axis coverage is therefore a **track-level** property, not a per-scenario one: across the 35 dress-code scenarios, garment is constrained in 35 (100%), pattern in 29 (83%), color in 24 (69%), and all three in 18 (51%). Forcing 100% on every axis would mean inventing color/pattern bans with no conventional basis, which is exactly the defect removed in §15.

**Why the physical archetypes produce instances.** A physical scenario (e.g. blizzard) does not constrain color, so any liked color is situation-appropriate — color is then a *pure* preference probe while the garment alone carries TPO. The relaxed `check_axis_compatibility` treats an unconstrained active axis as "all values TPO-compatible" (in v1 the physical archetypes produced 0 because color/pattern were unconstrained). Dress-coded archetypes keep their color/pattern constraints, so A/B are restricted to compatible values; there, garment may itself be the preference axis while color or pattern carries the TPO violation.

**Scenario revision history** (details in the `configs/scenarios.py` docstring):
- **v5–v7**: constraint-set fixes (mourning pattern pool, heat/aquatic outerwear, casual-leisure formal garments, wedding color/pattern tightening) and 7 new scenarios (practical_work ×3, citizenship oath, baptism, graduation, climbing gym) bringing 53 → 60.
- **v8**: hemisphere-ambiguity fix — month-name seasonal cues replaced with season words ("in January" → "in the middle of winter"), since month names flip meaning in the Southern Hemisphere while season words travel with the asker.
- **v9**: explicit/implicit sharpening after a full 240-seed audit — implicit seeds never state the constraint (8 leaky seeds rewritten), always license the inference (season cues added to 2 under-determined heat seeds), and dress-coded explicit seeds always state the dress expectation (19 seeds strengthened). `severe_weather` implicit seeds necessarily mention the weather (it IS the situation) and should be treated as weak-implicit in explicit-vs-implicit analyses.
- **v10** (see `docs/redesign_v2_plan.md` §15): scenarios that conflicted with the track definitions were removed rather than rebalanced — `severe_weather` ×4 (used the cold-weather jacket list with no waterproof/protective attribute to express the answer), `casual_leisure` ×4 (an "overdressed" dress-code judgment, not physical unsuitability), citizenship oath ×1 (color restriction with no conventional basis), and 5 that generated for almost no users. `aquatic_water` ×3 were reworded as *cover-up over swimwear* questions since `swimwear` is not in the vocabulary, the graduation white ban (copied from weddings) was dropped, and the blanket floral ban was relaxed for 4 business scenarios and golf. 60 → 59 scenarios; the 24 profiles are unchanged (SHA256 `5c06493d…3168`).
- **v11** (see `docs/redesign_v2_plan.md` §16): the dress-code garment preference axis was collapsing onto the `blazer/formal_shirt/dress` anchor trio (45% of the axis; one user had 13 of 15 items on a single pair) because only three ANCHOR garments existed and each profile took one like + one dislike, leaving every user exactly **one** formal fallback pair. Widening the scenarios alone does not fix it (measured: 6.59x → 6.99x). Fixed on both sides: `compatible` lists widened in 17 normative-formal scenarios (additions only — `slacks`/`long_skirt`; no new prohibitions), and `GARMENT_ANCHOR` grown to five with **2 likes + 2 dislikes + 1 neutral** per profile (garment quota 3+3 → **4+4**; color stays 3+3, pattern 2+2 — the asymmetry is deliberate and disclosed). Result: top-3 concentration 6.59x → **3.70x**, per-user worst pair share 87% → **28%**. All profiles/queries/plans regenerated as `wacv_scenario_v2`.

## Canonical Vocabulary

`configs/config.py` defines the benchmark labels (not a full fashion taxonomy): 13 colors, 6 patterns (`solid, striped, checkered, floral, polka_dot, leopard` — camouflage/argyle/plaid removed for retrieval stability), 20 garments. The heavy-outerwear label is **`fleece_jacket`** (renamed from `fleece`, which retrieved neck gaiters); rendering aliases ("fleece jacket", "leopard print", "formal shirt") live only in the retrieval/rendering layer, never in the config.

## Backward-Designed Profiles

**8 preference archetypes × 3 variants = 24 users** (`configs/profiles.py`), each hardcoded so likes/dislikes span multiple garment functional groups. Design rules baked in:

1. **Maximize compatibility** across the 20 scenario archetypes — each variant's 4 garment likes / 4 dislikes leave ~12 neutral garments, so almost every scenario has a clean neutral TPO-compatible vs TPO-incompatible garment pair for the planner. All **1,416 user × scenario combinations (24 × 59) are non-empty**; the residual holes are per-axis, not per-combination (users with no liked color inside the mourning/judicial `{black, navy, gray}` pools — intentional).

   **Per-axis quotas are asymmetric and this must be stated in the paper:** garment **4+4** (2 ANCHOR + 2 FREE per side), color **3+3** (1 ANCHOR + 2 FREE), pattern **2+2**. The extra garment budget buys formal-pair diversity — with five ANCHOR garments (`blazer, formal_shirt, dress, slacks, long_skirt`) at 2 likes + 2 dislikes, each user has **4 formal fallback pairs instead of 1**, and the 5th anchor always stays preference-neutral so the strictest scenarios keep a neutral TPO-compatible garment.
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
python -m configs.scenarios              # scenario/archetype/axis-slot counts
python -m configs.profiles               # like/dislike tallies + overlap check
python -m scripts.validate_options       # construction validity + confound audit
python -m scripts.report_track_balance   # per-track axis + vs-pair balance
python -m scripts.report_track_balance --full   # every realized pair (appendix)
```

Set `POD_VARIANT=<tag>` to redirect all data paths to `data_<tag>/`. Both audit
scripts honour it, so run them with the same variant that generated the data —
with `POD_VARIANT` unset they read `data/`. The current variant is
`wacv_scenario_v2`:

```bash
export POD_VARIANT=wacv_scenario_v2
```

## Current Scale (pre-retrieval)

Reported per track, never pooled:

Current variant: **`wacv_scenario_v2`** (`data_wacv_scenario_v2/`). The earlier
`wacv_scenario_v1` is kept for comparison — **do not quote v1 numbers**, they were
superseded by the ANCHOR widening (`docs/redesign_v2_plan.md` §16).

| | Physical | Dress-code |
|---|---:|---:|
| scenarios / archetypes | 24 / 7 | 35 / 13 |
| option plans | 1,152 | 2,188 |
| preference axis | color 50%, pattern 50% | color 39.1%, pattern 39.0%, garment 21.8% |
| TPO violation axis | garment 100% | garment 57.2%, color 27.7%, pattern 15.1% |

- 24 users × 59 scenarios = **1,416 combinations, all non-empty**
- 3,139 queries generated → **2,882 evaluable** (257 have no constructible 4-option plan, all dress-code — 226 on the pattern axis where strict-formal scenarios allow only `solid`/`striped`, 31 on garment) → **3,340 option plans**
- These are **pre-retrieval** numbers. Image realizability and human validation are not yet done; recompute the same tables afterwards before quoting them as final benchmark statistics.
- The downstream report should headline the **counterbalanced subset** (`validation_report.counterbalanced_ids.json`; currently 2,632/3,340 = 79%), on which the preference-blind value-prior is ≈0.50 by construction. Values appearing fewer than 3 times cannot be counterbalanced at all and are excluded from the subset by definition.
- Comparable in scale to: MMPB (~500), NaturalBench (~900), BLINK (~3.8k)

**Reference SHA256** (seed 42, bit-identical on regeneration):

```
profiles.jsonl     6642f47a850acbe63b5f916b4c246a092c6ca49ff45b725387df7e5849d7c68f
queries.jsonl      fc1c355332d76131fae7558cdcc2a308951cdc39817e92d581ae0576ce468cf5
option_plans.jsonl 8922488f2b952c357c1c64fa1c3e20d3bec343103e3a7cd96a64c981a976a3cf
```

## Scoring: two tracks, never pooled

`multimodal_eval.py` and `text_only_eval.py` split plans by their `track` field,
write results to `<out-dir>/{track}/…`, and report each track separately.
`--track physical|dress_code` scores one of them. **There is deliberately no code
path that pools the two into a single accuracy** — a pooled number mixes two
different benchmarks and must not reach the paper. Plans without a `track` field
abort the run rather than being scored silently.

Dress-code judgments are scoped by `EVAL_FRAME_CLAUSE` (`configs/scenarios.py`),
which is interpolated into every eval system prompt. **Any eval run made before
2026-07-24 predates that wiring and used a different prompt — those numbers are
not comparable and must be discarded or labelled "pre-clause".**

## Project Structure

```
pod_bench/
├── configs/
│   ├── config.py           # canonical vocabulary, paths, providers (variant-aware)
│   ├── scenarios.py        # 59 canonical scenarios / 20 archetypes, track-split
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
