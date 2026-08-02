# POD-Bench v2: Personalized Outfit Decision Benchmark

A VLM personalization benchmark that tests whether vision-language models can recommend fashion items by jointly considering **user preferences** (likes/dislikes) and **TPO constraints** (Time, Place, Occasion).

## Key Design: Canonical Extreme Scenarios

Unlike v1 (random TPO sampling → ambiguous contrasts), v2 uses **59 curated canonical extreme scenarios** across **20 archetypes** (`configs/scenarios.py`) where TPO-compatible vs TPO-incompatible distinctions are **grounded in physical danger, functional impossibility, or widely shared convention** — not in the authors' taste. Physical scenarios rest on the first two and are culture-invariant; dress-code scenarios rest on the third and are therefore explicitly scoped by `EVAL_FRAME_CLAUSE` (currently: mainstream contemporary United States conventions, unless the query states a different rule). The convention-based half is **not yet human-validated** — annotation is planned (§4); until it lands, treat those labels as authored, not verified.

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
- **v12** (`wacv_scenario_v3`; see `docs/wacv_scenario_v3_report.md`): garment vocabulary expanded from 20 to 23 by replacing `trench_coat` with `pea_coat`/`long_coat` and adding `suit_vest`/`polo_shirt`. All 73 profile scenarios and the profile anchors were rebuilt. Dress garment top-3 concentration is **2.59x**, with 0 construction failures.
- **v13** (`wacv_scenario_v4`; see `docs/wacv_scenario_v4_report.md`): the color ANCHOR/RESERVE tiers were removed. Each profile now receives 3 liked and 3 disliked colors from the full 13-color vocabulary by least-used-first global assignment, subject to the existing cell-liveness predicate. Garment and pattern tiers, quotas, scenarios, and vocabularies are unchanged; v3 remains preserved.
- **v14** (`wacv_scenario_v5`; see `docs/wacv_scenario_v5_report.md`): two construction changes and one catalog correction. (a) `solid` is treated as the **baseline level** of the pattern axis rather than a sixth pattern value — preference is assigned over the five patterned values only, so `solid` is preference-neutral for every user. (b) The option planner takes the human-audited image cell library as a **generation input**: a candidate is only considered if all four of its options land on cells a human marked available, applied as a candidate constraint inside the balance objectives rather than as a post-hoc drop. (c) `wedding_reception` pattern constraint corrected — floral guest attire is ordinary at a US wedding reception and had no conventional basis for being filed with leopard print; `floral` moves to compatible, `polka_dot` to incompatible. This is the same correction v10 applied to the business and golf scenarios. Scenario IDs, seeds, justifications, the garment and color quotas, and all vocabularies are unchanged; v4 remains preserved.

## Canonical Vocabulary

`configs/config.py` defines the benchmark labels (not a full fashion taxonomy): 13 colors, 6 patterns (`solid, striped, checkered, floral, polka_dot, leopard` — camouflage/argyle/plaid removed for retrieval stability), 23 garments. Rendering aliases ("fleece jacket", "wool coat", "leopard print", "formal shirt") live only in the retrieval/rendering layer, never in the config.

## Backward-Designed Profiles

**24 rule-generated preference profiles** (`construction/profile_generator.py`). Each profile is built from explicit axis quotas, not hand-authored personas. Garment keeps its ANCHOR/FREE construction. Color has no tiers: all 13 colors are eligible for the 3 liked and 3 disliked slots, assigned globally least-used-first subject to the existing cell-liveness predicate. Pattern has no tiers either in v5: preference is assigned over the five **patterned** values, and `solid` is reserved as the axis baseline. `configs/profiles.py` retains the historical v2 snapshot; v3/v4 data remain frozen and v5 is generated from the current rules. No semantic archetype labels exist in v3, v4, or v5.

1. **Maximize compatibility** across the 20 scenario archetypes — each variant's 4 garment likes / 4 dislikes leave 15 neutral garments, so almost every scenario has a clean neutral TPO-compatible vs TPO-incompatible garment pair for the planner. In v5, **1,401 of 1,416 user × scenario combinations (24 × 59) produce at least one query**, and 1,362 survive Stage 3 with at least one plan. Every one of the 59 scenarios ends up with **≥19 plans covering ≥17 of the 24 users**, and 39 of them cover all 24. The 15 query-level holes are listed in `docs/wacv_scenario_v5_report.md`.

   **Per-axis quotas are asymmetric and this must be stated in the paper:**

   | axis | vocabulary | likes | dislikes | assignment | neutral per user |
   |---|---:|---:|---:|---|---:|
   | garment | 23 | 4 | 4 | 2+2 from the 7 ANCHOR, 2+2 from the 16 FREE | **15** |
   | color | 13 | 3 | 3 | globally balanced over all 13 — no ANCHOR tier | **7** |
   | pattern | 6 | 2 | 2 | globally balanced over the 5 patterned values; `solid` never assigned | **2** |

   The seven garment anchors are `blazer, formal_shirt, dress, slacks, long_skirt, suit_vest, long_coat`; 2 are liked, 2 disliked, and 3 stay preference-neutral per user. A user's two neutral patterns are always `solid` plus one patterned value.
2. **Strict 2×2 with real dislikes.** B/D always use a profile-disliked value, never a neutral fallback — and never the baseline, since `solid` is neutral for everyone. The consequence is disclosed rather than patched: in the 13 scenarios whose compatible pattern set is `{solid, striped}` or `{solid}`, pattern cannot be the preference axis at all, because one value cannot form a like/dislike pair. Where there is no room to choose a pattern there is no room to express a pattern taste. Those same scenarios instead carry pattern as the *TPO* axis, which v4 could not construct at all.
3. **No value monoculture.** All 13 colors appear on both sides; each color is assigned to 9–14 users overall, with 5–7 likes and 4–7 dislikes. Each of the five patterned values is liked by 9–10 users and disliked by 9–10 (deviation ≤ 1, asserted by the generator); `solid` is liked and disliked by exactly 0. Profiles have no semantic archetype labels.

## 4-Option Structure

Each instance has 4 options along one active axis (`active_axis` ∈ {color, pattern, garment_category} — v5 realizes color 1,237 / pattern 1,062 / garment 342):
- **A** (tpo_and_preference): liked value + TPO-compatible violation axis
- **B** (tpo_only): disliked value + TPO-compatible violation axis
- **C** (preference_only): liked value + TPO-violated violation axis
- **D** (neither): disliked value + TPO-violated violation axis

The active axis carries preference; a *different* axis carries the TPO contrast and is always **preference-neutral** for that user. Usually the violation axis is the garment, but in dress-code scenarios color or pattern can carry the norm instead, which is what makes garment-active plans possible.

The remaining third axis is the **background**: identical across A/B/C/D, so it is a constant rather than a contrast and cannot help identify the answer. It must still be preference-neutral and situation-safe. When that axis is pattern, v5 pins it to `solid` — the baseline is neutral for every user, compatible in all 29 pattern-constrained scenarios, and 100% image-available, so the background never kills an item. When it is color, the value is drawn from `scenario-compatible ∩ user-neutral`; if that intersection is empty the axis is left unfixed rather than filled with a default (12 plans in v5). There is no implicit fallback anywhere downstream: an unfixed axis stays unfixed and the plan is simply not image-realizable.

The planner (`construction/option_planner.py`) reads the human-audited cell library first and only *considers* candidates whose four options all land on an available `color|garment_category|pattern` cell. This is a hard candidate constraint evaluated **before** the soft objectives below, not a filter applied to finished items — the difference matters, because dropping items afterwards destroys exactly the balance the objectives just bought (v4 lost 67% of its plans that way and its track split moved 62:38 → 69:31). Within that feasible set it assigns values with three global objectives:
- **Counterbalance**: each value is pushed toward equal A (liked) and B (disliked) use. The full v5 dataset's preference-blind value-prior accuracy is 0.544; the released active-value-prior-matched subset is exactly 0.50 by construction.
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
         ↑ generation input:   annotation/attribute_library.json  (human-audited available image cells)
Stage 4: Image Collection      fsiglip/ (FashionSigLIP KNN + SAM3 mask + patch re-rank + VLM garment judge)
Stage 5: Label Verification    (rule-based + multi-judge LLM ensemble)
Stage 6: Quality Audit         (assembly + vision-essentiality gate)
Stage 7: Evaluation            (multimodal + text-only MCQ, 3 scores, per-axis/archetype)
```

Stages 1–3 are deterministic (no LLM calls) and fully reproducible from `configs/` + `construction/` with a fixed seed. Since v5, Stage 3 additionally consumes `annotation/attribute_library.json` (the 1,237 available cells from the 1,661-cell human annotation pass). It is a **generation input**, so it is hash-pinned alongside the outputs: changing the library changes the released plans.

**Image collection (Stage 4) is the engineered core.** Retrieval is frozen on **Marqo-FashionSigLIP** (`fsiglip/serve_fsiglip_knn.py`, 651k Amazon-fashion corpus). The current collector is `fsiglip/collector_sam3.py`:
- **Garment axis — VLM judge.** A closed-vocabulary VLM classification of the worn garment (SAM3-mask *scoring* was rejected: as a localizer it cannot discriminate dress ↔ tank top). SAM3 text-prompted masks localize the garment for patch extraction.
- **Pattern axis — patch-based.** Per-tile pattern classification over the garment mask against a flat closed vocabulary (`PATTERN_VOCAB`). *Planned, not implemented:* a decision-aware pattern-FAMILY hit rule to rescue near-miss family matches — no family grouping exists in the current collector.
- **Color axis — per-tile color argmax** that catches Navy↔Blue / Beige↔Brown confusions a single pooled embedding passes.
- Gates are fail-open. *Planned, not implemented in the current collector:* gender consistency and intra-set URL dedup within each 4-option set. Both exist only in the retired `_archive/vit/collect_images_vit_coverage_v7.py` path and were not carried over to the FashionSigLIP collector.
- **Garment labels are the canonical 23 from `configs/config.py`, with no alias, synonym or hypernym layer anywhere.** `wool coat` had been used as a stand-in for `long_coat`; it is a hypernym (a pea coat is also a wool coat), so it competed with both coat classes in the VLM's closed vocabulary and depressed `pea_coat` availability to 34%, the lowest of all 23. All collectors now derive the vocabulary from the config, the `GARMENT_EQUIV_GROUPS` alias table is deleted, and the coats are separated by length and closure rather than fabric. The 139 affected cells are queued for re-collection — their current verdicts are provisional. Full record in `docs/PROCESS.md` §8.

## Quick Start

```bash
pip install -r requirements.txt

# Stages 1–3 (deterministic, no servers needed)
python -m construction.profile_generator --force
python -m construction.query_generator  --force
python -m construction.option_planner   --force \
    --cell-library annotation/attribute_library.json \
    --solid-baseline

# Stage 4: serve the FashionSigLIP index, then collect
#   see docs/SETUP_FSIGLIP.md (extract → build_faiss → serve → collect)

# Audits
python -m configs.scenarios              # scenario/archetype/axis-slot counts
python -m configs.profiles               # like/dislike tallies + overlap check
python -m scripts.validate_options       # construction validity + confound audit
python -m scripts.report_track_balance   # per-track axis + vs-pair balance
python -m scripts.report_track_balance --full   # every realized pair (appendix)
```

Both Stage 3 flags default to off, so omitting them reproduces the pre-v5
planner exactly. The released v5 data is built with both on.

Set `POD_VARIANT=<tag>` to redirect all data paths to `data_<tag>/`. Both audit
scripts honour it, so run them with the same variant that generated the data —
with `POD_VARIANT` unset they read `data/`. The current variant is
`wacv_scenario_v5`:

```bash
export POD_VARIANT=wacv_scenario_v5
```

## Current Scale (pre-retrieval)

Reported per track, never pooled:

Current variant: **`wacv_scenario_v5`** (`data_wacv_scenario_v5/`). The v2, v3
and v4 data and hashes remain preserved as historical artifacts.

| | Physical | Dress-code |
|---|---:|---:|
| scenarios / archetypes | 24 / 7 | 35 / 13 |
| option plans | 1,010 | 1,631 |
| preference axis | color 56.6%, pattern 43.4% | color 40.8%, pattern 38.3%, garment 21.0% |
| TPO violation axis | garment 100% | garment 54.1%, color 27.8%, pattern 18.1% |
| four options all image-available | 977 (96.7%) | 1,594 (97.7%) |

- 24 users × 59 scenarios: **1,401 / 1,416 combinations produce a query**, 1,362 produce a plan. No scenario is empty and none falls below 17 of 24 users; 39 of 59 cover all 24. The 15 query-level holes are listed in `docs/wacv_scenario_v5_report.md`.
- 2,561 queries generated → **2,270 unique query contexts represented** → **2,641 option plans (the item unit)**. Plans outnumber represented queries because dress-code queries can emit multiple violation-axis variants; `plan_id` — never `query_id` — is the item key.
- **2,571 of 2,641 plans (97.3%) have a human-approved image cell for all four options**, because availability is a construction-time constraint rather than a post-hoc filter. The 70 exceptions are known and enumerated in `docs/wacv_scenario_v5_report.md`: 12 plans whose background color axis had no scenario-compatible neutral value and was therefore left unfixed rather than defaulted, and 58 where the availability check ran against the query's original background pattern before the `--solid-baseline` override replaced it. Nothing has been silently defaulted to make a plan realizable.
- These are still **pre-retrieval** numbers in the sense that the per-plan image folders are not assembled and the labels are not human-validated. Recompute the tables on the frozen `plan_id` manifest before quoting them as final benchmark statistics.
- **Prompt version 2** (2026-07-27). `EVAL_FRAME_CLAUSE` now scopes dress-code judgments to mainstream contemporary **United States** conventions, and both evaluators add an explicit ordering rule ("first eliminate situation-inappropriate options, then prefer"). Results from prompt version 1 are **not comparable** and must not be pooled or trended against version 2; every run records `prompt_version` in its `.meta.json`.
- Reproducibility is checkable in one command: `bash scripts/verify_release.sh` pins the generation-input hash, regenerates the dataset into a scratch variant, compares the three SHA256s, runs the validator, and runs `tests/test_option_validator_mutations.py` (6 mutation classes × 8 plans, must report **48/48 detected** — a validator that cannot fail is not evidence). Reproducibility is a property of HEAD: v5 changed the pattern-preference rule, so v3/v4 can no longer be rebuilt from this tree and are reverified by checking out the commit that produced them.
- The downstream report should headline the **active-value-prior-matched subset** (`validation_report.counterbalanced_ids.json`; currently 1,982/2,641 = 75.0%, of which 1,933 are image-complete), on which a preference-blind guess at the ACTIVE-axis value is 0.50 by construction. It matches that one prior only — position, garment, and image-quality confounds are untouched, so it is not a "confound-free" pool. That file lists **plan_ids** (`{"id_kind": "plan_id", "ids": [...]}`). Values appearing fewer than 3 times cannot be counterbalanced at all and are excluded from the subset by definition.
- Comparable in scale to: MMPB (~500), NaturalBench (~900), BLINK (~3.8k)

**Reference SHA256** (seed 42, bit-identical on regeneration):

```
profiles.jsonl     7d1cda17eccf3d73337b50bc7ed36f63e96214193930a9ea37760e187a0afd71
queries.jsonl      cc87f1c2281a52d9fe7f28676358a85963e4a17428b9eb619776210528f27671
option_plans.jsonl ad86fd50f72f46b508f2be58ead3c2e47878ce88c2ff93ef82f628530d97ccfd
```

The option-plan hash is **provisional**: 139 `long_coat` / `pea_coat` image
cells are queued for re-collection after a garment-vocabulary correction
(`docs/PROCESS.md` §8), and the plan set must be rebuilt once the library is
re-annotated. Plans, tracks and axis distributions are unaffected — the
correction changed 624 `search_query` strings from `wool coat` to `long coat`
and nothing else. The pre-correction hash was
`813e9ab11d955794aa8eaf0b389970dc12166db8c094b124f1d8442773241f29`.

The Stage 3 generation input is pinned to the same standard:

```
annotation/attribute_library.json  72cc8665d6f92d143f850b751bba1767c342a2e9b857292e6738666bea86baae
```

Historical v2, v3 and v4 SHA256 values remain recorded in
`docs/wacv_scenario_v3_report.md`, `docs/wacv_scenario_v4_report.md` and
`docs/wacv_scenario_v5_report.md`.

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
│   └── profiles.py         # 24 rule-generated profiles (axis-specific quotas)
├── construction/           # Stages 1–3 (deterministic pipeline)
│   ├── compatibility.py    # user × scenario relaxed 2×2 compatibility
│   ├── profile_generator.py
│   ├── query_generator.py
│   └── option_planner.py   # counterbalance + confusability + diversity
├── fsiglip/                # Stage 4 (frozen FashionSigLIP backend)
│   ├── extract_fsiglip.py / build_faiss_fsiglip.py / serve_fsiglip_knn.py
│   └── collector_sam3.py  # current collector
├── scripts/                # eval + validation (multimodal_eval, text_only_eval, validate_options, …)
├── exp/                    # evaluation harnesses
│   ├── llm_eval/           # text-only eval (was text_exp/)
│   └── vlm_eval/           # vision-language eval (results/ is gitignored)
├── docs/                   # SETUP.md, SETUP_FSIGLIP.md, GARMENT_TAXONOMY_REDESIGN.md, …
├── _archive/               # retired pipelines, kept for provenance — nothing here runs
│   ├── src/                # legacy collectors (utils.py promoted to scripts/)
│   ├── before_configs/     # pre-revision configs/profiles/scenarios snapshot
│   ├── vit/                # retired ViT-L/14 collection path
│   ├── QwenEmb/            # Qwen3-VL-Embedding backend trial (not adopted)
│   └── fsiglip2/           # zooclaw-FashionSigLIP2 backend trial (not adopted)
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
