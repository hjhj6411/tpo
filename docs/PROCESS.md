# `wacv_scenario_v4` image-realization process

Status snapshot: **2026-07-31, annotation complete**. This document records
the process actually used to turn the deterministic 4-option plans into a
human-checked image library. It also records the point at which plan-level
image realization currently stops. It is not a proposal for a replacement
pipeline.

> **Superseded for construction, current for annotation.** Sections 1–6 are the
> record of the v4 run and stay as written. The two open construction decisions
> in §7 were resolved on 2026-08-01 by rebuilding Stages 1–3 as
> `wacv_scenario_v5`, which consumes the library described in §4 as a generation
> input and reaches 2,571 / 2,641 four-image plans (97.3%) instead of 1,012 /
> 3,027 (33.4%). §7 below is rewritten accordingly; see
> `docs/wacv_scenario_v5_report.md`. The annotation artifacts, their hashes and
> the screening funnel are unchanged and still authoritative.
>
> **§8 is a correction to §3–§5, not a continuation of them.** The garment
> vocabulary used to screen the `long_coat` and `pea_coat` cells contained a
> hypernym (`wool coat`). The code is fixed; those 139 cells are queued for
> re-collection and their current verdicts should be treated as provisional.

## Current outcome

- Construction produced 3,027 `plan_id` items, each with options A--D.
- The availability screen measured all 1,794
  `color x garment x pattern` cells and 179,400 retrieval candidates.
- Of the 1,661 fully specified cells referenced by at least one option plan,
  human annotation marked 1,237 available and 424 excluded. Nothing is
  pending.
- Under the strict current image key
  `color|garment_category|pattern`, 1,012 of 3,027 plans have an available
  image for all four options. The strict plan survival rate is **33.43%**.
- The final plan folders have **not** yet been assembled under
  `data_wacv_scenario_v4/images/`. The authoritative artifact at this stage is
  `annotation/attribute_library.json`.

```text
configs + seed 42
        |
        v
3,027 option plans (A/B/C/D)
        |
        v
1,794 full-grid cells x FSigLIP top 100 = 179,400 candidates
        |
        +-- benchmark-required full cells only: 1,661 / 166,100 candidates
        |       +-- SAM3 mask OK:          127,859
        |       +-- final score > 0:        38,829
        |       `-- saved top candidates:   19,012
        |
        v
human cell annotation: 1,237 selected + 424 excluded + 0 pending
        |
        v
strict four-image plans: 1,012 survive / 3,027 total
```

## 1. Deterministic benchmark construction

The active variant is `wacv_scenario_v4`. Stages 1--3 are deterministic and
use seed 42. They produced 24 profiles, 2,942 generated queries, 2,611
represented query contexts, and 3,027 option plans. `plan_id`, not `query_id`,
is the item key because one query context can produce several violation-axis
variants.

```bash
export POD_VARIANT=wacv_scenario_v4

python -m construction.profile_generator --force
python -m construction.query_generator --force
python -m construction.option_planner --force
python -m scripts.validate_options
bash scripts/verify_release.sh
```

The frozen option-plan hash is:

```text
727a79804e79f5def16dc3543f5ecebe8e4d04982842bc4c90ce252f8078982f  data_wacv_scenario_v4/options/option_plans.jsonl
```

Construction details and the other reference hashes remain in
`docs/wacv_scenario_v4_report.md`.

## 2. Frozen FashionSigLIP retrieval

The image corpus contains 651,489 Amazon-fashion images embedded with
Marqo-FashionSigLIP. Normalized vectors are searched with FAISS inner product.
The retrieval service contract is:

```bash
conda activate fsiglip
python fsiglip/serve_fsiglip_knn.py --port 1235 --gpu 0
```

```bash
curl -s http://127.0.0.1:1235/health
```

The detailed extract, index, and serving procedure is in
`docs/SETUP_FSIGLIP.md`. Retrieval rank is preserved in
`screen_sam3_candidates.jsonl`; the numeric FashionSigLIP similarity is not
preserved in the merged screening artifact.

## 3. SAM3 availability screen

The final screen covers 13 colors, 23 garments, and 6 patterns:

```text
13 x 23 x 6 = 1,794 cells
100 retrieved images per cell = 179,400 candidate rows
```

The sweep was run in color shards and merged into
`availability_audit/images/`. `merge_manifest.json` records source counts,
hashes, copy mode, and image verification. The equivalent one-run command is:

```bash
export POD_VARIANT=wacv_scenario_v4
conda activate sam3

python -m availability_audit.screen_sam3 \
  --out-dir availability_audit/images
```

This is a GPU experiment. Do not launch it on a login node. The four Qwen VLM
servers used by the screen must be running on ports 8001--8004 with enough GPU
memory left for SAM3; the recorded setup used
`--gpu-memory-utilization 0.72`. The screen is resumable, and its config guard
refuses to mix changed conditions into an existing result directory.

For every candidate with a valid SAM3 localization, the screen computes:

```text
pattern score:  target share from 5-density garment-mask patches
color score:    target share from 15-density garment-mask patches
garment score:  closed-vocabulary Qwen3-VL-4B pass (1) or fail (0)
final score:    sqrt(pattern * color * garment)
```

Pattern or color scores at or below the recorded `attr_min_score=0.2` produce
a zero final score. The final run settings are in
`availability_audit/images/screen_sam3_config.json`.

### Required-cell candidate funnel

The following counts restrict the full screen to the 1,661 fully specified
cells required by the option plans.

| Stage | Count | Retained from previous stage |
| --- | ---: | ---: |
| FashionSigLIP top-100 candidates | 166,100 | 100.00% |
| SAM3 mask OK | 127,859 | 76.98% |
| Final score greater than zero | 38,829 | 30.37% |
| Saved score-ranked candidates | 19,012 | 48.96% |

SAM3 rejected 38,232 candidates as `no_sam3_mask` and 9 as
`no_valid_patches`. Of the mask-valid candidates, 89,030 received a zero final
score. Saving is capped per cell, so the difference between 38,829 positive
candidates and 19,012 saved candidates is truncation, not an additional
invalidity judgment.

The merged screening hashes are:

```text
7e0d85d4b8b97de6ba6a4e5c7ff6275b441ee2879cf1cdc25c98bff34d5ef42a  availability_audit/images/screen_sam3.json
ab6729c3e1088cffe6d5cf1042df78d1ce9317f812cb1863b009102159ceaacf  availability_audit/images/screen_sam3_candidates.jsonl
3b8ffcbe99f5e6a6b2e6e1a55c256387fac5f28f66a95c55d4987f33fc3e7fe3  availability_audit/images/screen_sam3_config.json
```

## 4. Human cell annotation

The annotation unit is one complete `(color, garment, pattern)` cell, not one
plan or one option occurrence. A selected cell can therefore supply several
plans. Only the 1,661 cells explicitly present with all three attributes in
the option plans were placed in the default annotation queue.

```bash
cd ~/pod_bench
conda activate pod

python -m annotation.serve_annotator \
  --variant wacv_scenario_v4 \
  --screen-dir availability_audit/images \
  --port 8765
```

The UI was run in the defaults used by this command:

- `mode=select`, `pick_n=1`, and `show_n=10`;
- scorer values and scorer ranks hidden from the annotator;
- the ten score-ranked candidates shuffled with deterministic seed 42;
- no automatic acceptance (`auto_accepted_records=0`).

Restarting the same command is safe. Decisions are appended to
`annotation/cell_annotations.jsonl`, the derived library is atomically
replaced after every decision, and completed cells are omitted from the next
queue. A chosen crop is copied under `annotation/images_final/` without
silently overwriting an earlier numbered image.

### Completion and outcomes

```text
target cells       1,661
available          1,237  (74.47%)
excluded             424  (25.53%)
pending                 0
```

| Exclusion reason | Cells |
| --- | ---: |
| `pattern_absent` | 220 |
| `no_such_garment` | 194 |
| `color_mismatch` | 10 |

The selected images have these diagnostic statistics:

| Metric | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Composite screen score | 0.8635 | 0.9339 | 0.2305 | 1.0000 |
| Pattern score | 0.9242 | 1.0000 | 0.2143 | 1.0000 |
| Color score | 0.8304 | 0.9390 | 0.2022 | 1.0000 |
| Garment score | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Screen scorer rank | 3.55 | 3 | 1 | 10 |
| Original FashionSigLIP retrieval rank | 16.79 | 7 | 1 | 100 |

Screen-rank top-1 agreement is 32.01%; 60.71% of selections are within the
screen top 3, and 75.91% are within its top 5. Because scorer ties are common,
the stored scorer rank can inherit FashionSigLIP order inside a tie.

Cheap completion checks are:

```bash
python -m annotation.serve_annotator \
  --variant wacv_scenario_v4 \
  --screen-dir availability_audit/images \
  --port 8765 \
  --dry-run

python -m annotation.analyze_annotations \
  --annotations annotation/cell_annotations.jsonl \
  --library annotation/attribute_library.json
```

The dry run must report `completed: 1661 / 1661` and `this session: 0 cells`.
The completed annotation snapshot hashes are:

```text
046178bb854d0cb1450859f717d0db22f665e70540e1934fa3b508c1eaf15282  annotation/cell_annotations.jsonl
72cc8665d6f92d143f850b751bba1767c342a2e9b857292e6738666bea86baae  annotation/attribute_library.json
```

`cell_annotations.jsonl` is the append-only decision history.
`attribute_library.json` is the authoritative latest state for downstream
assembly. There are 1,336 physical JPGs under `images_final/`, but only 1,237
are referenced by the latest library. The 99 superseded files must not be
included by globbing the directory.

## 5. Four-option plan survival

A plan survives the current strict audit only when all of the following hold
for each of A, B, C, and D:

1. `color`, `garment_category`, and `pattern` are all present;
2. the exact `color|garment_category|pattern` cell exists in the library;
3. that cell's latest status is `available` and has an image path.

No default such as `missing pattern -> solid` was applied.

| Strict plan state | Plans | Share of 3,027 |
| --- | ---: | ---: |
| All four images available | **1,012** | **33.43%** |
| At least one exact cell excluded | 1,212 | 40.04% |
| Config attributes incomplete | 803 | 26.53% |

Among the 2,224 plans whose four options have complete image attributes:

| Unavailable option images in plan | Plans |
| ---: | ---: |
| 0 | 1,012 |
| 1 | 660 |
| 2 | 404 |
| 3 | 117 |
| 4 | 31 |

The active-value-prior-matched subset retains 767 of 2,186 plans, or 35.09%.

### Composition after image filtering

| Slice | Surviving / original | Survival rate |
| --- | ---: | ---: |
| Physical track | 318 / 1,152 | 27.60% |
| Dress-code track | 694 / 1,875 | 37.01% |
| Color active axis | 345 / 1,262 | 27.34% |
| Pattern active axis | 567 / 1,330 | 42.63% |
| Garment active axis | 100 / 435 | 22.99% |

The resulting 1,012-plan set is therefore not a random reduction of the
constructed benchmark. It shifts toward the dress-code track and the pattern
active axis. Balance and confound audits must be rerun on the final manifest,
not copied from the 3,027-plan construction report.

Six scenarios currently have zero strict-surviving plans:

- `celebration_graduation`
- `civic_court_appearance`
- `mourn_funeral`
- `mourn_memorial`
- `stage_greenscreen_shoot`
- `stage_tv_interview`

Their zero survival comes from incomplete config attributes, not a human image
rejection. The highest scenario survival is `social_gallery_opening` at
73.44%. The lowest nonzero scenario survival is `relig_temple_ceremony` at
7.41%.

## 6. Structural findings that must not be hidden

### Incomplete image keys

There are 803 plans for which all four options lack one required image-key
attribute:

- 799 plans omit `pattern` from A--D;
- 4 plans omit `color` from A--D.

This follows the construction policy that leaves a non-active axis unfixed
when no preference-neutral, TPO-safe fixed value is available. That is valid
for the text specification but incompatible with an exact three-axis image
library. The annotator intentionally did not invent an attribute for these
plans.

Silently interpreting a missing pattern as `solid` is not part of the current
benchmark. It would change option semantics and can introduce an unintended
pattern cue. Any fill policy must be explicit, regenerated, and revalidated.

### Selection labels are cell-level

`select` mode establishes that at least one usable image exists for a cell and
records the selected image. It does not label every unselected candidate as
invalid. Candidate-level precision, AUC, or rejection rates cannot be claimed
from this annotation session; those require a separate `judge-all` study.

### Image folders are not the state database

The physical image directory intentionally retains superseded selections for
traceability. Always read the latest paths from `attribute_library.json`.
Deleting or deduplicating old files was not part of this session.

## 7. Next reproducible stage

### Resolved on 2026-08-01 by `wacv_scenario_v5`

1. ~~Decide whether the 803 incomplete plans are excluded or regenerated with
   explicit background attributes.~~ **Resolved: regenerated.** The cause was
   the pattern axis carrying a preference on `solid`, which left users with no
   preference-neutral plain background. `solid` is now the baseline level of
   the axis — never assigned as a like or a dislike — and the background
   pattern is pinned to it whenever pattern is neither the preference nor the
   TPO axis. Incomplete image keys fall from 803 to **12**, all on the color
   axis, all cases where `scenario-compatible ∩ user-neutral` is genuinely
   empty. No implicit fallback was added anywhere, in the assembler or
   elsewhere; those 12 stay unfixed and unrealizable by design.
2. ~~Decide whether to keep the current 1,012 strict survivors or regenerate
   plans that touch one of the 424 excluded cells.~~ **Resolved: regenerated.**
   Keeping the survivors was rejected because the 33.4% reduction is not random
   — it moved the track split from 38:62 to 31:69 and the physical preference
   axis from 50:50 to 42:58, undoing the planner's counterbalance. Stage 3 now
   takes `annotation/attribute_library.json` as a generation input and applies
   availability as a candidate constraint *before* the balance objectives, so
   the balance is optimized inside the feasible set. Result: **2,571 / 2,641
   four-image plans (97.3%)**, no empty scenario, no scenario below 17 of 24
   users. Details in `docs/wacv_scenario_v5_report.md`.

### Remaining

The next stage is plan assembly, not another annotation sweep. All paths below
refer to `wacv_scenario_v5`.

3. **Re-collect the 139 `long_coat` / `pea_coat` cells** and re-annotate them.
   The garment vocabulary they were screened under was contaminated by a
   hypernym; the code is fixed but the cells are not. See §8. This blocks the
   manifest freeze, because it changes which cells are available.
4. Annotate the 27 unannotated `solid` cells, a half-day pass that converts
   directly into additional four-image plans.
5. Freeze a `plan_id` manifest, then rerun track, active-axis, scenario, and
   counterbalance reports **on that manifest**. Do not copy the 2,641-plan
   construction report forward as if it were the manifest report.
6. Materialize exactly four files per retained plan under
   `data_wacv_scenario_v5/images/<plan_id>/{A,B,C,D}.jpg`, resolving paths only
   through `attribute_library.json`. Never glob `annotation/images_final/`: it
   holds 99 superseded files that the library no longer references.
7. Run `scripts.multimodal_eval.preflight_images` through the evaluator CLI
   before any GPU evaluation. Missing or undecodable images must fail the
   preflight rather than silently falling back to text.
8. Run independent image grading and the multimodal/text-only evaluations,
   keeping physical and dress-code results separate.

Known and deliberately deferred: 58 plans fail the four-image test because the
availability check runs against the query's background pattern before
`--solid-baseline` replaces it with `solid`. Closing that ordering gap changes
the plan set and therefore the released hashes, so it belongs to a later
variant, not to a patch on this one.

No conda environments, Slurm jobs, or long GPU experiments were changed as part
of writing this process record. The v5 construction change is confined to
`configs/scenarios.py`, `construction/option_planner.py` and
`construction/profile_generator.py`; `data_wacv_scenario_v4/` and all
annotation artifacts are untouched.

## 8. Garment vocabulary contamination (`wool coat`) and its correction

Status: **code fixed 2026-08-01; the 139 affected cells are NOT yet
re-collected.** Nothing in `annotation/` has been modified.

### 8-1. What was wrong

`wool coat` was used as a stand-in label for the canonical garment `long_coat`.
It is not a synonym but a **hypernym**: a pea coat is also a wool coat, and so
is a duffle coat. A label that sits above its target in the taxonomy cannot be
made safe by a better model — an image of a pea coat answered as "wool coat" is
simply correct.

It had entered four places at once, because the garment vocabulary was
hand-copied into every consumer instead of being derived from
`configs/config.py` the way the pattern axis already was:

| Site | Form |
| --- | --- |
| `construction/option_planner.py` `GARMENT_SEARCH_PHRASE` | `"long_coat": "wool coat"` — the only non-identity entry of 23 |
| `fsiglip/collector_sam3.py`, `collect_img_sam3.py`, `collect_topk_..._benchmark_crop.py` | `wool coat` inside the closed VLM vocabulary and the parsing dictionary |
| `GARMENT_EQUIV_GROUPS` in the same three files | `{"wool coat", "long coat", "wool overcoat", "overcoat"}` |
| `availability_audit/audit_config.py`, `scripts/build_faiss_index.py` | `long_coat -> ["wool coat", "wool overcoat"]` retrieval aliases |

### 8-2. What actually happened to the shipped library — two different paths

The contamination reached the two collection paths differently, and the
distinction matters for what re-collection should be expected to change.

**The path that produced the current library: `availability_audit/screen_sam3.py`.**
It imports its scorers from `fsiglip/collect_img_sam3.py`. Its *retrieval* was
never contaminated — `measure_cell` builds the query from the canonical name, and
the stored queries confirm it (`"black long coat"`, `"gray long coat"`, never
`"wool coat"`). What was contaminated is the **closed vocabulary shown to the
VLM**. `score_garment_vlm` appends the target to the vocabulary when it is
absent, so the model was shown a 24-item list holding `wool coat`, `long coat`
and `pea coat` as siblings, while the prompt told it in the same breath that
*"a pea coat is a short double-breasted wool coat"*. The run config records
`allow_garment_equivalence = False`, so PASS required an exact match: any coat
the model resolved to the hypernym was **rejected**, not falsely accepted. This
depresses availability, and it depresses it hardest for pea coats, which the
prompt actively invited the model to call `wool coat`. Measured on the screen
rows: `garment_score = 1.0` for 4,099/7,800 `long_coat` candidates but only
1,677/7,800 for `pea_coat`, and `pea_coat`'s human exclusions are dominated by
`no_such_garment` (30 of 42).

The run manifest did not catch it either: `screen_sam3_config.json` records
`garment_vocabulary` as the canonical 23 underscore labels, because that field
is built from `FASHION_ATTRIBUTE_AXES` — not from the list actually handed to
the VLM. The manifest and the model disagreed and nothing compared them. The
recorded `allow_garment_equivalence: false` is now pinned rather than read from
a flag, since the alias layer it referred to no longer exists.

**The per-plan path (`fsiglip/collector_sam3.py`, Stage 4 proper, not yet run at
scale).** This one reads `search_query` out of `option_plans.jsonl`, so it would
have retrieved `long_coat` cells with the literal string `"gray wool coat"` —
the retrieval-side contamination, pulling pea coats into long_coat cells. Fixed
before it ever ran.

Both defects are removed by the same change; only the second matches the
"long_coat absorbs pea coat candidates" story, and it never reached the shipped
library.

### 8-3. The correction

The principle is one line: **the garment vocabulary is the canonical 23 in
`configs/config.py`, and nothing else — no aliases, no hypernyms, no synonym
layer.**

- `GARMENT_SEARCH_PHRASE["long_coat"]` is now `"long coat"`, making all 23
  entries identity mappings. The two coats were introduced in v12 as a
  *length*-specific pair, and only the canonical names encode that.
- All three `fsiglip/` collectors derive `GARMENT_VOCAB` from
  `configs.config.FASHION_ATTRIBUTE_AXES["garment_category"]`, with a loud
  (never silent) fallback. `TEXT_QUERY_GARMENT_VOCAB`, `DEFAULT_GARMENTS` and
  `GARMENT_QUERY_TERMS` are now the same list. The 16 aliases that used to live
  in `DEFAULT_GARMENTS` were verified unused against all 12,108 real search
  queries.
- `GARMENT_EQUIV_GROUPS`, `garment_equivalence_set()` and
  `--allow-garment-equivalence` are deleted from all three collectors. The table
  never affected a verdict (`allow_equivalence` defaulted to False) but was
  passed `allow_equivalence=True` by `assert_vocab_covers_targets`, the one
  guard against vocabulary drift — which it therefore silenced. Six of its
  twelve groups were hypernyms and three swallowed a sibling of the canonical
  23: `wool coat` ⊃ `pea coat`, `pants` ⊃ `jeans`/`leggings`, `knitwear` ⊃
  `cardigan`.
- `VLM_GARMENT_PROMPT` now separates the coats by **length and closure**, and
  says explicitly not to judge by fabric, since both are usually wool.
- Retrieval aliases for `long_coat` are reduced to the canonical name in
  `availability_audit/audit_config.py` and `scripts/build_faiss_index.py`.
- `--only-garments` was added to all three collectors so the re-collection can
  touch only the affected cells.

Verified after the change: all three collectors expose exactly 23 categories,
`wool coat` is absent, and `assert_vocab_covers_targets` passes silently on all
10,564 options of the current plan set — the guard is now load-bearing again.

`fsiglip/collect_img_sam3.py` is **not** a dead copy and must not be archived:
`availability_audit/screen_sam3.py` imports its scorers, which makes it the
implementation behind the shipped library. `vit/` keeps the old contaminated
vocabulary and is marked RETIRED in its header; `retrieval/` carries no garment
vocabulary at all.

Resolved by the author (2026-08-01): `fsiglip/run_benchmark_crop_4gpu.sh` is not
in use, so `collect_topk_..._benchmark_crop.py` is not an execution path either.
The live pipeline is `availability_audit/screen_sam3.py` for screening and
`annotation/serve_annotator.py` for selection. All three collector copies were
fixed identically regardless, so no path can reintroduce the hypernym.

### 8-4. Re-collection runbook (screen → reset → re-annotate)

The workflow is the one already in use: `availability_audit/screen_sam3.py`
produces candidates, `annotation/serve_annotator.py` makes the final selection.
`fsiglip/run_benchmark_crop_4gpu.sh` and
`fsiglip/collect_topk_..._benchmark_crop.py` are **not** part of it.

Nothing below touches the 21 other garments. `availability_audit/images/`,
`annotation/cell_annotations.jsonl` and `annotation/attribute_library.json` are
only read until step 2 explicitly appends to the log.

#### Step 1 — re-screen the two coats (GPU, long)

Serve FashionSigLIP on 1235 and the four VLM workers on 8001–8004 first, then:

```bash
cd ~/pod_bench
conda activate sam3

python -m availability_audit.screen_sam3 \
  --garments long_coat,pea_coat \
  --out-dir availability_audit/images_coat_redo \
  --client-url http://127.0.0.1:1235/knn-service \
  --score-url  http://127.0.0.1:1235/score-image-files \
  --gpus 0,1,2,3 --vlm-ports 8001,8002,8003,8004
```

13 colours × 2 garments × 6 patterns = **156 cells in 13 colour waves**. A fresh
`--out-dir` is required, not a convenience: the run manifest now records
`vlm_garment_vocabulary`, so the same-conditions guard correctly refuses to mix
these measurements into `availability_audit/images/`, which was screened under
the contaminated vocabulary. `--redo` combined with `--garments` is rejected
outright — it would delete the whole directory's results, not just the subset.

Each worker reports the garment vocabulary it will actually show the VLM and
the parent aborts before measuring a single cell if it is not the benchmark
axis. That check is the one that was missing.

#### Step 2 — retract the old coat verdicts

```bash
# dry run: writes nothing, prints exactly what would change
python -m annotation.reset_cells --garments long_coat,pea_coat

# then, to apply
python -m annotation.reset_cells --garments long_coat,pea_coat --apply \
  --reason "rescreened under corrected garment vocabulary (PROCESS.md 8)" \
  --annotator <your name>
```

Expected: 139 target cells, 86 `available` + 53 `excluded`, 139 verdicts
retracted. `cell_annotations.jsonl` stays append-only — a `reset` record carries
the verdict it supersedes, and `attribute_library.json` is not written by this
script. The annotator rebuilds the library from the log on its next start.

#### Step 3 — re-annotate

```bash
conda activate pod

python -m annotation.serve_annotator \
  --variant wacv_scenario_v4 \
  --screen-dir availability_audit/images_coat_redo \
  --port 8765
```

The queue is exactly the 139 reset cells: target cells come from the plans and
the full attribute grid, while the annotatable set is the intersection with the
loaded snapshot, which holds only coats.

**Use `--variant wacv_scenario_v4` here, not v5.** Target keys define which
cells `build_library` writes, and v4's 1,661-cell universe is the one the
library was built on. v5's plans need only 1,137 cells, so annotating under v5
would rebuild the library with 524 fewer cells — no verdict would be lost (the
log is the source of truth) but the shipped library and its pinned hash would
silently narrow.

Three cells v5 needs are coats that were never annotated at all
(`beige|pea_coat|solid`, `red|long_coat|solid`, `red|pea_coat|solid`); they sit
outside the v4 target set and are not part of this pass. They belong with the
other 18 never-annotated v5 cells in §7 step 4.

#### Step 4 — after the human pass

1. Record the new `annotation/attribute_library.json` hash here and in README.
2. Regenerate the option plans against the new library — do **not** reuse the
   current plan set:
   ```bash
   export POD_VARIANT=wacv_scenario_v5
   python -m construction.option_planner --force \
       --cell-library annotation/attribute_library.json --solid-baseline
   ```
3. Rerun `scripts.validate_options` and `scripts.report_track_balance`, and
   update the provisional hash in README and
   `docs/wacv_scenario_v5_report.md`.

Review guidance for the human pass — the UI label must read `long coat` /
`pea coat`, never `wool coat`:

> pea coat = hip length, double-breasted, wide lapels
> long coat = knee length or below, usually single-breasted
> **Judge by length, not by fabric.** Both are usually wool.

Success criteria. `pea_coat` availability should rise from 22/64 (34%) and its
`no_such_garment` exclusions should fall from 30. `long_coat` availability may
**drop** from 64/75 (85%); that is a success signal, not a failure, if the old
number was inflated by pea coats. Report both together. If all three numbers are
unchanged, the contamination hypothesis is wrong and the before/after images
must be compared directly.

### 8-5. `slacks` spot check (done)

`pants` was the other hypernym that swallowed siblings (`jeans`, `leggings`).
`slacks` was retrieved under its canonical name and sits at 87% availability, so
it was not re-collected; instead 20 available cells were inspected — all 11
`solid` cells plus 9 patterned ones. **No jeans and no leggings appear**: no
five-pocket denim construction, no denim wash, nothing skin-tight and knit.
Two cells are borderline on formality rather than category (`gray|slacks|solid`
is a soft elastic-waist knit trouser, `purple|slacks|solid` reads as a utility
or scrubs trouser) and the patterned cells skew to wide-leg palazzo silhouettes,
which is a property of the corpus. No action taken.
