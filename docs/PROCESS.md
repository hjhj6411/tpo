# `wacv_scenario_v4` image-realization process

Status snapshot: **2026-07-31, annotation complete**. This document records
the process actually used to turn the deterministic 4-option plans into a
human-checked image library. It also records the point at which plan-level
image realization currently stops. It is not a proposal for a replacement
pipeline.

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

The next stage is plan assembly, not another annotation sweep.

1. Decide whether the 803 incomplete plans are excluded or regenerated with
   explicit background attributes. Do not add an implicit fallback only in
   the image assembler.
2. Decide whether to keep the current 1,012 strict survivors or regenerate
   plans that touch one of the 424 excluded cells. The 660 plans missing only
   one option image are the smallest recovery target.
3. Freeze a `plan_id` manifest, then rerun track, active-axis, scenario, and
   counterbalance reports on that manifest.
4. Materialize exactly four files per retained plan under
   `data_wacv_scenario_v4/images/<plan_id>/{A,B,C,D}.jpg`, resolving paths only
   through `attribute_library.json`.
5. Run `scripts.multimodal_eval.preflight_images` through the evaluator CLI
   before any GPU evaluation. Missing or undecodable images must fail the
   preflight rather than silently falling back to text.
6. Run independent image grading and the multimodal/text-only evaluations,
   keeping physical and dress-code results separate.

No construction files, option-plan semantics, conda environments, Slurm jobs,
or long GPU experiments were changed as part of writing this process record.
