# `wacv_scenario_v4` construction report

Generated with seed 42 on 2026-07-30 from the v3 code/data baseline. The
released v3 files and their hashes remain unchanged.

## Scope and rationale

v3 divided the 13-color vocabulary into three assignment groups:

- `COLOR_ANCHOR = {black, navy, gray}`: one liked and one disliked anchor per
  user;
- `COLOR_RESERVE = {orange, yellow, pink, white}`: never assigned to a user
  preference;
- six remaining FREE colors: two likes and two dislikes per user.

This guaranteed neutral violation-color supply but restricted preferences to
9 of 13 colors. v4 removes the color ANCHOR/RESERVE distinction. Every profile
now receives three liked and three disliked colors from the full 13-color
vocabulary. Candidate `(like triple, dislike triple)` assignments are ordered
least-used-first by the existing global side-specific usage counters, with the
seeded random value used only as a deterministic tie-break. The first disjoint
assignment satisfying the existing `_cells_alive_everywhere` predicate is
selected.

The color constants and their S1/S2 validation were deleted rather than left as
legacy symbols: v3 is already frozen in `data_wacv_scenario_v3/`, while keeping
inactive tier names in the active generator would make accidental reuse in v4
more likely. Garment ANCHOR/FREE construction, the seven garment anchors and
4+4 quota, pattern QUIET/EXPRESSIVE construction and 2+2 quota, all scenarios,
and all vocabularies are unchanged. `configs/config.py` contained no color-tier
comment and therefore required no change.

The exhaustive color search contains
`C(13,3) × C(10,3) = 34,320` disjoint assignments per profile. Generating and
validating all 24 in-memory profile variants took about 1.96 seconds; the full
profile-generator command took about 2.15 seconds. No candidate cap was needed.
Five legacy RNG draws are deliberately advanced per profile so removing the
old color-anchor choices does not perturb the next profile's garment
assignment. Direct comparison confirmed that all 24 v4 garment and pattern
preference lists equal v3.

## Reference versus measured results

| Metric | v3 | v4 guide | v4 measured |
| --- | ---: | ---: | ---: |
| colors appearing in preferences | 9 / 13 | **13 / 13** | **13 / 13** |
| users per appearing color | 16 each | 10–12 | **10–12** |
| live `(user, scenario)` combinations | 1,416 | **1,414** | **1,414** |
| total option plans | 3,340 | ~3,024 | **3,027** |
| garment-preference plans | 461 | ~432 | **435** |
| dress garment top-3 / uniform | 2.59x | ~2.64x | **2.71x** |
| color-violation surface-cue accuracy | 69.2% | ~69% | **71.6%** |

The plan and garment-active counts differ from the guide by only three. The
dress garment concentration differs by 0.07x. These small shifts come from the
exact seeded color triples changing which dress-code queries retain a neutral
color or pattern violation pair, after which the unchanged planner expands
queries over their feasible violation axes.

The color-violation surface-cue diagnostic predicts which member of each
`(compatible, incompatible)` color pair is compatible using that value's
global compatible-role rate. It rose 2.4 percentage points because the feasible
color-violation set changed from 610 v3 plans to 549 v4 plans; no scenario rule
or planner heuristic changed.

## Color assignment balance

Every color appears on both preference sides. `users` is the number of profiles
containing the color in either side; likes and dislikes are disjoint within a
profile.

| color | users | likes | dislikes |
| --- | ---: | ---: | ---: |
| beige | 11 | 6 | 5 |
| black | 12 | 6 | 6 |
| blue | 10 | 5 | 5 |
| brown | 12 | 6 | 6 |
| gray | 11 | 5 | 6 |
| green | 11 | 6 | 5 |
| navy | 11 | 6 | 5 |
| orange | 11 | 5 | 6 |
| pink | 10 | 5 | 5 |
| purple | 11 | 5 | 6 |
| red | 11 | 5 | 6 |
| white | 12 | 6 | 6 |
| yellow | 11 | 6 | 5 |

## Non-live user/scenario combinations

There are exactly two non-live combinations among 24 users × 59 canonical
scenarios:

1. `U007 × stage_greenscreen_shoot` — a garment preference pair exists
   (`leggings` liked, `sweater` disliked), but both incompatible colors
   (`green`, `blue`) are liked, so color cannot supply a preference-neutral
   violation. Pattern also cannot supply it because its only compatible value,
   `solid`, is liked.
2. `U019 × stage_tv_interview` — a garment preference pair exists, but both
   incompatible colors (`green`, `white`) are disliked. Pattern cannot supply
   the violation because its only compatible value, `solid`, is liked.

These are the two accepted coverage losses from removing the color reserve;
no replacement tier was introduced.

## Validation and confound audit

- Profiles: 24.
- Generated queries: 2,942; represented query IDs in plans: 2,611.
- Option plans: 3,027 (physical 1,152; dress-code 1,875).
- Construction validation: 0 structural/preference failures and 0 TPO
  failures.
- Mutation test: all 8 observed `(track, active axis, violation axis)`
  combinations × 6 mutation classes = 48/48 detected.
- Full-set preference-blind active-value accuracy: 0.540 overall
  (color 0.577, pattern 0.510, garment 0.526).
- Active-value-prior-matched subset: 2,186 / 3,027 = 72.2%; its matched prior
  is 0.50 by construction.
- Dress garment-preference concentration: 77 realized pairs, top three 10.6%,
  or 2.71x the uniform top-three baseline. Per-user garment pair counts are
  min/median/max 8/10/12, with average/worst top-pair share 17%/25%.

## Hashes and determinism

The complete seed-42 profile, query, and option pipeline was generated twice.
The following SHA256 values were identical after the forced second generation:

```text
profiles.jsonl     1a760a7d7ba445e30ab583b1f3f5986f5ef50a45b994bb619eb33033dec30aa9
queries.jsonl      357efb3301c1e74fea7d785ae8a248dfb1c0f8f16b00bf4f99478d12d908efc8
option_plans.jsonl 727a79804e79f5def16dc3543f5ecebe8e4d04982842bc4c90ce252f8078982f
```

Historical v3 hashes remain:

```text
profiles.jsonl     c91f848ca634e4a787841e71036a254f93d0fae4c78cc534d3409bb8e837c7dd
queries.jsonl      a85832fec48ad5fd8adff06066ae90d4ab3d88e306e8cd527492f5aff2056864
option_plans.jsonl fa0a5b3fb8f903b1d01669c936b014c37d9d683f842003bb9f08f5ae067a7982
```
