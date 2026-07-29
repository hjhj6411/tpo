# `wacv_scenario_v3` construction report

Generated with seed 42 on 2026-07-29. The released v2 files and their hashes
remain unchanged.

## Scenario review

The 73 profile-generation scenarios (59 canonical + 14 profile-only) were
reviewed. No scenario needed a departure from the requested starting
assignment:

- `suit_vest` follows `blazer`. Court and mourning contexts keep it compatible
  as a tailored layer worn with formal clothing, not as a complete outfit by
  itself.
- `long_coat` follows `blazer`; it is the formal wool outerwear label.
- `pea_coat` inherits the former `trench_coat` status, including compatibility
  in the two mourning and four religious scenarios.
- `polo_shirt` follows `t_shirt`, except that it is compatible in
  `biz_client_meeting`, `club_golf_round`, `club_yacht_regatta`, and the three
  garden scenarios. It remains inappropriate for board meetings, investor
  pitches, job interviews, court, mourning, and other formal ceremonies.
- `sweater` and `cardigan` were added to the six requested semi-formal
  scenarios.

Scenario constraints remain intentionally partial: omitted garments are
neutral, rather than silently prohibited. Every scenario constraint is
disjoint and vocabulary-valid, and the 73-scenario catalog collectively covers
all 23 garments.

## Retrieval and verifier vocabulary

`wool coat` is the retrieval rendering for `long_coat`. In the frozen 651k
FashionSigLIP corpus text, `wool coat` occurred 70 times versus 2 for
`long wool coat`; this check selected a retrieval phrase and was not used to
assign image labels.

Garment verification is exact-match by default. True-synonym expansion is
available only with `--allow-garment-equivalence`. The opt-in groups keep
`pea_coat` separate from `long_coat`, keep `suit_vest` separate from fleece and
puffer vests, and do not merge sweater/cardigan or hoodie/sweatshirt. Pattern
anchors are derived from `configs.config`; `plaid` is not an anchor.

## Measurements

| Metric | v2 | sim8 guide | v3 measured |
| --- | ---: | ---: | ---: |
| garment-preference plans | 478 | ~437 | 461 |
| dress garment top-3 / uniform | 3.70x | ~2.4x | 2.59x |
| largest dress garment pair | 8.4% | ~3.4% | 3.5% |
| per-user garment pairs (min/median/max) | 6/8/11 | 8/10/13 | 8/10/12 |
| physical violation garment top-3 / uniform | 2.89x | ~2.34x | 2.30x |
| total plans | 3,340 | ~3,317 | 3,340 |

The 23-plan difference from the approximate sim8 total is small (0.7%).
The deterministic seven-anchor regeneration yielded 461 rather than about 437
garment-active plans; the remaining axes total 2,879 plans. Compared with v2,
v3 has 17 fewer garment-active, 16 more color-active, and one more
pattern-active plan. The largest garment-active shifts are the two club
scenarios (-17 together) and ultra-formal scenarios (+4), reflecting the new
profile assignments and reviewed compatibility pools.

Blind active-value exploit accuracy is 0.512 overall (color 0.508, pattern
0.501, garment 0.563). The active-value-prior-matched subset is 2,672 / 3,340
(80.0%).

New-label usage counts below are `(plans containing label / option slots)`.
On a garment-active plan a selected value occupies two slots.

| label | garment-active | all plans |
| --- | ---: | ---: |
| `suit_vest` | 96 / 192 | 330 / 718 |
| `long_coat` | 88 / 176 | 318 / 698 |
| `pea_coat` | 32 / 64 | 216 / 434 |
| `polo_shirt` | 16 / 32 | 403 / 910 |

`polo_shirt` is well represented overall but comparatively sparse as the
active preference value (16 plans, with A/B counts 13/3); this should be
reported in garment-axis analyses rather than hidden.

## Validation and hashes

- Construction validation: 0 structural/preference failures and 0 TPO
  failures across 3,340 plans.
- Mutation test: all 8 observed `(track, active axis, violation axis)`
  combinations x 6 mutation classes = 48/48 detected.
- A clean seed-42 regeneration was byte-identical for all three artifacts.

```text
profiles.jsonl     c91f848ca634e4a787841e71036a254f93d0fae4c78cc534d3409bb8e837c7dd
queries.jsonl      a85832fec48ad5fd8adff06066ae90d4ab3d88e306e8cd527492f5aff2056864
option_plans.jsonl fa0a5b3fb8f903b1d01669c936b014c37d9d683f842003bb9f08f5ae067a7982
```

Historical v2 hashes (preserved):

```text
profiles.jsonl     6642f47a850acbe63b5f916b4c246a092c6ca49ff45b725387df7e5849d7c68f
queries.jsonl      fc1c355332d76131fae7558cdcc2a308951cdc39817e92d581ae0576ce468cf5
option_plans.jsonl 8922488f2b952c357c1c64fa1c3e20d3bec343103e3a7cd96a64c981a976a3cf
```
