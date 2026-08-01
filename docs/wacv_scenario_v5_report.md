# `wacv_scenario_v5` construction report

Generated with seed 42 on 2026-08-01 from the v4 code/data baseline. The
released v3 and v4 files and their hashes remain unchanged.

Scope: Stages 1–3 (profiles → queries → option plans) plus one scenario-catalog
correction. No GPU work, no image re-collection, no re-annotation, and no image
folder materialization is part of this variant.

## Scope and rationale

v4 constructed 3,027 option plans without knowing which `color|garment|pattern`
cells a human had actually approved. Measured against the completed annotation
(`annotation/attribute_library.json`, 1,237 available cells), only **1,012
plans (33.4%)** could supply an image for all four options. Two independent
defects produced that number.

### Defect 1 — the availability constraint ran at the wrong end of the pipeline

The planner optimized counterbalance, confusable-pair avoidance and per-user
variety over the full attribute grid, and image availability was applied
afterwards as a filter. Losing 2,015 plans is not the real cost; the real cost
is that **the loss was not random**. The constructed track split 38:62 became
31:69 among survivors, and the physical preference axis went 50:50 → 42:58.
Every balance property the planner had been asked to enforce was silently
undone by the drop.

v5 moves the same constraint to the front: the option planner reads the cell
library as a **generation input** and a candidate is only *scored* if all four
of its options land on an available cell. Because the filter is a candidate
constraint rather than a post-hoc drop, the balance objectives keep optimizing
inside the feasible region. This introduces no new restriction — "only garments
that exist in the Amazon corpus" was already binding in v4; it merely becomes
visible to the code that has to balance around it.

### Defect 2 — `solid` was treated as a pattern value instead of a baseline

Color and garment have no null level. Pattern does: a plain garment has no
pattern, and `solid` is the name of that absence. v4 gave every user a like or
a dislike over it, which is the factorial-design error of entering the control
condition as a treatment level. Two structural consequences followed.

**803 plans had no image key.** The background axis (the third axis, identical
across A/B/C/D) must be preference-neutral and situation-safe. Its pattern
candidates were `{solid, striped, checkered}` — loud patterns are excluded on
purpose so the background can never act as a second visual cue. Because the
v4 QUIET tier drew 1 like + 1 dislike from `{solid, striped}`, **all 24 users
held both**, so any user who also held `checkered` had an empty pool. 799 plans
carried no pattern and 4 carried no color, and six scenarios lost every item:
`mourn_funeral`, `mourn_memorial`, `civic_court_appearance`,
`celebration_graduation`, `stage_greenscreen_shoot`, `stage_tv_interview`.

**Pattern could never be the TPO axis in a formal scenario.** A violation pair
needs a compatible-and-neutral value and an incompatible-and-neutral value.
Formal dress codes admit essentially `{solid, striped}`, both preference values
for every user, so the compatible-neutral side was empty for all 24 profiles.
Measured on the released v4 data: **0 of 3,027 plans put the violation on
pattern in any of the 14 formal scenarios.** The benchmark structurally could
not express "leopard print is wrong at a funeral" — its single most
representative contrast.

v5 assigns pattern preference over the five patterned values only and holds
`solid` neutral for everyone. The quota is unchanged (2 likes + 2 dislikes, two
neutral values per user); only the assignment vocabulary shrinks from 6 to 5,
and one of each user's two neutral patterns is now always `solid`. `solid`
needs no special-casing: it is compatible in all 29 pattern-constrained
scenarios and its image availability is 100% (272/272), where the loud patterns
v4 used as backgrounds sat at 57% (leopard) and 60% (polka dot).

### Two smaller changes

- **Scenario-level pair balance.** `assign_ab_values` already rotated (A, B)
  pairs per user, but nothing stopped many users from receiving the *same*
  pair inside one scenario. A `scen_pair` counter is inserted between the
  per-user and the global term: `user_pair → scen_pair → pair_use → per-user
  value use → net → random`. The global term deliberately stays behind the
  per-user terms; promoting it collapses individual users onto a single A
  value, which was observed previously.
- **`wedding_reception` pattern constraint.** v7 filed `floral` as incompatible
  alongside `leopard`. Floral guest attire is an ordinary choice at a US
  wedding reception and is standard advice in spring and summer wedding
  etiquette; there is no conventional basis for grouping it with loud animal
  print. `floral` moves to compatible and `polka_dot` to incompatible. This is
  the same correction v10 applied to the four business scenarios and golf, and
  it aligns the scenario with its neighbour `social_gallery_opening`. Only
  `wedding_reception` is touched.

  The rationale is conventional, not statistical. The coverage effect is a
  consequence, never a motive: reversing that order would read as "the sample
  was thin so the convention was reclassified", which is exactly the defence
  this benchmark cannot afford to lose.

## Reference versus measured results

Every v4 figure below is recomputed from the released v4 data against the same
cell library, not copied from an earlier report.

| Metric | v4 | v5 target | v5 measured |
| --- | ---: | ---: | ---: |
| option plans | 3,027 | ~2,641 | **2,641** |
| four options all image-available | 1,012 (33.4%) | ~2,571 (97%↑) | **2,571 (97.3%)** |
| scenarios with 0 usable plans | 6 | 0 | **0** |
| scenarios under 12 users | 24 | 0 | **0** |
| scenarios covering all 24 users | 3 / 59 | 33 / 59 ↑ | **39 / 59** |
| physical preference axis color : pattern | 42 : 58 | ~55 : 45 | **57 : 43** |
| pattern as TPO axis in formal scenarios | 0 | 250 ↑ | **162** |

The v4 column for the last five rows is measured on its 1,012 image-usable
plans, since a scenario with no realizable item is empty in practice regardless
of how many text-only plans reference it.

**The last row misses its stated target and the discrepancy is in the target,
not the build.** The three released files are byte-identical to the supplied
reference artifacts, so every metric here is by definition the reference
implementation's own output. The whole dataset contains 296 pattern-violation
plans; 162 of them fall in the 14 formal scenarios (155 under the narrower
definition "scenarios whose compatible pattern set is a subset of
`{solid, striped}`", which after the wedding_reception correction numbers 13).
A count of 250 within that subset is not reachable. The structural claim the
row exists to support does hold: the count was exactly **0** in v4 and is
non-zero in every one of the 14 scenarios in v5.

## Integrity

| Check | Requirement | Result |
| --- | --- | --- |
| `validate_options` structural / preference failures | 0 | **0 / 2,641** |
| `validate_options` TPO failures | 0 | **0 / 2,641** |
| `test_option_validator_mutations` | 48/48 detected | **48 / 48** |
| pipeline run twice, three SHA256s | identical | **identical** |
| `solid` appearing as a preference A/B value | 0 | **0** |
| background value not preference-neutral | 0 | **0** |
| background value differing across A–D | 0 | **0** |

The mutation test covers 6 mutation classes over 8 plans spanning every
realized `(track, active axis, violation axis)` combination. A validator that
cannot fail is not evidence, which is why it is run on every release.

## Scale and composition

| | Physical | Dress-code |
| --- | ---: | ---: |
| scenarios / archetypes | 24 / 7 | 35 / 13 |
| option plans | 1,010 | 1,631 |
| preference axis | color 572 (56.6%), pattern 438 (43.4%) | color 665 (40.8%), pattern 624 (38.3%), garment 342 (21.0%) |
| TPO violation axis | garment 1,010 (100%) | garment 882 (54.1%), color 453 (27.8%), pattern 296 (18.1%) |
| four options image-available | 977 (96.7%) | 1,594 (97.7%) |

- 24 profiles; 2,561 queries generated; 2,270 query contexts represented in
  plans; 2,641 plans. `plan_id`, never `query_id`, is the item key.
- Query types: explicit 1,552, implicit 1,089.
- Preference-blind active-value accuracy over the full set: **0.544**
  (color 0.555, pattern 0.513, garment 0.601).
- Active-value-prior-matched subset: **1,982 / 2,641 = 75.0%**, of which 1,933
  are image-complete. On that subset a preference-blind guess at the active-axis
  value is 0.50 by construction — that one prior only, not a confound-free pool.

### Scenario coverage

No scenario is empty and none falls below 17 users.

| Users covered | Scenarios |
| ---: | ---: |
| 24 | 39 |
| 23 | 8 |
| 22 | 2 |
| 21 | 5 |
| 20 | 2 |
| 19 | 1 |
| 17 | 2 |

The six scenarios that had zero usable plans in v4:

| Scenario | Plans | Users |
| --- | ---: | ---: |
| `stage_tv_interview` | 38 | 21 |
| `stage_greenscreen_shoot` | 33 | 21 |
| `civic_court_appearance` | 27 | 23 |
| `mourn_funeral` | 27 | 23 |
| `mourn_memorial` | 25 | 21 |
| `celebration_graduation` | 19 | 17 |

1,401 of the 1,416 user × scenario combinations produce at least one query and
1,362 produce at least one plan. The 15 query-level holes are:

`U004×club_yacht_regatta`, `U004×social_company_gala`,
`U005×wedding_reception`, `U006×wedding_reception`, `U007×stage_tv_interview`,
`U010×wedding_reception`, `U012×stage_tv_interview`, `U015×wedding_reception`,
`U016×wedding_reception`, `U021×wedding_reception`,
`U022×stage_greenscreen_shoot`, `U024×celebration_graduation`,
`U024×civic_court_appearance`, `U024×mourn_funeral`, `U024×mourn_memorial`.

## Profile assignment balance

Pattern preference is now assigned over the five patterned values by global
least-used-first selection, alternating which side draws first by profile index
so neither side is systematically constrained by the other's picks. The
generator asserts a global deviation of at most 1 per side.

| pattern | users | likes | dislikes |
| --- | ---: | ---: | ---: |
| `solid` | **0** | **0** | **0** |
| striped | 19 | 9 | 10 |
| checkered | 19 | 10 | 9 |
| floral | 20 | 10 | 10 |
| polka_dot | 19 | 10 | 9 |
| leopard | 19 | 9 | 10 |

Color assignment keeps the v4 rule (no ANCHOR tier, 3+3 from all 13 colors)
with one tie-break refinement: candidate triples are ordered by total load and
then by the **peak** load of any single color, because minimizing the sum alone
let one color drift ahead whenever the cell-liveness predicate rejected the
flattest combinations.

| color | users | likes | dislikes |
| --- | ---: | ---: | ---: |
| beige | 12 | 6 | 6 |
| black | 12 | 6 | 6 |
| blue | 10 | 5 | 5 |
| brown | 13 | 6 | 7 |
| gray | 11 | 6 | 5 |
| green | 14 | 7 | 7 |
| navy | 12 | 6 | 6 |
| orange | 10 | 5 | 5 |
| pink | 10 | 5 | 5 |
| purple | 10 | 5 | 5 |
| red | 9 | 5 | 4 |
| white | 11 | 5 | 6 |
| yellow | 10 | 5 | 5 |

Garment quotas, anchors and construction are unchanged from v4.

## The 70 plans without four images

Availability is enforced at candidate time, so the exceptions are few and each
has a named cause. Neither was patched with a default value.

| Cause | Plans |
| --- | ---: |
| background color axis left unfixed (no scenario-compatible neutral color for that user) | 12 |
| background pattern pinned to `solid` after the availability check had run against the query's original background pattern | 58 |

The first is the intended behaviour from `docs/PROCESS.md` §6: when
`scenario-compatible ∩ user-neutral` is empty on the background color axis, the
axis stays unfixed and the plan is simply not image-realizable. Inventing a
value there would put a non-neutral constant in front of the user. All 12
belong to `U002`, `U008` and `U024` in color-constrained formal scenarios.

The second is a real ordering gap in the planner and should be closed in a
later variant, not this one — closing it changes the plan set and therefore the
released hashes. `assign_garment_pairs` and the violation-value assigners test
availability using the background pattern recorded on the *query*, while
`plan_option_variant` afterwards overwrites that background with `solid` under
`--solid-baseline`. When the two disagree the checked cell is not the built
cell. All 58 cases have this shape (`checkered→solid` 41, `striped→solid` 7,
`floral→solid` 6, `polka_dot→solid` 4), and all are color-active or
garment-active plans whose violation axis is garment or color. Threading the
baseline override into the pre-assignment checks would recover an estimated
2.2% of plans.

## Limitations that cannot be fixed by balancing

These four were measured, not assumed. Stating them with their causes is a
better defence than hiding them.

1. **Pattern cannot be the preference axis in the 13 plain-pattern scenarios.**
   Their compatible set is `{solid, striped}` or `{solid}`, and `solid` is
   neutral for everyone, so a like/dislike pair cannot be formed. This is a
   property of the situation: where there is no room to choose a pattern there
   is no room to express a pattern taste — the same logic as the benchmark's
   "tank top in winter" case. Pattern preference is measured in the scenarios
   that leave the choice open. In exactly these scenarios pattern instead
   carries the *norm*, producing items v4 could not construct at all.

2. **The pattern violation axis converges on `solid → X`** (`leopard/solid` 47%,
   `polka_dot/solid` 39%, `floral/solid` 7% of 296 plans). Every one of the
   affected queries has `solid` as its only compatible-and-neutral pattern, so
   no balancing term has an alternative to spread onto. This follows from
   formal dress codes admitting essentially one pattern, and is a property of
   the domain rather than an artifact of the data.

3. **The dress-code color preference axis has a 7.7% modal pair**
   (`navy/white`, 51 of 665). It concentrates on five users whose alternative
   feasible pairs averaged 1.5. Formal palettes are narrow, so the expressible
   preference contrast is narrow with them.

4. **Per-axis quotas are deliberately asymmetric**: garment 4+4 of 23, color
   3+3 of 13, pattern 2+2 of 5 patterned values with `solid` held as baseline.
   Vocabulary size and the presence of a null level differ by axis. This is
   also why per-axis metrics are never pooled into a single number.

## Hashes and determinism

The complete seed-42 profile, query and option pipeline was generated twice.
The following SHA256 values were identical after the forced second generation:

```text
profiles.jsonl     7d1cda17eccf3d73337b50bc7ed36f63e96214193930a9ea37760e187a0afd71
queries.jsonl      cc87f1c2281a52d9fe7f28676358a85963e4a17428b9eb619776210528f27671
option_plans.jsonl ad86fd50f72f46b508f2be58ead3c2e47878ce88c2ff93ef82f628530d97ccfd
```

`profiles.jsonl` and `queries.jsonl` match the reference implementation byte for
byte. `option_plans.jsonl` did too, at
`813e9ab11d955794aa8eaf0b389970dc12166db8c094b124f1d8442773241f29`, until the
garment-vocabulary correction in `docs/PROCESS.md` §8 changed 624
`search_query` strings from `wool coat` to `long coat`. Nothing else moved: the
same 2,641 plans, the same track split, the same axis distributions, the same
validator results. **The option-plan hash is provisional** until the 139
re-collected `long_coat` / `pea_coat` cells are re-annotated and the plans are
rebuilt against the new library.

Stage 3 now has a generation input, pinned to the same standard:

```text
annotation/attribute_library.json  72cc8665d6f92d143f850b751bba1767c342a2e9b857292e6738666bea86baae
```

Reproduce and verify with:

```bash
bash scripts/verify_release.sh wacv_scenario_v5
```

which checks the input hash, rebuilds into `data_wacv_scenario_v5_verify/`,
compares the three output hashes, runs the validator and runs the mutation
test. Reproducibility is a property of HEAD: v5 changed the pattern-preference
rule in `construction/profile_generator.py`, so v3 and v4 can no longer be
rebuilt bit-identically from this tree. Their released directories stay as
historical artifacts.

Historical v4 hashes remain:

```text
profiles.jsonl     1a760a7d7ba445e30ab583b1f3f5986f5ef50a45b994bb619eb33033dec30aa9
queries.jsonl      357efb3301c1e74fea7d785ae8a248dfb1c0f8f16b00bf4f99478d12d908efc8
option_plans.jsonl 727a79804e79f5def16dc3543f5ecebe8e4d04982842bc4c90ce252f8078982f
```

Historical v3 hashes remain in `docs/wacv_scenario_v4_report.md`.
