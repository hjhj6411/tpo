"""
Profile Generator v5 — deterministic preference-only narratives.

This version deliberately avoids demographic or biographical context.
Narratives are generated from the structured like/dislike attributes only.
No age, occupation, gender, nationality, lifestyle, or inferred identity is mentioned.

Updated for the cleaned garment/pattern vocabulary:
- formal_shirt is rendered as "formal shirt"
- leopard is rendered as "leopard print"
- underscore labels are rendered as human-readable fashion terms

── v5: `solid` is the BASELINE of the pattern axis, not a pattern value ──────

Colour and garment have no null level: clothing is always *some* garment in
*some* colour. The pattern axis is different — a plain garment has no pattern,
and `solid` is the name of that absence. In factorial terms `solid` is the
control condition, not a fifth treatment level.

v4 treated it as an ordinary value and gave every user a like or a dislike over
it (the QUIET tier drew 1 like + 1 dislike from {solid, striped}). Two things
broke as a result, and both were structural rather than incidental:

1. **No preference-neutral plain option existed for anyone.** The background
   axis — the third axis, held identical across A/B/C/D — must be neutral for
   that user and safe for the scenario. Its pattern candidates were
   {solid, striped, checkered} (the loud patterns are excluded on purpose so
   the background can never become a second visual cue). Every user had both
   `solid` and `striped` as preference values, so users who also owned
   `checkered` had an empty candidate pool. 799 plans ended up with no pattern
   at all, which makes an exact `color|garment|pattern` image key impossible;
   six scenarios lost every item.

2. **Pattern could never be the TPO axis in formal scenarios.** A violation
   pair needs a compatible-AND-neutral value and an incompatible-AND-neutral
   value. Formal dress codes admit essentially {solid, striped}, both of which
   were preference values for every user, so the compatible-neutral side was
   empty for all 24 profiles. Measured on v4: of 679 candidate queries, exactly
   0 could put the violation on pattern. The benchmark could not express
   "leopard print is wrong at a funeral" — its single most representative
   contrast.

v5 therefore assigns preference over the five PATTERNED values only and leaves
`solid` neutral for everyone. The quota is unchanged (2 likes + 2 dislikes, two
neutral values per user); what changes is that the assignment vocabulary is 5
values instead of 6 and that one of each user's two neutral patterns is always
`solid`. `solid` is also compatible in all 29 pattern-constrained scenarios and
has 100% image availability, so it needs no special-casing downstream.

Cost, disclosed rather than hidden: in the 13 scenarios whose compatible
pattern set is {solid, striped} (or {solid}), pattern can no longer be the
PREFERENCE axis, because a like/dislike pair cannot be formed from one value.
That is a property of the situation — where there is no room to choose a
pattern there is no room to express a pattern taste — and the same scenarios
gain the pattern-violation items that v4 could not construct at all.

PATTERN_QUIET / PATTERN_EXPRESSIVE are retained as inactive constants so the
v4 tiers stay readable next to the data they produced. They must not be used
by the v5 assignment path.
"""

import argparse
import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from .utils import save_jsonl, load_jsonl, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import PROFILES_DIR, PHASE1_CONFIG, FASHION_ATTRIBUTE_AXES
from configs.profiles import get_all_variants
from configs.scenarios import (
    CANONICAL_SCENARIOS,
    FRAME_SCOPED_ARCHETYPES,
    PROFILE_GENERATION_SCENARIOS,
)


AXIS_LABELS = {
    "garment_category": "garment categories",
    "color": "colors",
    "pattern": "patterns",
}

VALUE_DISPLAY_NAMES = {
    # garments
    "t_shirt": "t-shirt",
    "long_sleeve_t_shirt": "long-sleeve t-shirt",
    "tank_top": "tank top",
    "formal_shirt": "formal shirt",
    "sweatshirt": "sweatshirt",
    "sweater": "sweater",
    "hoodie": "hoodie",
    "cardigan": "cardigan",
    "blazer": "blazer",
    "windbreaker": "windbreaker",
    "leather_jacket": "leather jacket",
    "puffer_jacket": "puffer jacket",
    "fleece_jacket": "fleece jacket",
    "pea_coat": "pea coat",
    "long_coat": "long coat",
    "suit_vest": "suit vest",
    "polo_shirt": "polo shirt",
    "jeans": "jeans",
    "slacks": "slacks",
    "shorts": "shorts",
    "leggings": "leggings",
    "dress": "dress",
    "mini_skirt": "mini skirt",
    "long_skirt": "long skirt",
    # patterns
    "solid": "solid",
    "striped": "striped",
    "checkered": "checkered",
    "floral": "floral print",
    "polka_dot": "polka dot",
    "leopard": "leopard print",
}


def flatten_to_keywords(structured):
    likes, dislikes = [], []
    for axis, prefs in structured.items():
        for v in prefs.get("likes", []):
            likes.append(f"{axis}:{v}")
        for v in prefs.get("dislikes", []):
            dislikes.append(f"{axis}:{v}")
    return likes, dislikes


def _pretty_value(value: str) -> str:
    return VALUE_DISPLAY_NAMES.get(value, value.replace("_", " "))


def _join_values(values):
    values = [_pretty_value(v) for v in values]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _group_keywords(keywords):
    grouped = {}
    for kw in keywords:
        if ":" not in kw:
            continue
        axis, value = kw.split(":", 1)
        grouped.setdefault(axis, []).append(value)
    return grouped


def _preference_clause(grouped):
    clauses = []
    for axis in ["garment_category", "color", "pattern"]:
        values = grouped.get(axis, [])
        if values:
            clauses.append(f"{AXIS_LABELS.get(axis, axis)} such as {_join_values(values)}")
    return "; ".join(clauses)


def build_preference_narrative(likes, dislikes):
    """Build a deterministic, list-style narrative with no demographics."""
    like_text = _preference_clause(_group_keywords(likes))
    dislike_text = _preference_clause(_group_keywords(dislikes))

    sentences = []
    if like_text:
        sentences.append(f"This user likes {like_text}.")
    else:
        sentences.append("This user has no explicitly listed fashion likes.")

    if dislike_text:
        sentences.append(f"This user dislikes {dislike_text}.")
    else:
        sentences.append("This user has no explicitly listed fashion dislikes.")

    return " ".join(sentences)


# Backward-compatible alias for older code paths.
def narrative_fallback(likes, dislikes):
    return build_preference_narrative(likes, dislikes)


# ════════════════════════════════════════════════════════════════════
# Axis-quota rules (R1-R5) — see docs/redesign_v2_plan.md §5
#
# The garment tiers are DERIVED from the scenario catalog, then validated
# against the fixed sets below so catalog drift fails loudly instead of
# silently breaking the by-construction guarantees.
#   ANCHOR  — compatible across every normative-formal coded scenario
#             -> garment takes 2 likes + 2 dislikes (R1)
#   FREE    — the remaining garments; garment takes 2+2 subject to R4
#
# Color has no tiers in wacv_scenario_v4: all 13 colors participate in one
# globally balanced 3-like / 3-dislike assignment subject to R5. Pattern keeps
# its QUIET/EXPRESSIVE split unchanged.
# ════════════════════════════════════════════════════════════════════

NORMATIVE_FORMAL_ARCHETYPES = FRAME_SCOPED_ARCHETYPES - {"club_code"}

# Widened 2026-07-24 from the blazer/formal_shirt/dress trio. With only three
# anchors and one like + one dislike per profile, each user had exactly ONE
# formal fallback pair, so the dress-code garment preference axis collapsed onto
# the three anchor-trio matchups (45% of the axis, per-user worst 87%). The
# strictest scenarios (black-tie gala, opera premiere) admitted nothing else at
# all. Seven anchors at 2+2 leave three neutral per user, preserving formal
# fallback supply while widening the global pair vocabulary. Requires the compat-side widening of
# the 17 normative-formal scenarios (docs/redesign_v2_plan.md §16) — S1 below
# fails loudly if that widening is reverted.
GARMENT_ANCHOR = [
    "blazer", "formal_shirt", "dress", "slacks", "long_skirt",
    "suit_vest", "long_coat",
]
# `solid` is the BASELINE level of the pattern axis, not a pattern value.
# Unlike color and garment, the pattern axis has a null level: a plain garment
# has no pattern. v4 treated that null as an ordinary value and gave every user
# a like/dislike over it, which left no preference-neutral plain option for any
# user. Preference is therefore assigned over the five PATTERNED values only,
# and `solid` stays neutral for every user (full rationale in the module
# docstring).
PATTERN_BASELINE = "solid"
PATTERN_PREFERENCE = ["striped", "checkered", "floral", "polka_dot", "leopard"]

# ── inactive: the v4 two-tier pattern split ───────────────────────────────
# Kept so the rule that produced data_wacv_scenario_v4/ stays readable next to
# the data. QUIET drew 1 like + 1 dislike from {solid, striped} for every user
# — the assignment over the baseline that v5 removed — and EXPRESSIVE drew
# 1 + 1 from the four loud patterns. Neither list is read by the v5 assignment
# path; do not reintroduce them there.
PATTERN_QUIET = ["solid", "striped"]
PATTERN_EXPRESSIVE = ["floral", "checkered", "polka_dot", "leopard"]

# Historical exceptions used only to reproduce the frozen seed-42 profiles.
# These scenarios are excluded from the active evaluation catalog.
PROFILE_BOUNDARY_SCENARIOS = {
    "club_tennis_whites", "stage_backstage_crew",
    "sports_home_game", "sports_derby_match", "sports_school_rivalry",
}

def derive_and_validate_tiers():
    """Validate the garment tier and return (garment_free, all_colors)."""
    norm = [s for s in PROFILE_GENERATION_SCENARIOS
            if s.get("archetype") in NORMATIVE_FORMAL_ARCHETYPES]
    if not norm:
        raise ValueError("no normative-formal scenarios found in catalog")

    g_common = set(FASHION_ATTRIBUTE_AXES["garment_category"])
    for s in norm:
        g_common &= set(s["garment_category"]["compatible"])

    if not set(GARMENT_ANCHOR) <= g_common:
        raise ValueError(
            f"S1 violated: garment ANCHOR {GARMENT_ANCHOR} not compatible in "
            f"every normative-formal scenario (common={sorted(g_common)})")

    garment_free = [g for g in FASHION_ATTRIBUTE_AXES["garment_category"]
                    if g not in GARMENT_ANCHOR]
    color_free = list(FASHION_ATTRIBUTE_AXES["color"])
    return garment_free, color_free


# Style pools removed (2026-07-16): they partitioned the FREE vocabulary
# into fixed like/dislike sides per pool, which starved whole A/B matchups
# (a pair can only appear if some profile has one side liked and the other
# disliked). FREE picks are now assigned globally: least-used-first over the
# WHOLE free vocabulary, so every value lands on both sides of the ledger
# across the 24 profiles and the feasible pair space stays dense.
#
# Formal-compatible FREE garments: coded (formal) scenarios admit mostly
# these beyond the ANCHOR trio. Each profile is steered to carry >=1 of
# them on each side so formal scenarios admit non-ANCHOR garment pairs
# (otherwise the garment axis collapses to blazer/formal_shirt/dress
# matchups — observed 70% of the axis).
FORMAL_FREE_GARMENTS = ["sweater", "cardigan", "pea_coat"]

# Post-generation manual touch-ups from profile inspection. SWAPS exchange a
# value between two variants, so every global like/dislike tally — and the
# balanced A/B pair distributions built on them — stays byte-identical.
# Applied before validation: R1-R5 re-run on the touched result. The value
# asserts make this fail loudly if the seed (and thus variant contents)
# ever changes.
# 2026-07-16 inspection: variants 9/20 (U010/U021) shared identical garment
# AND color likes — a duplicated-taste twin. Two swaps broke the twin.
# 2026-07-24: emptied because the ANCHOR 5x(2+2) change alters every variant, so
# the old swaps are stale by construction (their value asserts would fire).
# Re-run twin inspection on the new seed-42 output and re-author if a twin
# reappears.
MANUAL_SWAPS = []


def _apply_manual_swaps(variants):
    for key, ia, va, ib, vb in MANUAL_SWAPS:
        da, db = variants[ia][2], variants[ib][2]
        assert va in da[key] and vb in db[key], (
            f"manual swap stale: {key} {va}@{ia} / {vb}@{ib} — regenerate "
            f"inspection swaps for the current seed")
        assert vb not in da[key] and va not in db[key], (
            f"manual swap no-op/conflict: {key} {va}@{ia} / {vb}@{ib}")
        da[key][da[key].index(va)] = vb
        db[key][db[key].index(vb)] = va
    return variants

def _balanced_choice(candidates, use_counter, rank, rng):
    """Least-globally-used first; pool rank then rng as tie-breakers."""
    return min(candidates,
               key=lambda v: (use_counter[v],
                              rank.index(v) if v in rank else len(rank),
                              rng.random()))


def _garment_pairs_alive_everywhere(likes, dislikes):
    """R4 predicate: every scenario keeps >=1 preference-neutral garment on
    BOTH its compatible and incompatible sides."""
    prefs = set(likes) | set(dislikes)
    for s in PROFILE_GENERATION_SCENARIOS:
        gc = s["garment_category"]
        if not (set(gc["compatible"]) - prefs):
            return False
        if not (set(gc["incompatible"]) - prefs):
            return False
    return True


def _cells_alive_everywhere(g_likes, g_dis, c_likes, c_dis, p_likes, p_dis):
    """R5 predicate for the frozen profile-design catalog.

    Every non-boundary cell admits an A/B on at least one axis. The historical
    boundary IDs are excluded from the active evaluation catalog, where all
    user-scenario cells are required to be live.
    """
    likes = {"garment_category": set(g_likes), "color": set(c_likes),
             "pattern": set(p_likes)}
    dis = {"garment_category": set(g_dis), "color": set(c_dis),
           "pattern": set(p_dis)}
    for s in PROFILE_GENERATION_SCENARIOS:
        if s["scenario_id"] in PROFILE_BOUNDARY_SCENARIOS:
            continue
        alive = False
        for axis in ("garment_category", "color", "pattern"):
            cons = s.get(axis)
            if cons is None:
                alive = True
                break
            comp = set(cons["compatible"])
            if likes[axis] & comp and dis[axis] & comp:
                alive = True
                break
        if not alive:
            return False
    return True


def generate_rule_variants(seed=42, n_variants=24):
    """Generate preference variants — pool-free, globally balanced.

    R1 garment: 2 ANCHOR likes + 2 ANCHOR dislikes (balanced); FREE 2+2 by
                exhaustive enumeration over the WHOLE free vocabulary,
                preferring combos with >=1 formal-compatible FREE garment
                on each side (keeps coded scenarios' garment pairs off the
                ANCHOR trio), then least global use, subject to R4.
    R2 color:   3 likes + 3 dislikes over all 13 colors, least-globally-used
                across profiles and subject to R5. There are no color tiers.
    R3 pattern: 2 likes + 2 dislikes over the five PATTERNED values,
                least-globally-used first, alternating by profile index which
                side draws first. `solid` is the axis baseline and is never
                assigned, so every user has it plus one patterned value as
                their preference-neutral patterns, supplying the
                violation/background values.
    R4:         FREE garment picks must keep every scenario's
                preference-neutral garment pair alive (physical included).

    Returns list of (pool_id, variant_idx, variant_dict) triples,
    matching configs.profiles.get_all_variants() output shape
    (pool_id is the constant "global" now that pools are gone).
    """
    garment_free, color_free = derive_and_validate_tiers()
    formal_free = [g for g in FORMAL_FREE_GARMENTS if g in garment_free]

    rng = random.Random(seed)
    use = {k: Counter() for k in
           ("ga_like", "ga_dis", "gf_like", "gf_dis",
            "c_like", "c_dis")}
    out = []

    # R3 pattern: global least-used-first counters over PATTERN_PREFERENCE.
    # These consume no draws from `rng`, so the garment and color assignments
    # below are bit-identical to what they would be without them.
    pat_like_use = Counter(); pat_dis_use = Counter(); pat_use = Counter()

    for idx in range(n_variants):
        # ---- R1 garment: ANCHOR 2+2 (balanced), 3 anchors left neutral ----
        # Three anchors stay preference-neutral for this user, which is what
        # keeps a neutral TPO-compatible garment available in the strictest
        # scenarios (same ANCHOR-arithmetic guarantee the 3x(1+1) design relied
        # on, now with two anchors on each preference side).
        ga_likes = []
        for _ in range(2):
            ga_likes.append(_balanced_choice(
                [g for g in GARMENT_ANCHOR if g not in ga_likes],
                use["ga_like"], [], rng))
        ga_dislikes = []
        for _ in range(2):
            ga_dislikes.append(_balanced_choice(
                [g for g in GARMENT_ANCHOR
                 if g not in ga_likes and g not in ga_dislikes],
                use["ga_dis"], [], rng))

        # ---- R1 garment: FREE 2+2 with R4 (exhaustive, balanced-first) ----
        # Enumerate every (like-pair, dislike-pair) combo over the whole
        # FREE vocabulary; walk them preferring formal-coverage on both
        # sides, then least-globally-used; take the first that keeps every
        # scenario's preference-neutral garment pair alive. Exhaustive, so
        # this only raises when no valid assignment exists at all.
        r2 = random.Random(seed * 7919 + idx * 101)
        combos = []
        for lp in combinations(garment_free, 2):
            dis_pool = [g for g in garment_free if g not in lp]
            for dp in combinations(dis_pool, 2):
                load = (use["gf_like"][lp[0]] + use["gf_like"][lp[1]]
                        + use["gf_dis"][dp[0]] + use["gf_dis"][dp[1]])
                f_gap = ((not set(lp) & set(formal_free))
                         + (not set(dp) & set(formal_free)))
                combos.append((f_gap, load, r2.random(), lp, dp))
        combos.sort()
        for _, _, _, lp, dp in combos:
            likes = ga_likes + list(lp)
            dislikes = ga_dislikes + list(dp)
            if _garment_pairs_alive_everywhere(likes, dislikes):
                gf_likes, gf_dis = list(lp), list(dp)
                break
        else:
            raise RuntimeError(
                f"R4 failed for variant {idx}: no FREE garment "
                f"assignment keeps all scenario pools alive")

        # ---- R3 pattern: 2+2 over the five PATTERNED values ----
        # `solid` (the baseline) is never assigned. Selection is the same
        # tier-free, least-used-first global assignment v4 already uses for
        # color. Consumes no RNG draws, so garment/color draws are unchanged.
        p_likes, p_dis = [], []
        # alternate which side draws first so neither side is systematically
        # constrained by the other's picks
        order = ["like", "dis", "like", "dis"] if idx % 2 == 0 else \
                ["dis", "like", "dis", "like"]
        for side in order:
            cand = [v for v in PATTERN_PREFERENCE if v not in p_likes and v not in p_dis]
            if side == "like":
                pick = min(cand, key=lambda v: (pat_like_use[v], pat_use[v], v))
                p_likes.append(pick); pat_like_use[pick] += 1
            else:
                pick = min(cand, key=lambda v: (pat_dis_use[v], pat_use[v], v))
                p_dis.append(pick); pat_dis_use[pick] += 1
            pat_use[pick] += 1

        # ---- R2 color: unrestricted 3+3 with R5 (exhaustive) ----
        # The old 3-anchor color split consumed five draws from the shared RNG
        # per profile (three like candidates, then two dislike candidates).
        # Preserve those draws so this color-only v4 change does not perturb
        # the next profile's garment assignment.
        for _ in range(5):
            rng.random()

        c_combos = []
        for lp in combinations(color_free, 3):
            dis_pool = [c for c in color_free if c not in lp]
            for dp in combinations(dis_pool, 3):
                # minimise total load first, then the PEAK load of any single
                # colour: sum alone lets one colour drift ahead of the rest
                # whenever R5 rejects the flattest combos.
                loads = ([use["c_like"][c] for c in lp]
                         + [use["c_dis"][c] for c in dp])
                load = (sum(loads), max(loads))
                c_combos.append((load, r2.random(), lp, dp))
        c_combos.sort()
        for _, _, lp, dp in c_combos:
            if _cells_alive_everywhere(likes, dislikes,
                                       list(lp), list(dp),
                                       p_likes, p_dis):
                c_likes, c_dis = list(lp), list(dp)
                break
        else:
            raise RuntimeError(
                f"R5 failed for variant {idx}: no color assignment "
                f"keeps every non-boundary (user, scenario) cell alive")

        for cnt, vals in (("ga_like", ga_likes), ("ga_dis", ga_dislikes),
                          ("gf_like", gf_likes), ("gf_dis", gf_dis),
                          ("c_like", c_likes), ("c_dis", c_dis)):
            for v in vals:
                use[cnt][v] += 1

        out.append(("global", idx, {
            "garment_likes": likes,
            "garment_dislikes": dislikes,
            "color_likes": c_likes,
            "color_dislikes": c_dis,
            "pattern_likes": p_likes,
            "pattern_dislikes": p_dis,
        }))
    return out


def validate_rule_profiles(variants):
    """Assert R1-R5 hold for the generated set; return a summary string."""
    expr_like = Counter()
    expr_dis = Counter()
    for arch_id, var_idx, v in variants:
        gl, gd = set(v["garment_likes"]), set(v["garment_dislikes"])
        cl, cd = set(v["color_likes"]), set(v["color_dislikes"])
        pl, pd = set(v["pattern_likes"]), set(v["pattern_dislikes"])
        tag = f"{arch_id}/{var_idx}"
        assert not gl & gd and not cl & cd and not pl & pd, f"{tag}: overlap"
        # Garment carries 4+4 (ANCHOR 2 + FREE 2 per side); color stays 3+3
        # but is drawn freely from the full vocabulary.
        # The asymmetry is deliberate — it buys formal garment pair diversity —
        # and must be disclosed in the paper's methodology.
        assert len(gl) == len(gd) == 4 and len(cl) == len(cd) == 3, f"{tag}: quota"
        assert len(gl & set(GARMENT_ANCHOR)) == 2, f"{tag}: R1 like quota"
        assert len(gd & set(GARMENT_ANCHOR)) == 2, f"{tag}: R1 dislike quota"
        assert len(pl) == len(pd) == 2, f"{tag}: R3 quota"
        assert PATTERN_BASELINE not in pl and PATTERN_BASELINE not in pd, \
            f"{tag}: R3 baseline must stay preference-neutral"
        assert pl <= set(PATTERN_PREFERENCE) and pd <= set(PATTERN_PREFERENCE), \
            f"{tag}: R3 patterned-only quota"
        assert _garment_pairs_alive_everywhere(v["garment_likes"],
                                               v["garment_dislikes"]), f"{tag}: R4"
        assert _cells_alive_everywhere(
            v["garment_likes"], v["garment_dislikes"],
            v["color_likes"], v["color_dislikes"],
            v["pattern_likes"], v["pattern_dislikes"]), f"{tag}: R5"
        for x in pl: expr_like[x] += 1
        for x in pd: expr_dis[x] += 1
    lo_l, hi_l = min(expr_like.values()), max(expr_like.values())
    lo_d, hi_d = min(expr_dis.values()), max(expr_dis.values())
    assert hi_l - lo_l <= 1 and hi_d - lo_d <= 1, \
        f"pattern assignment unbalanced: like={dict(expr_like)} dis={dict(expr_dis)}"
    assert expr_like[PATTERN_BASELINE] == 0 and expr_dis[PATTERN_BASELINE] == 0, \
        "baseline pattern must never be assigned"
    return (f"R1-R5 OK: {len(variants)} variants, "
            f"pattern like {dict(expr_like)} / dislike {dict(expr_dis)}, "
            f"baseline '{PATTERN_BASELINE}' neutral for all")


def generate_profile(user_idx, pool_id, variant_idx, variant,
                     provider_override=None):
    """Generate one profile record from a preference variant."""
    structured = {
        "garment_category": {
            "likes": variant["garment_likes"],
            "dislikes": variant["garment_dislikes"],
        },
        "color": {
            "likes": variant["color_likes"],
            "dislikes": variant["color_dislikes"],
        },
        "pattern": {
            "likes": variant["pattern_likes"],
            "dislikes": variant["pattern_dislikes"],
        },
    }
    likes, dislikes = flatten_to_keywords(structured)
    narrative = build_preference_narrative(likes, dislikes)

    # NOTE: the style pool (archetype) is generation-internal — it seeds
    # persona-coherent sampling and will guide chat-log generation later,
    # but it is NOT part of the user definition, must never reach an
    # evaluated model's input, and is saved to a sidecar file instead.
    return {
        "user_id": f"U{user_idx:03d}",
        "domain": "fashion",
        "structured_attributes": structured,
        "likes_keywords": likes,
        "dislikes_keywords": dislikes,
        "narrative_profile": narrative,
    }, {
        "user_id": f"U{user_idx:03d}",
        "style_pool": pool_id,
        "variant_index": variant_idx,
    }


def run_pipeline(n_users, output_path, force=False, provider=None,
                 legacy=False, seed=42):
    mode = "legacy hand-authored variants" if legacy else "axis-quota rules R1-R5"
    log_step(f"Profile Generator v5 — {mode}, {n_users} users")

    if output_path.exists() and not force:
        existing = load_jsonl(output_path)
        if len(existing) >= n_users:
            print(f"  Already have {len(existing)} profiles.")
            return existing
        profiles = existing
        start_idx = len(existing)
    else:
        profiles = []
        start_idx = 0

    if legacy:
        all_variants = get_all_variants()
    else:
        all_variants = _apply_manual_swaps(generate_rule_variants(seed=seed))
        print(f"  {validate_rule_profiles(all_variants)}")
    if n_users > len(all_variants):
        print(f"  WARNING: {n_users} users requested but only "
              f"{len(all_variants)} variants defined. Capping.")
        n_users = len(all_variants)

    gen_meta = []
    for i in range(start_idx, n_users):
        pool_id, var_idx, variant = all_variants[i]
        print(f"\n  [{i+1}/{n_users}] style_pool={pool_id}, variant={var_idx}")
        try:
            p, meta = generate_profile(i + 1, pool_id, var_idx, variant,
                                       provider_override=provider)
            profiles.append(p)
            gen_meta.append(meta)
            print(f"    {p['user_id']}")
            print(f"    likes:    {p['likes_keywords']}")
            print(f"    dislikes: {p['dislikes_keywords']}")
            print(f"    narrative: {p['narrative_profile'][:100]}...")
            if (i + 1) % 5 == 0:
                save_jsonl(profiles, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")

    save_jsonl(profiles, output_path)
    meta_path = output_path.with_name(output_path.stem + "_generation_meta.jsonl")
    save_jsonl(gen_meta, meta_path)
    print(f"\n  ✓ Saved {len(profiles)} profiles (+ style-pool sidecar {meta_path.name})")

    from .compatibility import get_compatible_instances, print_compatibility_report
    instances, stats = get_compatible_instances(profiles)
    print_compatibility_report(stats)
    print(f"  Total compatible instances: {len(instances)}")
    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int,
                        default=PHASE1_CONFIG["n_users_total"])
    parser.add_argument("--output", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--provider", type=str, default=None,
                        help="Kept for CLI compatibility; ignored by deterministic generator.")
    parser.add_argument("--legacy", action="store_true",
                        help="Use the hand-authored configs/profiles.py variants "
                             "instead of the axis-quota rules (pre-v2 behavior).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(args.n_users, args.output, args.force, args.provider,
                 legacy=args.legacy, seed=args.seed)


if __name__ == "__main__":
    main()
