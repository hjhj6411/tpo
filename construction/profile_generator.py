"""
Profile Generator v4 — deterministic preference-only narratives.

This version deliberately avoids demographic or biographical context.
Narratives are generated from the structured like/dislike attributes only.
No age, occupation, gender, nationality, lifestyle, or inferred identity is mentioned.

Updated for the cleaned garment/pattern vocabulary:
- formal_shirt is rendered as "formal shirt"
- leopard is rendered as "leopard print"
- underscore labels are rendered as human-readable fashion terms
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
# Tier-quota rules (R1-R4) — see docs/redesign_v2_plan.md §5
#
# Vocabulary tiers are DERIVED from the scenario catalog, then validated
# against the fixed sets below so catalog drift fails loudly instead of
# silently breaking the by-construction guarantees.
#   ANCHOR  — compatible across every normative-formal coded scenario
#             -> garment takes 2 likes + 2 dislikes (R1), color 1+1 (R2)
#   RESERVE — never assigned to preferences; supplies TPO-violation
#             values (color axis only — pattern has no RESERVE tier:
#             every pattern may carry preference, and per-user neutrality
#             for violations/backgrounds comes from the 2 EXPRESSIVE
#             values each profile leaves unassigned)
#   FREE    — persona variety, sampled from anonymous style pools
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
COLOR_ANCHOR = ["black", "navy", "gray"]
COLOR_RESERVE = ["orange", "yellow", "pink", "white"]   # hard-RESERVE
PATTERN_QUIET = ["solid", "striped"]                     # 1 like + 1 dislike split
# Expressive patterns: 1 like + 1 dislike per profile, balanced globally
# (each value 6x like / 6x dislike across 24 variants). The 2 unassigned
# values stay preference-neutral per user, supplying that user's
# TPO-violation and background values — so unlike color there is no hard
# RESERVE, and every pattern can appear in A/B pairs somewhere.
PATTERN_EXPRESSIVE = ["floral", "checkered", "polka_dot", "leopard"]

# scenarios exempt from the S2 check (documented in the design doc):
# safety_visibility's incompatible side is guaranteed by ANCHOR arithmetic
# instead; greenscreen's {green, blue} is instruction-scoped (explicit seeds).
# festive_bright forbids exactly the 3 ANCHOR colors {black,navy,gray}: R2
# gives every user exactly one anchor like + one anchor dislike, so exactly one
# anchor is always neutral -> a preference-neutral violation color is always
# available (same ANCHOR-arithmetic guarantee as safety_visibility).
S2_EXCEPTION_SCENARIOS = {
    "visib_night_road_run", "visib_night_bike_commute",
    "visib_dawn_roadside_cleanup", "stage_greenscreen_shoot",
    "festive_holiday_party", "festive_bright_theme_party", "festive_street_fiesta",
}
# (sports_spirit's incompatible team/rival colors are hard-RESERVE, so they
#  satisfy S2 directly and need no exception.)

# Historical exceptions used only to reproduce the frozen seed-42 profiles.
# These scenarios are excluded from the active evaluation catalog.
PROFILE_BOUNDARY_SCENARIOS = {
    "club_tennis_whites", "stage_backstage_crew",
    "sports_home_game", "sports_derby_match", "sports_school_rivalry",
}

def derive_and_validate_tiers():
    """Recompute tiers from the catalog and assert they match the rule
    constants. Returns (garment_free, color_free)."""
    norm = [s for s in PROFILE_GENERATION_SCENARIOS
            if s.get("archetype") in NORMATIVE_FORMAL_ARCHETYPES]
    if not norm:
        raise ValueError("no normative-formal scenarios found in catalog")

    g_common = set(FASHION_ATTRIBUTE_AXES["garment_category"])
    c_common = set(FASHION_ATTRIBUTE_AXES["color"])
    for s in norm:
        g_common &= set(s["garment_category"]["compatible"])
        c = s.get("color")
        if c and c.get("incompatible"):
            c_common &= set(c["compatible"])

    if not set(GARMENT_ANCHOR) <= g_common:
        raise ValueError(
            f"S1 violated: garment ANCHOR {GARMENT_ANCHOR} not compatible in "
            f"every normative-formal scenario (common={sorted(g_common)})")
    if not set(COLOR_ANCHOR) <= c_common:
        raise ValueError(
            f"color ANCHOR {COLOR_ANCHOR} not universally compatible "
            f"(common={sorted(c_common)})")

    for s in PROFILE_GENERATION_SCENARIOS:  # S2: violation-value supply guaranteed
        c = s.get("color")
        if (c and c.get("incompatible")
                and s["scenario_id"] not in S2_EXCEPTION_SCENARIOS
                and not set(c["incompatible"]) & set(COLOR_RESERVE)):
            raise ValueError(
                f"S2 violated: {s['scenario_id']} incompatible colors "
                f"{c['incompatible']} contain no hard-RESERVE color")

    garment_free = [g for g in FASHION_ATTRIBUTE_AXES["garment_category"]
                    if g not in GARMENT_ANCHOR]
    color_free = [c for c in FASHION_ATTRIBUTE_AXES["color"]
                  if c not in COLOR_ANCHOR + COLOR_RESERVE]
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
    R2 color:   1 ANCHOR like + 1 dislike (balanced); FREE 2+2
                least-globally-used over the whole FREE color vocabulary;
                RESERVE colors never assigned.
    R3 pattern: solid/striped alternating 1+1 + 1 like / 1 dislike from the
                EXPRESSIVE set (balanced ordered pairs — every value 6x like
                and 6x dislike; the 2 unassigned expressive values stay
                preference-neutral per user, supplying violation/background
                values).
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
            "ca_like", "ca_dis", "cf_like", "cf_dis")}
    out = []

    # R3 expressive: the 12 ordered (like, dislike) pairs, each twice across
    # the 24 variants -> every expressive value lands exactly 6x as like and
    # 6x as dislike.
    expr_pairs = [(l, d) for l in PATTERN_EXPRESSIVE
                  for d in PATTERN_EXPRESSIVE if l != d] * 2
    random.Random(seed + 4649).shuffle(expr_pairs)

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

        # ---- R3 pattern: quiet 1+1 split + expressive 1+1 ----
        # (chosen before color so the R5 cell-liveness predicate can see
        # the full variant when picking the color combo)
        quiet_like = "solid" if idx % 2 == 0 else "striped"
        quiet_dis = "striped" if quiet_like == "solid" else "solid"
        e_like, e_dis = expr_pairs[idx % len(expr_pairs)]
        p_likes = [quiet_like, e_like]
        p_dis = [quiet_dis, e_dis]

        # ---- R2 color: ANCHOR 1+1 + FREE 2+2 with R5 (exhaustive) ----
        ca_like = _balanced_choice(COLOR_ANCHOR, use["ca_like"], [], rng)
        ca_dis = _balanced_choice(
            [c for c in COLOR_ANCHOR if c != ca_like],
            use["ca_dis"], [], rng)
        c_combos = []
        for lp in combinations(color_free, 2):
            dis_pool = [c for c in color_free if c not in lp]
            for dp in combinations(dis_pool, 2):
                load = (use["cf_like"][lp[0]] + use["cf_like"][lp[1]]
                        + use["cf_dis"][dp[0]] + use["cf_dis"][dp[1]])
                c_combos.append((load, r2.random(), lp, dp))
        c_combos.sort()
        for _, _, lp, dp in c_combos:
            if _cells_alive_everywhere(likes, dislikes,
                                       [ca_like] + list(lp), [ca_dis] + list(dp),
                                       p_likes, p_dis):
                cf_likes, cf_dis = list(lp), list(dp)
                break
        else:
            raise RuntimeError(
                f"R5 failed for variant {idx}: no FREE color assignment "
                f"keeps every non-boundary (user, scenario) cell alive")

        for cnt, vals in (("ga_like", ga_likes), ("ga_dis", ga_dislikes),
                          ("gf_like", gf_likes), ("gf_dis", gf_dis),
                          ("ca_like", [ca_like]), ("ca_dis", [ca_dis]),
                          ("cf_like", cf_likes), ("cf_dis", cf_dis)):
            for v in vals:
                use[cnt][v] += 1

        out.append(("global", idx, {
            "garment_likes": likes,
            "garment_dislikes": dislikes,
            "color_likes": [ca_like] + cf_likes,
            "color_dislikes": [ca_dis] + cf_dis,
            "pattern_likes": p_likes,
            "pattern_dislikes": p_dis,
        }))
    return out


def validate_rule_profiles(variants):
    """Assert R1-R4 hold for the generated set; return a summary string."""
    reserve_c = set(COLOR_RESERVE)
    quiet_split = Counter()
    expr_like = Counter()
    expr_dis = Counter()
    for arch_id, var_idx, v in variants:
        gl, gd = set(v["garment_likes"]), set(v["garment_dislikes"])
        cl, cd = set(v["color_likes"]), set(v["color_dislikes"])
        pl, pd = set(v["pattern_likes"]), set(v["pattern_dislikes"])
        tag = f"{arch_id}/{var_idx}"
        assert not gl & gd and not cl & cd and not pl & pd, f"{tag}: overlap"
        # garment carries 4+4 (ANCHOR 2 + FREE 2 per side); color stays 3+3.
        # The asymmetry is deliberate — it buys formal garment pair diversity —
        # and must be disclosed in the paper's methodology.
        assert len(gl) == len(gd) == 4 and len(cl) == len(cd) == 3, f"{tag}: quota"
        assert len(gl & set(GARMENT_ANCHOR)) == 2, f"{tag}: R1 like quota"
        assert len(gd & set(GARMENT_ANCHOR)) == 2, f"{tag}: R1 dislike quota"
        assert len(cl & set(COLOR_ANCHOR)) == 1, f"{tag}: R2 like quota"
        assert len(cd & set(COLOR_ANCHOR)) == 1, f"{tag}: R2 dislike quota"
        assert not (cl | cd) & reserve_c, f"{tag}: R2 RESERVE color leak"
        assert len(pl) == len(pd) == 2, f"{tag}: R3 quota"
        assert len(pl & set(PATTERN_QUIET)) == 1 and len(pd & set(PATTERN_QUIET)) == 1, \
            f"{tag}: R3 quiet split"
        assert len(pl & set(PATTERN_EXPRESSIVE)) == 1 \
            and len(pd & set(PATTERN_EXPRESSIVE)) == 1, f"{tag}: R3 expressive quota"
        assert _garment_pairs_alive_everywhere(v["garment_likes"],
                                               v["garment_dislikes"]), f"{tag}: R4"
        assert _cells_alive_everywhere(
            v["garment_likes"], v["garment_dislikes"],
            v["color_likes"], v["color_dislikes"],
            v["pattern_likes"], v["pattern_dislikes"]), f"{tag}: R5"
        quiet_split[(pl & set(PATTERN_QUIET)).pop()] += 1
        expr_like[(pl & set(PATTERN_EXPRESSIVE)).pop()] += 1
        expr_dis[(pd & set(PATTERN_EXPRESSIVE)).pop()] += 1
    assert quiet_split["solid"] == quiet_split["striped"] == len(variants) // 2, \
        f"quiet split unbalanced: {dict(quiet_split)}"
    per_val = len(variants) // len(PATTERN_EXPRESSIVE)
    for v in PATTERN_EXPRESSIVE:
        assert expr_like[v] == expr_dis[v] == per_val, \
            (f"expressive unbalanced: {v} like={expr_like[v]} "
             f"dislike={expr_dis[v]} (target {per_val})")
    return (f"R1-R4 OK: {len(variants)} variants, quiet {dict(quiet_split)}, "
            f"expressive like {dict(expr_like)} / dislike {dict(expr_dis)}")


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
    mode = "legacy hand-authored variants" if legacy else "tier-quota rules R1-R4"
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
                             "instead of the tier-quota rules (pre-v2 behavior).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(args.n_users, args.output, args.force, args.provider,
                 legacy=args.legacy, seed=args.seed)


if __name__ == "__main__":
    main()
