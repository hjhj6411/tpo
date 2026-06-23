"""
POD-Bench v2 — User Preference Archetypes (HARDCODED, 24 users)

8 preference archetypes × 3 variants = 24 users.

Two design goals, both addressed by hardcoding:

(1) MAXIMIZE compatibility across the 15 scenario archetypes.
    Each variant's garment likes (~3) and dislikes (~3) leave ~12 neutral
    garments, so for (almost) every scenario there is at least one neutral
    TPO-compatible garment and one neutral TPO-incompatible garment — the
    clean A/B (compat) vs C/D (incompat) garment pair the planner needs.

(2) ELIMINATE the `solid`-monoculture confound at the source.
    Liked patterns are spread across a compact all-over-pattern vocabulary.
    The pattern axis intentionally excludes `graphic_print` and `animal_print`
    because retrieval often returns localized front graphics/logos or motif-only
    animal cues instead of full-body garment patterns. `plaid` is folded into
    `checkered` as a retrieval alias rather than treated as a separate label.
    Population tally of pattern_likes (48 slots):
        striped 9, checkered 9, argyle 8, solid 7, floral 5,
        polka_dot 5, camouflage 5
    Population tally of color_likes (48 slots) spreads dark/neutral (~29) and
    bright (~19) so the benchmark is not all-dark-solid. Combined with the
    counterbalanced option planner, the blind value-prior collapses to ~0.50.

A NEW archetype, `bold_expressive`, deliberately likes bright colors and clear
all-over statement patterns (floral / argyle / polka_dot). Such users are still
highly compatible because, under the relaxed compatibility check, physical
scenarios (cold/heat/athletic/…) impose no color/pattern constraint — so their
bright, loud preferences are valid there, balancing the dataset away from a
dark/solid monoculture.
"""

# ═══════════════════════════════════════════════════════════
#  PREFERENCE ARCHETYPES (8 archetypes × 3 variants = 24 users)
# ═══════════════════════════════════════════════════════════

PREFERENCE_ARCHETYPES = [

    # ── 1. Classic Formal ─────────────────────────────────
    {
        "archetype_id": "classic_formal",
        "persona_hint": "Prefers timeless tailored pieces; avoids bulky or overly casual items.",
        "variants": [
            {
                "garment_likes": ["trench_coat", "blazer", "sweater"],
                "garment_dislikes": ["parka", "hoodie", "shorts"],
                "color_likes": ["navy", "beige"],
                "color_dislikes": ["orange", "red"],
                "pattern_likes": ["striped", "checkered"],
                "pattern_dislikes": ["argyle", "camouflage"],
            },
            {
                "garment_likes": ["coat", "shirt", "sweater"],
                "garment_dislikes": ["windbreaker", "tank_top", "hoodie"],
                "color_likes": ["black", "gray"],
                "color_dislikes": ["yellow", "pink"],
                "pattern_likes": ["solid", "striped"],
                "pattern_dislikes": ["camouflage", "argyle"],
            },
            {
                "garment_likes": ["blazer", "blouse", "coat"],
                "garment_dislikes": ["parka", "tank_top", "shorts"],
                "color_likes": ["white", "navy"],
                "color_dislikes": ["brown", "green"],
                "pattern_likes": ["checkered", "argyle"],
                "pattern_dislikes": ["polka_dot", "camouflage"],
            },
        ],
    },

    # ── 2. Casual Sporty ──────────────────────────────────
    {
        "archetype_id": "casual_sporty",
        "persona_hint": "Comfortable athletic-leaning pieces; dislikes rigid formal wear.",
        "variants": [
            {
                "garment_likes": ["hoodie", "t_shirt", "shorts"],
                "garment_dislikes": ["suit_jacket", "trench_coat", "blazer"],
                "color_likes": ["blue", "red"],
                "color_dislikes": ["beige", "brown"],
                "pattern_likes": ["camouflage", "solid"],
                "pattern_dislikes": ["floral", "argyle"],
            },
            {
                "garment_likes": ["tank_top", "shorts", "jacket"],
                "garment_dislikes": ["coat", "blazer", "dress"],
                "color_likes": ["green", "gray"],
                "color_dislikes": ["navy", "pink"],
                "pattern_likes": ["striped", "polka_dot"],
                "pattern_dislikes": ["floral", "camouflage"],
            },
            {
                "garment_likes": ["t_shirt", "hoodie", "jeans"],
                "garment_dislikes": ["suit_jacket", "trench_coat", "blouse"],
                "color_likes": ["white", "black"],
                "color_dislikes": ["brown", "yellow"],
                "pattern_likes": ["argyle", "checkered"],
                "pattern_dislikes": ["floral", "camouflage"],
            },
        ],
    },

    # ── 3. Minimalist ─────────────────────────────────────
    {
        "archetype_id": "minimalist",
        "persona_hint": "Clean, understated pieces; avoids loud or bulky items.",
        "variants": [
            {
                "garment_likes": ["coat", "shirt", "sweater"],
                "garment_dislikes": ["parka", "hoodie", "tank_top"],
                "color_likes": ["black", "white"],
                "color_dislikes": ["orange", "purple"],
                "pattern_likes": ["solid", "striped"],
                "pattern_dislikes": ["camouflage", "argyle"],
            },
            {
                "garment_likes": ["trench_coat", "blouse", "sweater"],
                "garment_dislikes": ["windbreaker", "hoodie", "shorts"],
                "color_likes": ["gray", "beige"],
                "color_dislikes": ["red", "yellow"],
                "pattern_likes": ["checkered", "striped"],
                "pattern_dislikes": ["camouflage", "argyle"],
            },
            {
                "garment_likes": ["blazer", "jeans", "shirt"],
                "garment_dislikes": ["parka", "tank_top", "dress"],
                "color_likes": ["navy", "white"],
                "color_dislikes": ["pink", "green"],
                "pattern_likes": ["solid", "checkered"],
                "pattern_dislikes": ["floral", "camouflage"],
            },
        ],
    },

    # ── 4. Adventurous Outdoor ────────────────────────────
    {
        "archetype_id": "adventurous_outdoor",
        "persona_hint": "Rugged outdoor gear; dislikes formal or delicate items.",
        "variants": [
            {
                "garment_likes": ["parka", "windbreaker", "jeans"],
                "garment_dislikes": ["blazer", "dress", "suit_jacket"],
                "color_likes": ["green", "brown"],
                "color_dislikes": ["pink", "purple"],
                "pattern_likes": ["checkered", "camouflage"],
                "pattern_dislikes": ["floral", "polka_dot"],
            },
            {
                "garment_likes": ["jacket", "hoodie", "shorts"],
                "garment_dislikes": ["trench_coat", "blouse", "suit_jacket"],
                "color_likes": ["navy", "beige"],
                "color_dislikes": ["purple", "orange"],
                "pattern_likes": ["checkered", "camouflage"],
                "pattern_dislikes": ["argyle", "floral"],
            },
            {
                "garment_likes": ["coat", "windbreaker", "hoodie"],
                "garment_dislikes": ["suit_jacket", "blazer", "skirt"],
                "color_likes": ["gray", "blue"],
                "color_dislikes": ["red", "yellow"],
                "pattern_likes": ["argyle", "striped"],
                "pattern_dislikes": ["floral", "polka_dot"],
            },
        ],
    },

    # ── 5. Elegant ────────────────────────────────────────
    {
        "archetype_id": "elegant",
        "persona_hint": "Graceful, polished pieces; avoids overly casual or sporty items.",
        "variants": [
            {
                "garment_likes": ["dress", "blouse", "coat"],
                "garment_dislikes": ["hoodie", "tank_top", "windbreaker"],
                "color_likes": ["black", "purple"],
                "color_dislikes": ["orange", "green"],
                "pattern_likes": ["floral", "solid"],
                "pattern_dislikes": ["camouflage", "argyle"],
            },
            {
                "garment_likes": ["blazer", "skirt", "shirt"],
                "garment_dislikes": ["parka", "hoodie", "shorts"],
                "color_likes": ["navy", "pink"],
                "color_dislikes": ["brown", "yellow"],
                "pattern_likes": ["floral", "striped"],
                "pattern_dislikes": ["camouflage", "argyle"],
            },
            {
                "garment_likes": ["trench_coat", "dress", "blouse"],
                "garment_dislikes": ["t_shirt", "tank_top", "windbreaker"],
                "color_likes": ["beige", "white"],
                "color_dislikes": ["gray", "red"],
                "pattern_likes": ["floral", "polka_dot"],
                "pattern_dislikes": ["argyle", "camouflage"],
            },
        ],
    },

    # ── 6. Streetwear ─────────────────────────────────────
    {
        "archetype_id": "streetwear",
        "persona_hint": "Urban street style; prefers hoodies, jackets, bold all-over patterns; avoids traditional formal.",
        "variants": [
            {
                "garment_likes": ["hoodie", "jacket", "jeans"],
                "garment_dislikes": ["suit_jacket", "trench_coat", "blouse"],
                "color_likes": ["black", "red"],
                "color_dislikes": ["beige", "purple"],
                "pattern_likes": ["argyle", "camouflage"],
                "pattern_dislikes": ["floral", "polka_dot"],
            },
            {
                "garment_likes": ["t_shirt", "windbreaker", "hoodie"],
                "garment_dislikes": ["blazer", "coat", "dress"],
                "color_likes": ["white", "green"],
                "color_dislikes": ["navy", "brown"],
                "pattern_likes": ["polka_dot", "striped"],
                "pattern_dislikes": ["argyle", "floral"],
            },
            {
                "garment_likes": ["hoodie", "shorts", "jacket"],
                "garment_dislikes": ["suit_jacket", "trench_coat", "skirt"],
                "color_likes": ["gray", "blue"],
                "color_dislikes": ["pink", "beige"],
                "pattern_likes": ["camouflage", "argyle"],
                "pattern_dislikes": ["floral", "polka_dot"],
            },
        ],
    },

    # ── 7. Relaxed Neutral ────────────────────────────────
    {
        "archetype_id": "relaxed_neutral",
        "persona_hint": "Easy-going, middle-of-the-road taste; avoids extremes in either direction.",
        "variants": [
            {
                "garment_likes": ["sweater", "shirt", "jeans"],
                "garment_dislikes": ["parka", "suit_jacket", "tank_top"],
                "color_likes": ["blue", "beige"],
                "color_dislikes": ["orange", "purple"],
                "pattern_likes": ["striped", "checkered"],
                "pattern_dislikes": ["camouflage", "argyle"],
            },
            {
                "garment_likes": ["jacket", "pants", "t_shirt"],
                "garment_dislikes": ["coat", "suit_jacket", "shorts"],
                "color_likes": ["navy", "white"],
                "color_dislikes": ["yellow", "pink"],
                "pattern_likes": ["checkered", "solid"],
                "pattern_dislikes": ["camouflage", "argyle"],
            },
            {
                "garment_likes": ["shirt", "jeans", "sweater"],
                "garment_dislikes": ["trench_coat", "blazer", "tank_top"],
                "color_likes": ["black", "green"],
                "color_dislikes": ["pink", "red"],
                "pattern_likes": ["solid", "argyle"],
                "pattern_dislikes": ["floral", "camouflage"],
            },
        ],
    },

    # ── 8. Bold Expressive (NEW) ──────────────────────────
    # Likes bright colors + clear all-over statement patterns; balances the
    # dataset away from a dark/solid monoculture. Highly compatible via physical
    # scenarios (which impose no color/pattern TPO constraint).
    {
        "archetype_id": "bold_expressive",
        "persona_hint": "Loves vivid colors and statement all-over patterns; avoids plain, muted looks.",
        "variants": [
            {
                "garment_likes": ["dress", "t_shirt", "skirt"],
                "garment_dislikes": ["suit_jacket", "coat", "blazer"],
                "color_likes": ["red", "yellow"],
                "color_dislikes": ["gray", "black"],
                "pattern_likes": ["floral", "argyle"],
                "pattern_dislikes": ["solid", "camouflage"],
            },
            {
                "garment_likes": ["blouse", "jeans", "jacket"],
                "garment_dislikes": ["parka", "trench_coat", "suit_jacket"],
                "color_likes": ["orange", "pink"],
                "color_dislikes": ["navy", "beige"],
                "pattern_likes": ["polka_dot", "floral"],
                "pattern_dislikes": ["solid", "camouflage"],
            },
            {
                "garment_likes": ["t_shirt", "dress", "hoodie"],
                "garment_dislikes": ["blazer", "coat", "suit_jacket"],
                "color_likes": ["purple", "green"],
                "color_dislikes": ["black", "gray"],
                "pattern_likes": ["argyle", "polka_dot"],
                "pattern_dislikes": ["solid", "striped"],
            },
        ],
    },
]


def get_all_variants():
    """Flatten all (archetype_id, variant_idx, variant_dict) triples."""
    out = []
    for arch in PREFERENCE_ARCHETYPES:
        for i, var in enumerate(arch["variants"]):
            out.append((arch["archetype_id"], i, {
                **var,
                "persona_hint": arch["persona_hint"],
            }))
    return out


if __name__ == "__main__":
    from collections import Counter
    variants = get_all_variants()
    print(f"Users: {len(variants)}")
    pc, cc = Counter(), Counter()
    overlap_errors = []
    for aid, vi, v in variants:
        for ax, like_k, dis_k in [("pattern", "pattern_likes", "pattern_dislikes"),
                                  ("color", "color_likes", "color_dislikes"),
                                  ("garment_category", "garment_likes", "garment_dislikes")]:
            ov = set(v[like_k]) & set(v[dis_k])
            if ov:
                overlap_errors.append((aid, vi, ax, sorted(ov)))
        for p in v["pattern_likes"]:
            pc[p] += 1
        for c in v["color_likes"]:
            cc[c] += 1
    print("pattern_likes tally:", dict(pc.most_common()))
    print("color_likes tally:", dict(cc.most_common()))
    print("like/dislike overlaps (should be empty):", overlap_errors)
