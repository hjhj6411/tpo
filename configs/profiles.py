"""
POD-Bench v2 — User Preference Archetypes

Profiles are backward-designed from the scenario catalog to maximize
user-scenario compatibility. Each user's dislikes span multiple garment
functional groups so that option B (disliked + TPO-compatible) can be
constructed across archetype families.

Design principle (per axis):
  - likes and dislikes each span ≥2 functional groups
  - At least 1 dislike in {heavy_outerwear} for cold-scenario B option
  - At least 1 dislike in {casual_tops, bottoms} for heat/casual B option
  - At least 1 dislike in {formal_tops, light_outerwear} for formal B option

v2.1 changes (compatibility fix):
  - classic_formal: sweater added to garment_likes (cold/casual compatible)
  - minimalist: sweater added to garment_likes (cold/casual compatible)
  - elegant: jacket added to garment_likes (athletic/outdoor compatible)
  - relaxed_neutral: hoodie or t_shirt added to garment_likes
    (athletic scenario A-option coverage)
  - pattern dislikes narrowed where they were blocking all pattern-axis
    compatible values (e.g. checkered removed from dislikes in some variants)
"""

# ═══════════════════════════════════════════════════════════
#  GARMENT PREFERENCE ARCHETYPES (7 archetypes × 3 users = 21)
# ═══════════════════════════════════════════════════════════

PREFERENCE_ARCHETYPES = [
    # ── 1. Classic Formal ─────────────────────────────────
    # v2.1: sweater added to likes in all 3 variants
    #   → enables A option in cold_ski_resort_town, cold_ice_festival, etc.
    {
        "archetype_id": "classic_formal",
        "persona_hint": "Prefers timeless formal pieces; avoids bulky or overly casual items.",
        "variants": [
            {
                "garment_likes": ["trench_coat", "blazer", "sweater"],
                "garment_dislikes": ["parka", "hoodie", "shorts"],
                "color_likes": ["navy", "beige"],
                "color_dislikes": ["gray", "orange"],
                "pattern_likes": ["solid", "striped"],
                "pattern_dislikes": ["graphic_print", "camouflage"],
            },
            {
                "garment_likes": ["coat", "shirt", "sweater"],
                "garment_dislikes": ["windbreaker", "t_shirt", "tank_top"],
                "color_likes": ["black", "white"],
                "color_dislikes": ["navy", "yellow"],
                "pattern_likes": ["striped", "checkered"],
                "pattern_dislikes": ["animal_print", "graphic_print"],
            },
            {
                "garment_likes": ["blazer", "blouse", "sweater"],
                "garment_dislikes": ["parka", "tank_top", "shorts"],
                "color_likes": ["white", "gray"],
                "color_dislikes": ["brown", "pink"],
                "pattern_likes": ["solid", "striped"],
                "pattern_dislikes": ["polka_dot", "graphic_print"],
            },
        ],
    },
    # ── 2. Casual Sporty ──────────────────────────────────
    # No change needed — already has strong athletic coverage
    {
        "archetype_id": "casual_sporty",
        "persona_hint": "Prefers comfortable athletic-leaning pieces; dislikes rigid formal wear.",
        "variants": [
            {
                "garment_likes": ["hoodie", "t_shirt", "shorts"],
                "garment_dislikes": ["suit_jacket", "trench_coat", "blazer"],
                "color_likes": ["black", "blue"],
                "color_dislikes": ["beige", "orange"],
                "pattern_likes": ["solid", "graphic_print"],
                "pattern_dislikes": ["floral", "animal_print"],
            },
            {
                "garment_likes": ["tank_top", "shorts", "jacket"],
                "garment_dislikes": ["coat", "blazer", "dress"],
                "color_likes": ["gray", "green"],
                "color_dislikes": ["navy", "pink"],
                "pattern_likes": ["solid", "striped"],
                "pattern_dislikes": ["floral", "camouflage"],
            },
            {
                "garment_likes": ["t_shirt", "hoodie", "shorts"],
                "garment_dislikes": ["suit_jacket", "trench_coat", "blouse"],
                "color_likes": ["white", "red"],
                "color_dislikes": ["brown", "yellow"],
                "pattern_likes": ["graphic_print", "solid"],
                "pattern_dislikes": ["floral", "animal_print"],
            },
        ],
    },
    # ── 3. Minimalist ─────────────────────────────────────
    # v2.1: sweater added to likes (neutral athletic/casual item)
    {
        "archetype_id": "minimalist",
        "persona_hint": "Favors clean, understated pieces; avoids loud or bulky items.",
        "variants": [
            {
                "garment_likes": ["coat", "shirt", "sweater"],
                "garment_dislikes": ["parka", "hoodie", "tank_top"],
                "color_likes": ["black", "white"],
                "color_dislikes": ["orange", "purple"],
                "pattern_likes": ["solid"],
                "pattern_dislikes": ["graphic_print", "animal_print"],
            },
            {
                "garment_likes": ["trench_coat", "blouse", "sweater"],
                "garment_dislikes": ["windbreaker", "hoodie", "shorts"],
                "color_likes": ["gray", "beige"],
                "color_dislikes": ["red", "yellow"],
                "pattern_likes": ["solid", "striped"],
                "pattern_dislikes": ["camouflage", "animal_print"],
            },
            {
                "garment_likes": ["blazer", "jeans", "sweater"],
                "garment_dislikes": ["parka", "tank_top", "dress"],
                "color_likes": ["navy", "white"],
                "color_dislikes": ["pink", "green"],
                "pattern_likes": ["solid", "checkered"],
                "pattern_dislikes": ["floral", "graphic_print"],
            },
        ],
    },
    # ── 4. Adventurous Outdoor ────────────────────────────
    # No change needed — already has strong cold/outdoor coverage
    {
        "archetype_id": "adventurous_outdoor",
        "persona_hint": "Gravitates toward rugged outdoor gear; dislikes formal or delicate items.",
        "variants": [
            {
                "garment_likes": ["parka", "windbreaker", "jeans"],
                "garment_dislikes": ["blazer", "dress", "suit_jacket"],
                "color_likes": ["green", "brown"],
                "color_dislikes": ["black", "pink"],
                "pattern_likes": ["solid", "checkered"],
                "pattern_dislikes": ["floral", "polka_dot"],
            },
            {
                "garment_likes": ["jacket", "hoodie", "shorts"],
                "garment_dislikes": ["trench_coat", "blouse", "suit_jacket"],
                "color_likes": ["navy", "beige"],
                "color_dislikes": ["purple", "orange"],
                "pattern_likes": ["plaid", "solid"],
                "pattern_dislikes": ["animal_print", "graphic_print"],
            },
            {
                "garment_likes": ["coat", "windbreaker", "hoodie"],
                "garment_dislikes": ["suit_jacket", "blazer", "skirt"],
                "color_likes": ["gray", "blue"],
                "color_dislikes": ["red", "yellow"],
                "pattern_likes": ["solid", "checkered"],
                "pattern_dislikes": ["floral", "camouflage"],
            },
        ],
    },
    # ── 5. Elegant ────────────────────────────────────────
    # v2.1: jacket added to likes in variants 0 and 2
    #   → enables A option in athletic_marathon, athletic_tennis (windbreaker/jacket compat)
    {
        "archetype_id": "elegant",
        "persona_hint": "Favors graceful, polished pieces; avoids overly casual or sporty items.",
        "variants": [
            {
                "garment_likes": ["dress", "blouse", "coat", "jacket"],
                "garment_dislikes": ["hoodie", "tank_top", "windbreaker"],
                "color_likes": ["black", "purple"],
                "color_dislikes": ["orange", "green"],
                "pattern_likes": ["solid", "floral"],
                "pattern_dislikes": ["camouflage", "graphic_print"],
            },
            {
                "garment_likes": ["blazer", "skirt", "shirt"],
                "garment_dislikes": ["parka", "hoodie", "shorts"],
                "color_likes": ["navy", "pink"],
                "color_dislikes": ["brown", "yellow"],
                "pattern_likes": ["striped", "floral"],
                "pattern_dislikes": ["camouflage", "animal_print"],
            },
            {
                "garment_likes": ["trench_coat", "dress", "jacket"],
                "garment_dislikes": ["t_shirt", "tank_top", "windbreaker"],
                "color_likes": ["beige", "white"],
                "color_dislikes": ["gray", "red"],
                "pattern_likes": ["solid", "floral"],
                "pattern_dislikes": ["graphic_print", "camouflage"],
            },
        ],
    },
    # ── 6. Streetwear ─────────────────────────────────────
    # No change needed — already has strong casual/athletic coverage
    {
        "archetype_id": "streetwear",
        "persona_hint": "Urban street style; prefers hoodies, jackets, graphic pieces; avoids traditional formal.",
        "variants": [
            {
                "garment_likes": ["hoodie", "jacket", "jeans"],
                "garment_dislikes": ["suit_jacket", "trench_coat", "blouse"],
                "color_likes": ["black", "red"],
                "color_dislikes": ["beige", "purple"],
                "pattern_likes": ["graphic_print", "solid"],
                "pattern_dislikes": ["floral", "animal_print"],
            },
            {
                "garment_likes": ["t_shirt", "windbreaker", "hoodie"],
                "garment_dislikes": ["blazer", "coat", "dress"],
                "color_likes": ["white", "green"],
                "color_dislikes": ["navy", "orange"],
                "pattern_likes": ["graphic_print", "solid"],
                "pattern_dislikes": ["striped", "animal_print"],
            },
            {
                "garment_likes": ["hoodie", "shorts", "jacket"],
                "garment_dislikes": ["suit_jacket", "trench_coat", "skirt"],
                "color_likes": ["gray", "blue"],
                "color_dislikes": ["pink", "brown"],
                "pattern_likes": ["solid", "graphic_print"],
                "pattern_dislikes": ["floral", "polka_dot"],
            },
        ],
    },
    # ── 7. Relaxed Neutral ────────────────────────────────
    # v2.1: t_shirt or hoodie added to garment_likes
    #   → enables A option in athletic scenarios
    {
        "archetype_id": "relaxed_neutral",
        "persona_hint": "Easy-going, middle-of-the-road taste; avoids extremes in either direction.",
        "variants": [
            {
                "garment_likes": ["sweater", "shirt", "jeans", "t_shirt"],
                "garment_dislikes": ["parka", "suit_jacket", "tank_top"],
                "color_likes": ["blue", "beige"],
                "color_dislikes": ["orange", "purple"],
                "pattern_likes": ["solid", "striped"],
                "pattern_dislikes": ["camouflage", "animal_print"],
            },
            {
                "garment_likes": ["jacket", "pants", "blouse", "hoodie"],
                "garment_dislikes": ["coat", "suit_jacket", "shorts"],
                "color_likes": ["navy", "white"],
                "color_dislikes": ["yellow", "pink"],
                "pattern_likes": ["striped", "solid", "checkered"],
                "pattern_dislikes": ["graphic_print", "animal_print"],
            },
            {
                "garment_likes": ["shirt", "jeans", "sweater", "t_shirt"],
                "garment_dislikes": ["trench_coat", "blazer", "tank_top"],
                "color_likes": ["black", "green"],
                "color_dislikes": ["pink", "red"],
                "pattern_likes": ["solid", "checkered"],
                "pattern_dislikes": ["floral", "camouflage"],
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