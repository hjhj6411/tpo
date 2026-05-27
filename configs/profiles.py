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
"""

# ═══════════════════════════════════════════════════════════
#  GARMENT PREFERENCE ARCHETYPES (7 archetypes × 3 users = 21)
# ═══════════════════════════════════════════════════════════
#
#  Each archetype defines a template. Within an archetype, 3 users
#  share the same functional-group pattern but differ in specific items
#  and in their color/pattern preferences.
#
#  Fields:
#    garment_likes / garment_dislikes : concrete items
#    color_likes / color_dislikes     : concrete values
#    pattern_likes / pattern_dislikes : concrete values
#    persona_hint : fed to narrative LLM for natural language generation

PREFERENCE_ARCHETYPES = [
    # ── 1. Classic Formal ─────────────────────────────────
    {
        "archetype_id": "classic_formal",
        "persona_hint": "Prefers timeless formal pieces; avoids bulky or overly casual items.",
        "variants": [
            {
                "garment_likes": ["trench_coat", "blazer"],
                "garment_dislikes": ["parka", "hoodie", "shorts"],
                "color_likes": ["navy", "beige"],
                "color_dislikes": ["gray", "orange"],
                "pattern_likes": ["solid"],
                "pattern_dislikes": ["graphic_print", "camouflage"],
            },
            {
                "garment_likes": ["coat", "shirt"],
                "garment_dislikes": ["windbreaker", "t_shirt", "tank_top"],
                "color_likes": ["black", "white"],
                "color_dislikes": ["navy", "yellow"],
                "pattern_likes": ["striped"],
                "pattern_dislikes": ["animal_print", "plaid"],
            },
            {
                "garment_likes": ["blazer", "blouse"],
                "garment_dislikes": ["parka", "tank_top", "shorts"],
                "color_likes": ["white", "gray"],
                "color_dislikes": ["brown", "pink"],
                "pattern_likes": ["solid", "striped"],
                "pattern_dislikes": ["polka_dot", "graphic_print"],
            },
        ],
    },
    # ── 2. Casual Sporty ──────────────────────────────────
    {
        "archetype_id": "casual_sporty",
        "persona_hint": "Prefers comfortable athletic-leaning pieces; dislikes rigid formal wear.",
        "variants": [
            {
                "garment_likes": ["hoodie", "t_shirt", "shorts"],
                "garment_dislikes": ["suit_jacket", "trench_coat", "blazer"],
                "color_likes": ["black", "blue"],
                "color_dislikes": ["beige", "orange"],
                "pattern_likes": ["solid"],
                "pattern_dislikes": ["floral", "animal_print"],
            },
            {
                "garment_likes": ["tank_top", "shorts", "jacket"],
                "garment_dislikes": ["coat", "blazer", "dress"],
                "color_likes": ["gray", "green"],
                "color_dislikes": ["navy", "pink"],
                "pattern_likes": ["solid", "striped"],
                "pattern_dislikes": ["plaid", "camouflage"],
            },
            {
                "garment_likes": ["t_shirt", "hoodie"],
                "garment_dislikes": ["suit_jacket", "trench_coat", "blouse"],
                "color_likes": ["white", "red"],
                "color_dislikes": ["brown", "yellow"],
                "pattern_likes": ["graphic_print"],
                "pattern_dislikes": ["floral", "checkered"],
            },
        ],
    },
    # ── 3. Minimalist ─────────────────────────────────────
    {
        "archetype_id": "minimalist",
        "persona_hint": "Favors clean, understated pieces; avoids loud or bulky items.",
        "variants": [
            {
                "garment_likes": ["coat", "shirt", "pants"],
                "garment_dislikes": ["parka", "hoodie", "tank_top"],
                "color_likes": ["black", "white"],
                "color_dislikes": ["orange", "purple"],
                "pattern_likes": ["solid"],
                "pattern_dislikes": ["graphic_print", "animal_print"],
            },
            {
                "garment_likes": ["trench_coat", "blouse"],
                "garment_dislikes": ["windbreaker", "hoodie", "shorts"],
                "color_likes": ["gray", "beige"],
                "color_dislikes": ["red", "yellow"],
                "pattern_likes": ["solid", "striped"],
                "pattern_dislikes": ["camouflage", "polka_dot"],
            },
            {
                "garment_likes": ["blazer", "jeans"],
                "garment_dislikes": ["parka", "tank_top", "dress"],
                "color_likes": ["navy", "white"],
                "color_dislikes": ["pink", "green"],
                "pattern_likes": ["solid"],
                "pattern_dislikes": ["floral", "graphic_print"],
            },
        ],
    },
    # ── 4. Adventurous Outdoor ────────────────────────────
    {
        "archetype_id": "adventurous_outdoor",
        "persona_hint": "Gravitates toward rugged outdoor gear; dislikes formal or delicate items.",
        "variants": [
            {
                "garment_likes": ["parka", "windbreaker", "jeans"],
                "garment_dislikes": ["blazer", "dress", "suit_jacket"],
                "color_likes": ["green", "brown"],
                "color_dislikes": ["black", "pink"],
                "pattern_likes": ["solid"],
                "pattern_dislikes": ["floral", "polka_dot"],
            },
            {
                "garment_likes": ["jacket", "hoodie", "shorts"],
                "garment_dislikes": ["trench_coat", "blouse", "suit_jacket"],
                "color_likes": ["navy", "beige"],
                "color_dislikes": ["purple", "orange"],
                "pattern_likes": ["plaid"],
                "pattern_dislikes": ["animal_print", "graphic_print"],
            },
            {
                "garment_likes": ["coat", "windbreaker"],
                "garment_dislikes": ["suit_jacket", "blazer", "skirt"],
                "color_likes": ["gray", "blue"],
                "color_dislikes": ["red", "yellow"],
                "pattern_likes": ["solid", "checkered"],
                "pattern_dislikes": ["floral", "camouflage"],
            },
        ],
    },
    # ── 5. Elegant ────────────────────────────────────────
    {
        "archetype_id": "elegant",
        "persona_hint": "Favors graceful, polished pieces; avoids overly casual or sporty items.",
        "variants": [
            {
                "garment_likes": ["dress", "blouse", "coat"],
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
                "pattern_likes": ["striped"],
                "pattern_dislikes": ["plaid", "animal_print"],
            },
            {
                "garment_likes": ["trench_coat", "dress"],
                "garment_dislikes": ["t_shirt", "tank_top", "windbreaker"],
                "color_likes": ["beige", "white"],
                "color_dislikes": ["gray", "red"],
                "pattern_likes": ["solid", "floral"],
                "pattern_dislikes": ["graphic_print", "camouflage"],
            },
        ],
    },
    # ── 6. Streetwear ─────────────────────────────────────
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
                "pattern_dislikes": ["floral", "plaid"],
            },
            {
                "garment_likes": ["t_shirt", "windbreaker"],
                "garment_dislikes": ["blazer", "coat", "dress"],
                "color_likes": ["white", "green"],
                "color_dislikes": ["navy", "orange"],
                "pattern_likes": ["graphic_print"],
                "pattern_dislikes": ["striped", "animal_print"],
            },
            {
                "garment_likes": ["hoodie", "shorts", "jacket"],
                "garment_dislikes": ["suit_jacket", "trench_coat", "skirt"],
                "color_likes": ["gray", "blue"],
                "color_dislikes": ["pink", "brown"],
                "pattern_likes": ["solid", "camouflage"],
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
                "color_dislikes": ["orange", "gray"],
                "pattern_likes": ["solid"],
                "pattern_dislikes": ["camouflage", "animal_print"],
            },
            {
                "garment_likes": ["jacket", "pants", "blouse"],
                "garment_dislikes": ["coat", "hoodie", "shorts"],
                "color_likes": ["navy", "white"],
                "color_dislikes": ["yellow", "purple"],
                "pattern_likes": ["striped", "solid"],
                "pattern_dislikes": ["graphic_print", "plaid"],
            },
            {
                "garment_likes": ["shirt", "jeans", "sweater"],
                "garment_dislikes": ["trench_coat", "blazer", "tank_top"],
                "color_likes": ["black", "green"],
                "color_dislikes": ["pink", "red"],
                "pattern_likes": ["solid"],
                "pattern_dislikes": ["floral", "polka_dot"],
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
