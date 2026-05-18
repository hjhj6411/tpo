"""
POD-Bench central configuration.

Design decisions (locked):
  - Domain: fashion only.
  - Profile = likes/dislikes keywords + narrative (MMPB-style).
  - All profile generation in ENGLISH (target = international venue).
  - Attribute axes: only objectively decidable axes (color, material, pattern,
    garment category, fit, sleeve length, neckline, formality).
    Style labels (minimalist, bohemian, ...) are excluded from likes/dislikes
    because they are ambiguous; they may appear in narrative for fluency only.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
PROFILES_DIR = DATA_DIR / "profiles"
QUERIES_DIR = DATA_DIR / "queries"
OPTIONS_DIR = DATA_DIR / "options"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
FINAL_DIR = DATA_DIR / "final"

for d in [PROFILES_DIR, QUERIES_DIR, OPTIONS_DIR, IMAGES_DIR, LABELS_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Fashion attribute taxonomy (8 axes, objectively decidable)
# ─────────────────────────────────────────────
# Standard sources:
#   - DeepFashion (Liu et al. CVPR 2016): 50 categories + 1000 attributes
#   - Fashionpedia (Jia et al. ECCV 2020): 27 main categories + 19 fine
#   - Berlin & Kay (1969): 11 universal basic color terms
#   - Fashion200K: brand-agnostic attribute vocabulary
#
# Each axis is *closed-set* (controlled vocabulary) for unambiguous matching.

FASHION_ATTRIBUTE_AXES = {
    "color": [
        # Berlin & Kay 11 basic terms + dark/light qualifiers common in fashion
        "black", "white", "gray", "navy", "blue", "red", "pink", "orange",
        "yellow", "green", "brown", "beige", "purple",
    ],
    "material": [
        "cotton", "linen", "wool", "silk", "cashmere", "denim", "leather",
        "suede", "polyester", "nylon", "knit", "fleece",
    ],
    "pattern": [
        "solid", "striped", "checkered", "plaid", "floral", "polka_dot",
        "graphic_print", "camouflage", "animal_print",
    ],
    "garment_category": [
        # 12 most common upper / lower / outer / one-piece categories
        "t_shirt", "shirt", "blouse", "sweater", "hoodie",
        "jacket", "coat", "trench_coat", "blazer",
        "dress", "skirt", "pants", "jeans", "shorts", "suit",
    ],
    "fit": [
        "slim", "regular", "loose", "oversized",
    ],
    "sleeve_length": [
        "sleeveless", "short_sleeve", "three_quarter_sleeve", "long_sleeve",
    ],
    "neckline": [
        "crew_neck", "v_neck", "scoop_neck", "turtleneck",
        "collared", "off_shoulder", "hooded",
    ],
    "formality_level": [
        "very_casual", "casual", "smart_casual", "business_casual", "formal",
    ],
}

# Number of likes / dislikes per axis (sampling target)
ATTRIBUTE_SAMPLING = {
    "n_likes_per_axis": (1, 2),     # min, max
    "n_dislikes_per_axis": (0, 2),  # min, max (some axes may have 0 dislikes)
    "n_axes_with_preferences": (5, 7),  # not all 8 axes need preferences
}


# ─────────────────────────────────────────────
# TPO axes for fashion (kept lean)
# ─────────────────────────────────────────────
FASHION_TPO = {
    "time": {
        "season": ["spring", "summer", "fall", "winter"],
        "weather": ["sunny", "rainy", "snowy", "hot_humid", "cold", "windy"],
        "time_of_day": ["morning", "daytime", "evening", "night"],
    },
    "place": {
        "venue": [
            "office", "cafe", "park", "beach", "mountain_hike",
            "city_street", "restaurant", "wedding_venue", "gym",
            "art_gallery", "concert_hall", "airport",
        ],
        "indoor_outdoor": ["indoor", "outdoor", "mixed"],
    },
    "occasion": {
        "activity": [
            "work_meeting", "casual_outing", "exercise", "first_date",
            "ceremony_attendance", "travel", "errands", "social_gathering",
        ],
        "formality_required": [
            "very_casual", "casual", "smart_casual", "business_casual", "formal",
        ],
    },
}


# ─────────────────────────────────────────────
# Query type ratios
# ─────────────────────────────────────────────
QUERY_TYPE_RATIO = {
    "explicit_tpo": 0.30,
    "implicit_tpo": 0.25,
    "visual_tpo": 0.30,
    "neutral": 0.15,
}


# ─────────────────────────────────────────────
# Option labels (4-choice MCQ)
# ─────────────────────────────────────────────
OPTION_LABELS = {
    "tpo_and_preference": "matches both TPO and user preference (correct)",
    "tpo_only":           "matches TPO but violates user preference",
    "preference_only":    "matches user preference but violates TPO",
    "neither":            "violates both",
}


# ─────────────────────────────────────────────
# Pipeline scale (Phase 1)
# ─────────────────────────────────────────────
PHASE1_CONFIG = {
    "n_users_total": 50,
    "n_queries_per_user": 20,
    "n_options_per_query": 4,
}


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
# Paid: GPT-5-mini only
GPT5_MINI = {
    "model_name": "gpt-5-mini",
    # Reasoning models: max_completion_tokens = reasoning + output tokens.
    # Reasoning alone consumes 300-500 tokens; set high to avoid truncation.
    "max_completion_tokens_default": 4096,
    "max_completion_tokens_long": 8192,
    # temperature parameter is NOT supported on gpt-5 family (default 1 only).
    # Setting a custom value returns 400 error.
}

# Free: local vLLM
FREE_MODELS = {
    "local_judge": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "api_base": "http://localhost:8000/v1",
    },
    "local_judge_alt": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "api_base": "http://localhost:8001/v1",
    },
    "vlm_evaluator": {
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "api_base": "http://localhost:8000/v1",
    },
}


# ─────────────────────────────────────────────
# Reliability thresholds
# ─────────────────────────────────────────────
LABEL_QUALITY_THRESHOLDS = {
    "krippendorff_alpha_min": 0.7,
    "cohen_kappa_min": 0.6,
    "judge_agreement_min": 2,  # 2 of 3 judges must agree
}

VISION_ESSENTIALITY_THRESHOLDS = {
    "blind_llm_max_acc": 0.40,
    "captioner_llm_max_acc": 0.45,
    "full_vlm_min_acc": 0.55,
}

IMAGE_COLLECTION = {
    "min_resolution": (224, 224),
    "preferred_resolution": (512, 512),
    "max_clip_distance_within_options": 0.45,
    "google_api_daily_quota": 100,
}
