"""
POD-Bench central configuration (v2: One Axis Per Instance)

Phase 1 design:
  - Axes: color, pattern, garment_category ONLY
    (material/fit/sleeve_length excluded — mixed-attribute risk)
  - One Axis Per Instance: each instance varies ONE attribute (active_axis);
    others are fixed across A/B/C/D
  - Provider abstraction: every LLM/VLM stage can route to 'gpt5_mini' (paid)
    or 'vllm'/'vllm_alt'/'vllm_vlm' (free local) via PROVIDERS dict
  - 15 users x 18 instances/user ≈ 270 instances across 3 axis tasks
  - Parent-ASIN variant search for color/pattern (no diffusion)
"""

from pathlib import Path

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


PHASE1_AXES = ["color", "pattern", "garment_category"]

FASHION_ATTRIBUTE_AXES = {
    "color": [
        "black", "white", "gray", "navy", "blue", "red", "pink", "orange",
        "yellow", "green", "brown", "beige", "purple",
    ],
    "pattern": [
        "solid", "striped", "checkered", "plaid", "floral", "polka_dot",
        "graphic_print", "camouflage", "animal_print",
    ],
    "garment_category": [
        "t_shirt", "shirt", "blouse", "sweater", "hoodie",
        "jacket", "coat", "trench_coat", "blazer", "parka", "windbreaker",
        "dress", "skirt", "pants", "jeans", "shorts", "suit_jacket",
        "tank_top",
    ],
}

ATTRIBUTE_SAMPLING = {
    "n_likes_per_axis": (1, 2),
    "n_dislikes_per_axis": (1, 2),
    "n_axes_with_preferences": (2, 3),
}

FASHION_TPO = {
    "time": {
        "season": ["spring", "summer", "fall", "winter"],
        "weather": ["sunny", "rainy", "snowy", "hot_humid", "cold", "windy"],
        "time_of_day": ["morning", "daytime", "evening", "night"],
    },
    "place": {
        "venue": ["office", "cafe", "park", "beach", "city_street",
                  "restaurant", "wedding_venue", "gym",
                  "art_gallery", "concert_hall", "airport"],
        "indoor_outdoor": ["indoor", "outdoor", "mixed"],
    },
    "occasion": {
        "activity": ["work_meeting", "casual_outing", "exercise", "first_date",
                      "ceremony_attendance", "travel", "errands", "social_gathering"],
        "formality_required": ["very_casual", "casual", "smart_casual",
                                "business_casual", "formal"],
    },
}

# Color-as-safety TPO exclusions (documented in paper Appendix)
COLOR_AS_SAFETY_TPO_EXCLUSIONS = [
    ("color", {"mountain_hike", "skiing", "hunting", "running_outdoors",
                "construction", "cycling_on_road", "fishing_boat"}),
]

QUERY_TYPE_RATIO = {
    "explicit_tpo": 0.40,
    "implicit_tpo": 0.30,
    "neutral":      0.30,
}

OPTION_LABELS = {
    "tpo_and_preference": "matches both TPO and user preference (correct)",
    "tpo_only":           "matches TPO but violates user preference",
    "preference_only":    "matches user preference but violates TPO",
    "neither":            "violates both",
}

PHASE1_CONFIG = {
    "n_users_total": 15,
    "n_instances_per_user": 18,
    "axis_task_distribution": {
        "color":            0.40,
        "pattern":          0.30,
        "garment_category": 0.30,
    },
    "n_options_per_query": 4,
}


# ─────────────────────────────────────────────
# Provider abstraction
# ─────────────────────────────────────────────
# Every pipeline stage maps to one of the providers below. To switch a stage
# from free vllm to paid gpt5_mini, change PROVIDERS[<stage>]["provider"].

PROVIDERS = {
    "profile_generation":    {"provider": "vllm"},
    "query_generation":      {"provider": "vllm"},
    "option_planning":       {"provider": "vllm"},
    "label_judge_primary":   {"provider": "vllm"},
    "label_judge_secondary": {"provider": "vllm_alt"},
    "label_judge_tertiary":  {"provider": "gpt5_mini"},
    "blind_solver":          {"provider": "vllm"},
    "captioner":             {"provider": "vllm_vlm"},
    "vlm_evaluator":         {"provider": "vllm_vlm"},
}

PROVIDER_ENDPOINTS = {
    "gpt5_mini": {
        "kind": "openai_api",
        "model_name": "gpt-5-mini",
        "api_base": "https://api.openai.com/v1",
        "uses_max_completion_tokens": True,
        "supports_temperature": False,
        "default_max_tokens": 4096,
        "long_max_tokens": 8192,
    },
    "vllm": {
        "kind": "openai_compat",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "api_base": "http://localhost:8000/v1",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "default_max_tokens": 2048,
        "long_max_tokens": 4096,
    },
    "vllm_alt": {
        "kind": "openai_compat",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "api_base": "http://localhost:8001/v1",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "default_max_tokens": 2048,
        "long_max_tokens": 4096,
    },
    "vllm_vlm": {
        "kind": "openai_compat",
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "api_base": "http://localhost:8002/v1",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "default_max_tokens": 512,
        "long_max_tokens": 1024,
    },
}


LABEL_QUALITY_THRESHOLDS = {
    "krippendorff_alpha_min": 0.7,
    "cohen_kappa_min": 0.6,
    "judge_agreement_min": 2,
}

VISION_ESSENTIALITY_THRESHOLDS = {
    "blind_llm_max_acc": 0.40,
    "captioner_llm_max_acc": 0.45,
    "full_vlm_min_acc": 0.55,
}

IMAGE_COLLECTION = {
    "min_resolution": (224, 224),
    "preferred_resolution": (512, 512),
    "max_clip_distance_within_options": 0.65,
    "min_ssim_for_color_variants": 0.35,
    "google_api_daily_quota": 100,
}
