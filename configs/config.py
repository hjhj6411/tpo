"""
POD-Bench clean configuration for image retrieval.

Design goals:
- remove semantically overlapping canonical labels (plaid/checkered, suit_jacket/blazer, jacket/*_jacket)
- keep garment labels short enough for CLIP retrieval and VLM judging
- remove low-stability or data-sparse labels requested by the user
- keep dress unified as `dress`
- keep skirt split as `mini_skirt` / `long_skirt`
- remove camouflage from the canonical pattern vocabulary
- add leopard as a single canonical animal-print pattern label
"""

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Variant-aware data root ───────────────────────────────
_VARIANT = os.environ.get("POD_VARIANT", "").strip()
DATA_DIR = ROOT / (f"data_{_VARIANT}" if _VARIANT else "data")

PROFILES_DIR = DATA_DIR / "profiles"
QUERIES_DIR  = DATA_DIR / "queries"
OPTIONS_DIR  = DATA_DIR / "options"
IMAGES_DIR   = DATA_DIR / "images"
LABELS_DIR   = DATA_DIR / "labels"
FINAL_DIR    = DATA_DIR / "final"

for d in [PROFILES_DIR, QUERIES_DIR, OPTIONS_DIR, IMAGES_DIR, LABELS_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Phase-1 axes ──────────────────────────────────────────
PHASE1_AXES = ["color", "pattern", "garment_category"]

# Canonical attribute vocabulary.
# Important: these are benchmark labels, not a full fashion taxonomy.
# Do not add aliases here. Synonyms should be handled only in retrieval/verifier logic.
FASHION_ATTRIBUTE_AXES = {
    "color": [
        "black", "white", "gray", "navy", "blue", "red", "pink", "orange",
        "yellow", "green", "brown", "beige", "purple",
    ],
    "pattern": [
        "solid", "striped", "checkered", "floral", "polka_dot", "leopard",
    ],
    "garment_category": [
        # tops
        "t_shirt", "tank_top", "formal_shirt",
        "sweatshirt", "sweater", "hoodie", "cardigan",
        # outerwear
        "blazer", "windbreaker", "leather_jacket", "puffer_jacket",
        "fleece_jacket", "trench_coat",
        # bottoms
        "jeans", "slacks", "shorts", "leggings",
        # one-piece / skirts
        "dress", "mini_skirt", "long_skirt",
    ],
}

GARMENT_FUNCTIONAL_GROUPS = {
    "basic_tops":       ["t_shirt", "tank_top", "formal_shirt"],
    "soft_tops":        ["sweatshirt", "sweater", "hoodie", "cardigan"],
    "light_outerwear":  ["windbreaker", "leather_jacket", "blazer"],
    "heavy_outerwear":  ["fleece_jacket", "puffer_jacket", "trench_coat"],
    "bottoms":          ["jeans", "slacks", "shorts", "leggings"],
    "dress_skirt":      ["dress", "mini_skirt", "long_skirt"],
}

COLOR_GROUPS = {
    "dark":    ["black", "navy", "gray", "brown"],
    "neutral": ["white", "beige"],
    "bright":  ["red", "blue", "green", "orange", "yellow", "pink", "purple"],
}

OPTION_LABELS = {
    "tpo_and_preference": "matches both TPO and user preference (correct)",
    "tpo_only":           "matches TPO but does not match user preference",
    "preference_only":    "matches user preference but violates TPO",
    "neither":            "violates both",
}

PHASE1_CONFIG = {
    "n_users_total": 24,
    "n_users_per_archetype": 3,
    "n_options_per_query": 4,
    "query_type_distribution": {
        "explicit_tpo": 0.55,
        "implicit_tpo": 0.45,
    },
}

# ── Provider abstraction ──────────────────────────────────
PROVIDERS = {
    "profile_generation":        {"provider": "vllm"},
    "query_generation":          {"provider": "vllm"},
    "option_planning":           {"provider": "vllm"},
    "label_judge_primary":       {"provider": "vllm"},
    "label_judge_secondary":     {"provider": "vllm_alt"},
    "label_judge_tertiary":      {"provider": "gpt5_mini"},
    "blind_solver":              {"provider": "vllm"},
    "captioner":                 {"provider": "vllm_vlm"},
    "vlm_evaluator":             {"provider": "vllm_vlm"},
    "image_verifier":            {"provider": "vllm_vlm"},
    "text_only_eval":            {"provider": "vllm"},
    "text_only_no_profile_eval": {"provider": "vllm"},
    "query_rewrite":             {"provider": "vllm"},
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
    "gpt5_nano": {
        "kind": "openai_api",
        "model_name": "gpt-5-nano",
        "api_base": "https://api.openai.com/v1",
        "uses_max_completion_tokens": True,
        "supports_temperature": False,
        "default_max_tokens": 4096,
        "long_max_tokens": 8192,
    },
    "vllm": {
        "kind": "openai_compat",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "api_base": "http://localhost:8002/v1",
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
    "qwen36_27b_choice": {
        "kind": "openai_compat",
        "model_name": os.environ.get("QWEN36_MODEL", "Qwen/Qwen3.6-27B"),
        "api_base": os.environ.get("QWEN36_API_BASE", "http://localhost:8000/v1"),
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "default_max_tokens": 8,
        "long_max_tokens": 16,
    },
    "vllm_vlm": {
        "kind": "openai_compat",
        "model_name": "Qwen/Qwen3-VL-30B-A3B-Instruct",
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
