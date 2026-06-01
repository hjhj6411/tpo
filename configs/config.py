"""
POD-Bench v2 central configuration — Canonical Extreme Scenario edition

Key changes from v1:
  - TPO_ATTR_INCOMPATIBILITIES removed -> CANONICAL_SCENARIOS in scenarios.py
  - Random TPO sampling removed -> scenario-based matching
  - Profile generation: archetype-based, backward-designed from scenarios
  - Option planner: uses scenario compatible/incompatible sets directly

Variant isolation:
  Set POD_VARIANT=siglip (or any tag) to redirect all data paths to
  data_<variant>/ instead of data/.  This lets ViT and FashionSigLIP
  runs coexist without touching each other's files.

  Examples:
    POD_VARIANT=siglip python src/collect_images_clip_retrieval.py ...
    POD_VARIANT=siglip python src/run_pipeline.py ...

  Unset (or POD_VARIANT='') -> original data/ layout (ViT run, unchanged).
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

GARMENT_FUNCTIONAL_GROUPS = {
    "heavy_outerwear": ["parka", "coat", "trench_coat"],
    "light_outerwear": ["jacket", "windbreaker", "blazer"],
    "formal_tops":     ["suit_jacket", "shirt", "blouse"],
    "casual_tops":     ["t_shirt", "hoodie", "tank_top", "sweater"],
    "bottoms":         ["pants", "jeans", "shorts"],
    "full_body":       ["dress", "skirt"],
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
    "n_users_total": 20,
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
        "model_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
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
        "model_name": "Qwen/Qwen3-VL-4B-Instruct",
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
