"""Availability-audit configuration.

All labels are validated against configs.config.FASHION_ATTRIBUTE_AXES at
import time so the audit can never drift from the canonical vocabulary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import FASHION_ATTRIBUTE_AXES, DATA_DIR

# ── endpoints ─────────────────────────────────────────────
KNN_URL = "http://127.0.0.1:1235"          # fsiglip/serve_fsiglip_knn.py
VLM_URL = "http://127.0.0.1:8002/v1"       # vLLM OpenAI-compat (Qwen3-VL)
VLM_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# ── outputs ───────────────────────────────────────────────
AUDIT_DIR = DATA_DIR / "retrieval" / "availability"
CANDIDATES_PATH = AUDIT_DIR / "candidates.jsonl"           # after retrieve
CATALOG_PATH = AUDIT_DIR / "availability_catalog.jsonl"    # after verify+classify
CORPUS_SUPPORT_PATH = AUDIT_DIR / "corpus_support.json"
PIPELINE_YIELD_PATH = AUDIT_DIR / "pipeline_yield.json"

# ── retrieval settings ────────────────────────────────────
TOP_K_PER_GARMENT = 500      # per garment, union over aliases, before dedup
KNN_BATCH = 100              # num_images per /knn-service call
PATCH_GRID = 5

# Garment-only retrieval aliases: recall expansion ONLY. The final garment
# label always comes from the VLM closed-vocabulary classification below.
GARMENT_RETRIEVAL_ALIASES = {
    "t_shirt":        ["t-shirt", "tee shirt"],
    "tank_top":       ["tank top", "sleeveless top", "camisole top"],
    "formal_shirt":   ["formal shirt", "dress shirt", "button-up shirt"],
    "polo_shirt":     ["polo shirt", "collared polo"],
    "sweatshirt":     ["sweatshirt", "crewneck sweatshirt"],
    "sweater":        ["sweater", "pullover", "knit sweater"],
    "hoodie":         ["hoodie", "hooded sweatshirt"],
    "cardigan":       ["cardigan"],
    "blazer":         ["blazer", "tailored jacket"],
    "windbreaker":    ["windbreaker", "lightweight nylon jacket"],
    "leather_jacket": ["leather jacket"],
    "puffer_jacket":  ["puffer jacket", "quilted down jacket"],
    "fleece_jacket":  ["fleece jacket", "fleece zip-up jacket"],
    "pea_coat":       ["pea coat", "peacoat"],
    # `wool coat` / `wool overcoat` are HYPERNYMS, not synonyms: a pea coat is
    # also a wool coat, so these aliases retrieved pea coats into the long_coat
    # bank. Recall expansion must never cross into a sibling of the canonical
    # 23 — the two coats are distinguished by LENGTH, and only the canonical
    # name encodes that.
    "long_coat":      ["long coat"],
    "suit_vest":      ["suit vest", "tailored waistcoat"],
    "jeans":          ["jeans", "denim pants"],
    "slacks":         ["slacks", "dress pants", "trousers"],
    "shorts":         ["shorts"],
    "leggings":       ["leggings"],
    "dress":          ["dress", "gown"],
    "mini_skirt":     ["mini skirt"],
    "long_skirt":     ["long skirt", "maxi skirt"],
}

# ── availability grading (pilot-calibrated; see README) ──
GRADE_GREEN_MIN = 8          # verified unique images
GRADE_YELLOW_MIN = 3
MASK_SUCCESS_MIN = 0.60      # green additionally requires this on pipeline_yield


def n_min(planned_uses: int, max_reuse_per_image: int = 2, safety_buffer: int = 2) -> int:
    """Usage-aware minimum support: ceil(uses / reuse cap) + buffer."""
    return -(-planned_uses // max(max_reuse_per_image, 1)) + safety_buffer


# ── pilot pairs (section 13 of README) ────────────────────
PILOT_RARE_PAIRS = [
    ("striped", "pea_coat"), ("striped", "puffer_jacket"),
    ("leopard", "sweatshirt"), ("polka_dot", "windbreaker"),
    ("floral", "fleece_jacket"), ("checkered", "puffer_jacket"),
    ("leopard", "fleece_jacket"), ("polka_dot", "leather_jacket"),
    ("floral", "long_coat"), ("striped", "suit_vest"),
]
PILOT_COMMON_PAIRS = [
    ("solid", "puffer_jacket"), ("striped", "formal_shirt"),
    ("checkered", "formal_shirt"), ("floral", "dress"),
    ("solid", "polo_shirt"), ("striped", "t_shirt"),
    ("leopard", "dress"), ("polka_dot", "dress"),
    ("solid", "long_coat"), ("checkered", "long_skirt"),
]

# ── import-time validation against canonical vocabulary ──
_G = set(FASHION_ATTRIBUTE_AXES["garment_category"])
_P = set(FASHION_ATTRIBUTE_AXES["pattern"])
assert set(GARMENT_RETRIEVAL_ALIASES) == _G, (
    f"alias map out of sync with vocab: "
    f"missing={_G - set(GARMENT_RETRIEVAL_ALIASES)}, "
    f"extra={set(GARMENT_RETRIEVAL_ALIASES) - _G}")
for p, g in PILOT_RARE_PAIRS + PILOT_COMMON_PAIRS:
    assert p in _P and g in _G, f"pilot pair out of vocab: ({p}, {g})"
