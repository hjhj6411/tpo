"""
STAGE 1 — Profile Generator (archetype-based, backward-designed from scenarios).

Profiles are constructed from 8 preference archetypes × 3 variants = 24 users.
Each variant guarantees cross-group coverage in garment likes/dislikes for
maximum scenario compatibility, and the 24-user population is balanced so that
NO color/pattern value is liked-only or disliked-only — a precondition for the
global counterbalancing done in STAGE 3 (see option_planner).

The narrative profile is a DETERMINISTIC template (no LLM): identical sentence
skeleton for every user, pure enumeration, always "This user ..." (never he/she).
This makes STAGE 1 fully reproducible and removes any model/provider dependency.
"""

import argparse
import sys
from pathlib import Path

from .utils import save_jsonl, load_jsonl, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import PROFILES_DIR, PHASE1_CONFIG
from configs.profiles import PREFERENCE_ARCHETYPES, get_all_variants


# ── structured → keyword lists (kept for downstream `*_keywords` fields) ──
def flatten_to_keywords(structured):
    likes, dislikes = [], []
    for axis, prefs in structured.items():
        for v in prefs.get("likes", []):
            likes.append(f"{axis}:{v}")
        for v in prefs.get("dislikes", []):
            dislikes.append(f"{axis}:{v}")
    return likes, dislikes


# ── deterministic narrative (replaces the former LLM call) ─────────────────
_GARMENT_PLURAL = {
    "t_shirt": "t-shirts", "shirt": "shirts", "blouse": "blouses",
    "sweater": "sweaters", "hoodie": "hoodies", "jacket": "jackets",
    "coat": "coats", "trench_coat": "trench coats", "blazer": "blazers",
    "parka": "parkas", "windbreaker": "windbreakers", "dress": "dresses",
    "skirt": "skirts", "pants": "pants", "jeans": "jeans", "shorts": "shorts",
    "suit_jacket": "suit jackets", "tank_top": "tank tops",
}


def _join(items):
    items = list(items)
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def narrative_template(structured):
    """Uniform, enumerative, pronoun-free narrative — same skeleton for all users.

    garment names are pluralised, colors kept as-is, pattern tokens normalised
    (graphic_print -> "graphic print") + the noun "patterns".
    """
    g = structured["garment_category"]
    c = structured["color"]
    p = structured["pattern"]
    gl = [_GARMENT_PLURAL.get(x, x.replace("_", " ")) for x in g["likes"]]
    gd = [_GARMENT_PLURAL.get(x, x.replace("_", " ")) for x in g["dislikes"]]
    pl = [x.replace("_", " ") for x in p["likes"]]
    pd = [x.replace("_", " ") for x in p["dislikes"]]
    return (
        f"This user likes wearing {_join(gl)}, prefers the colors {_join(c['likes'])}, "
        f"and likes {_join(pl)} patterns. "
        f"This user dislikes {_join(gd)}, avoids the colors {_join(c['dislikes'])}, "
        f"and dislikes {_join(pd)} patterns."
    )


def generate_profile(user_idx, archetype_id, variant_idx, variant):
    """Generate one profile from an archetype variant (fully deterministic)."""
    structured = {
        "garment_category": {"likes": variant["garment_likes"],
                             "dislikes": variant["garment_dislikes"]},
        "color": {"likes": variant["color_likes"],
                  "dislikes": variant["color_dislikes"]},
        "pattern": {"likes": variant["pattern_likes"],
                    "dislikes": variant["pattern_dislikes"]},
    }
    likes, dislikes = flatten_to_keywords(structured)
    return {
        "user_id": f"U{user_idx:03d}",
        "domain": "fashion",
        "preference_archetype": archetype_id,
        "variant_index": variant_idx,
        "structured_attributes": structured,
        "likes_keywords": likes,
        "dislikes_keywords": dislikes,
        "narrative_profile": narrative_template(structured),
    }


def run_pipeline(n_users, output_path, force=False):
    log_step(f"STAGE 1 — Profile Generator (archetype-based, {n_users} users)")

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

    all_variants = get_all_variants()
    n_variants = len(all_variants)
    if n_users > n_variants:
        print(f"  WARNING: {n_users} requested but only {n_variants} variants "
              f"defined. Capping to {n_variants}.")
        n_users = n_variants
    if n_users < n_variants:
        # R3: counterbalancing in STAGE 3 assumes the FULL balanced population.
        print(f"  WARNING: n_users={n_users} < {n_variants} variants. The population "
              f"balance (and STAGE-3 counterbalancing) requires ALL {n_variants} "
              f"users; dropping the tail leaves some values liked-only/disliked-only. "
              f"Use --n_users {n_variants}.")

    for i in range(start_idx, n_users):
        arch_id, var_idx, variant = all_variants[i]
        p = generate_profile(i + 1, arch_id, var_idx, variant)
        profiles.append(p)
        print(f"  [{i+1}/{n_users}] {p['user_id']}  {arch_id} v{var_idx}")

    save_jsonl(profiles, output_path)
    print(f"\n  ✓ Saved {len(profiles)} profiles → {output_path}")

    # compatibility sanity check
    from .compatibility import get_compatible_instances, print_compatibility_report
    instances, stats = get_compatible_instances(profiles)
    print_compatibility_report(stats)
    print(f"  Total compatible instances: {len(instances)}")
    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=PHASE1_CONFIG["n_users_total"])
    parser.add_argument("--output", type=Path, default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.n_users, args.output, args.force)


if __name__ == "__main__":
    main()
