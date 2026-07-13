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
import sys
from pathlib import Path

from .utils import save_jsonl, load_jsonl, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import PROFILES_DIR, PHASE1_CONFIG
from configs.profiles import get_all_variants


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
    "trench_coat": "trench coat",
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


def generate_profile(user_idx, archetype_id, variant_idx, variant,
                     provider_override=None):
    """Generate one profile from an archetype variant."""
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

    return {
        "user_id": f"U{user_idx:03d}",
        "domain": "fashion",
        "preference_archetype": archetype_id,
        "variant_index": variant_idx,
        "structured_attributes": structured,
        "likes_keywords": likes,
        "dislikes_keywords": dislikes,
        "narrative_profile": narrative,
    }


def run_pipeline(n_users, output_path, force=False, provider=None):
    log_step(f"Profile Generator v4 — deterministic preference-only, {n_users} users")

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
    if n_users > len(all_variants):
        print(f"  WARNING: {n_users} users requested but only "
              f"{len(all_variants)} variants defined. Capping.")
        n_users = len(all_variants)

    for i in range(start_idx, n_users):
        arch_id, var_idx, variant = all_variants[i]
        print(f"\n  [{i+1}/{n_users}] archetype={arch_id}, variant={var_idx}")
        try:
            p = generate_profile(i + 1, arch_id, var_idx, variant,
                                 provider_override=provider)
            profiles.append(p)
            print(f"    {p['user_id']}")
            print(f"    likes:    {p['likes_keywords']}")
            print(f"    dislikes: {p['dislikes_keywords']}")
            print(f"    narrative: {p['narrative_profile'][:100]}...")
            if (i + 1) % 5 == 0:
                save_jsonl(profiles, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")

    save_jsonl(profiles, output_path)
    print(f"\n  ✓ Saved {len(profiles)} profiles")

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
    args = parser.parse_args()
    run_pipeline(args.n_users, args.output, args.force, args.provider)


if __name__ == "__main__":
    main()
