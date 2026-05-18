"""
Profile Generator (Fashion only, English)
==========================================

Profile structure (MMPB-inspired):

  structured_attributes:    closed-set ground-truth (hidden from model)
  likes_keywords:           flat list of "axis:value" tokens (shown to model)
  dislikes_keywords:        flat list of "axis:value" tokens (shown to model)
  narrative_profile:        natural-language paragraph derived from keywords

The benchmark supports an ablation comparing:
  (a) keyword-only profile representation
  (b) narrative-only profile representation
  (c) keyword + narrative combined
This corresponds to MMPB's "4 levels of granularity" axis.

All profiles are generated in ENGLISH (target = international venue).

Attribute axes are *closed-set* and *objectively decidable*
(color, material, pattern, garment category, fit, sleeve length, neckline,
formality). Style labels like "minimalist" are excluded from likes/dislikes
because they are subjective and ambiguous.
"""

import argparse
import random
from pathlib import Path

from .utils import (
    call_gpt5_mini, parse_json_response,
    save_jsonl, load_jsonl, log_step,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    PROFILES_DIR, FASHION_ATTRIBUTE_AXES, ATTRIBUTE_SAMPLING,
    PHASE1_CONFIG, GPT5_MINI,
)


# ─────────────────────────────────────────────
# 1. Sample structured ground-truth attributes
# ─────────────────────────────────────────────

def sample_structured_attributes(seed: int) -> dict:
    """Sample a coherent set of likes/dislikes from closed-set axes.

    Strategy: avoid contradictions inside an axis (no liking & disliking
    the same value); allow likes to span multiple values per axis;
    ensure 5-7 axes have explicit preferences (the rest are unconstrained).
    """
    rng = random.Random(seed)
    axes = list(FASHION_ATTRIBUTE_AXES.keys())

    n_axes_min, n_axes_max = ATTRIBUTE_SAMPLING["n_axes_with_preferences"]
    n_axes = rng.randint(n_axes_min, n_axes_max)
    chosen_axes = rng.sample(axes, n_axes)

    n_likes_min, n_likes_max = ATTRIBUTE_SAMPLING["n_likes_per_axis"]
    n_dis_min, n_dis_max = ATTRIBUTE_SAMPLING["n_dislikes_per_axis"]

    structured = {}
    for axis in chosen_axes:
        values = list(FASHION_ATTRIBUTE_AXES[axis])
        rng.shuffle(values)

        n_like = rng.randint(n_likes_min, min(n_likes_max, len(values)))
        n_dis = rng.randint(n_dis_min, min(n_dis_max, max(0, len(values) - n_like)))

        likes = values[:n_like]
        dislikes = values[n_like : n_like + n_dis]

        structured[axis] = {
            "likes": likes,
            "dislikes": dislikes,
        }

    # Ensure at least the foundational axes (color, garment_category) are present
    # (otherwise it's hard to write a meaningful narrative)
    if "color" not in structured:
        colors = list(FASHION_ATTRIBUTE_AXES["color"])
        rng.shuffle(colors)
        structured["color"] = {"likes": colors[:2], "dislikes": colors[2:3]}

    return structured


def flatten_to_keywords(structured: dict) -> tuple[list[str], list[str]]:
    """Convert structured dict to flat keyword lists.

    Format: 'axis:value' (e.g., 'color:navy', 'material:linen').
    This explicit prefixing prevents ambiguity (e.g., 'cotton' could be
    misread as a color in raw form, but 'material:cotton' is unambiguous).
    """
    likes, dislikes = [], []
    for axis, prefs in structured.items():
        for v in prefs.get("likes", []):
            likes.append(f"{axis}:{v}")
        for v in prefs.get("dislikes", []):
            dislikes.append(f"{axis}:{v}")
    return likes, dislikes


# ─────────────────────────────────────────────
# 2. Generate English narrative from keywords
# ─────────────────────────────────────────────

NARRATIVE_SYSTEM = """\
You are a user profile writer for a personalization research benchmark.
You produce concise, specific English user profiles based on a list of
preference keywords supplied to you.

Strict rules:
1. Write in ENGLISH.
2. Faithfully reflect EVERY supplied like/dislike keyword in the narrative.
   Do not add new preferences that are not in the keyword list.
3. Express preferences in everyday natural language (e.g., 'tends to gravitate
   toward navy and beige tones', 'avoids floral or polka-dot patterns').
4. The narrative must describe ENDURING taste only.
   Do NOT include any situation/TPO information (no 'for the office',
   'on rainy days', 'when going hiking', etc.).
5. Length: 3-5 sentences, ~80-120 English words.
6. You MAY use light style descriptors (e.g., 'understated', 'practical')
   for fluency, but the substance must come from the keywords.
7. Output ONLY a JSON object with key 'narrative_profile'.
"""


def build_narrative_prompt(likes: list[str], dislikes: list[str],
                            user_idx: int) -> str:
    """Build the prompt for narrative generation."""
    # Light demographic context (independent of taste)
    rng = random.Random(user_idx * 31 + 7)
    age = rng.choice(["late 20s", "early 30s", "mid 30s", "early 40s"])
    occupation = rng.choice([
        "office worker", "freelancer", "graduate student", "designer",
        "engineer", "teacher", "consultant", "researcher",
    ])

    likes_str = ", ".join(likes) if likes else "(none specified)"
    dislikes_str = ", ".join(dislikes) if dislikes else "(none specified)"

    return f"""Generate a narrative profile for the following user.

Demographic context: a {age} {occupation}.

Fashion likes (keywords): {likes_str}
Fashion dislikes (keywords): {dislikes_str}

Write 3-5 sentences in English that describe this user's enduring fashion taste.
Every like and dislike above must be reflected in the narrative.
Do NOT include situation-specific information.

Output JSON:
{{
  "narrative_profile": "..."
}}"""


# ─────────────────────────────────────────────
# 3. Fallback (if GPT-5-mini returns empty)
# ─────────────────────────────────────────────

def narrative_fallback(likes: list[str], dislikes: list[str]) -> str:
    """Build a deterministic narrative if the API returns nothing.

    This protects the pipeline from API failures during the first cycle.
    Quality is lower than LLM-generated but preserves all preference info.
    """
    parts = []

    def group_by_axis(kws):
        groups = {}
        for kw in kws:
            if ":" in kw:
                ax, v = kw.split(":", 1)
                groups.setdefault(ax, []).append(v.replace("_", " "))
        return groups

    like_groups = group_by_axis(likes)
    dislike_groups = group_by_axis(dislikes)

    if like_groups:
        seg = []
        for ax, vs in like_groups.items():
            ax_h = ax.replace("_", " ")
            if len(vs) == 1:
                seg.append(f"{ax_h} of {vs[0]}")
            else:
                seg.append(f"{ax_h}s like {' or '.join(vs)}")
        parts.append("This user gravitates toward " + ", ".join(seg) + ".")

    if dislike_groups:
        seg = []
        for ax, vs in dislike_groups.items():
            ax_h = ax.replace("_", " ")
            seg.append(f"{' or '.join(vs)} {ax_h}")
        parts.append("They tend to avoid " + ", ".join(seg) + ".")

    if not parts:
        parts.append("This user has flexible fashion preferences with no strong likes or dislikes.")

    return " ".join(parts)


# ─────────────────────────────────────────────
# 4. Main pipeline
# ─────────────────────────────────────────────

def generate_profile(user_idx: int, seed: int) -> dict:
    """Generate a single user profile."""
    structured = sample_structured_attributes(seed)
    likes, dislikes = flatten_to_keywords(structured)

    prompt = build_narrative_prompt(likes, dislikes, user_idx)
    response = call_gpt5_mini(
        prompt=prompt,
        system=NARRATIVE_SYSTEM,
        max_completion_tokens=GPT5_MINI["max_completion_tokens_default"],
    )

    parsed = parse_json_response(response)
    if parsed and parsed.get("narrative_profile"):
        narrative = parsed["narrative_profile"].strip()
    else:
        narrative = narrative_fallback(likes, dislikes)

    return {
        "user_id": f"U{user_idx:03d}",
        "domain": "fashion",
        "structured_attributes": structured,    # ground-truth (hidden)
        "likes_keywords": likes,                # shown to model (keyword variant)
        "dislikes_keywords": dislikes,          # shown to model (keyword variant)
        "narrative_profile": narrative,         # shown to model (narrative variant)
        "seed": seed,
    }


def run_pipeline(n_users: int, output_path: Path, force: bool = False):
    log_step(f"Profile Generator — {n_users} users (English, fashion only)")

    if output_path.exists() and not force:
        existing = load_jsonl(output_path)
        if len(existing) >= n_users:
            print(f"  Already have {len(existing)} profiles. Skipping.")
            return existing
        print(f"  Resuming from {len(existing)} profiles")
        profiles = existing
        start_idx = len(existing)
    else:
        profiles = []
        start_idx = 0

    for i in range(start_idx, n_users):
        seed = 1000 + i * 17
        print(f"\n  [{i+1}/{n_users}] seed={seed}")
        try:
            p = generate_profile(i + 1, seed)
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
    print(f"\n  ✓ Saved {len(profiles)} profiles to {output_path}")
    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int,
                        default=PHASE1_CONFIG["n_users_total"])
    parser.add_argument("--output", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.n_users, args.output, args.force)


if __name__ == "__main__":
    main()
