"""
Profile Generator v2 (One Axis Per Instance, Phase 1)
Restricted to PHASE1_AXES (color, pattern, garment_category).
"""

import argparse
import random
import sys
from pathlib import Path

from .utils import call_llm, parse_json_response, save_jsonl, load_jsonl, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    PROFILES_DIR, FASHION_ATTRIBUTE_AXES, ATTRIBUTE_SAMPLING,
    PHASE1_CONFIG, PHASE1_AXES,
)


def sample_structured_attributes(seed):
    rng = random.Random(seed)
    axes = list(PHASE1_AXES)
    n_axes_min, n_axes_max = ATTRIBUTE_SAMPLING["n_axes_with_preferences"]
    n_axes = rng.randint(n_axes_min, min(n_axes_max, len(axes)))
    chosen_axes = rng.sample(axes, n_axes)

    n_likes_min, n_likes_max = ATTRIBUTE_SAMPLING["n_likes_per_axis"]
    n_dis_min, n_dis_max = ATTRIBUTE_SAMPLING["n_dislikes_per_axis"]

    structured = {}
    for axis in chosen_axes:
        values = list(FASHION_ATTRIBUTE_AXES[axis])
        rng.shuffle(values)
        n_like = rng.randint(n_likes_min, min(n_likes_max, len(values)))
        n_dis = rng.randint(n_dis_min, min(n_dis_max, max(0, len(values) - n_like)))
        structured[axis] = {
            "likes": values[:n_like],
            "dislikes": values[n_like:n_like + n_dis],
        }

    if "color" not in structured:
        colors = list(FASHION_ATTRIBUTE_AXES["color"])
        rng.shuffle(colors)
        structured["color"] = {"likes": colors[:2], "dislikes": colors[2:3]}

    return structured


def flatten_to_keywords(structured):
    likes, dislikes = [], []
    for axis, prefs in structured.items():
        for v in prefs.get("likes", []):
            likes.append(f"{axis}:{v}")
        for v in prefs.get("dislikes", []):
            dislikes.append(f"{axis}:{v}")
    return likes, dislikes


NARRATIVE_SYSTEM = """You write concise English user profiles for a fashion personalization benchmark.

Rules:
1. English only.
2. Reflect EVERY supplied like/dislike keyword in the narrative.
3. Use everyday natural language.
4. Describe enduring taste only — NO situation/TPO information.
5. Length: 3-5 sentences.
6. Output ONLY a JSON object: {"narrative_profile": "..."}
"""


def build_narrative_prompt(likes, dislikes, user_idx):
    rng = random.Random(user_idx * 31 + 7)
    age = rng.choice(["late 20s", "early 30s", "mid 30s", "early 40s"])
    occupation = rng.choice(["office worker", "freelancer", "graduate student",
                              "designer", "engineer", "teacher", "consultant",
                              "researcher"])
    return f"""Generate a narrative profile for the following user.

Demographic context: a {age} {occupation}.

Fashion likes: {', '.join(likes) if likes else '(none)'}
Fashion dislikes: {', '.join(dislikes) if dislikes else '(none)'}

Write 3-5 sentences. Every like and dislike must be reflected.
Do NOT include situation-specific information.

Output JSON:
{{"narrative_profile": "..."}}"""


def narrative_fallback(likes, dislikes):
    parts = []
    def group(kws):
        g = {}
        for kw in kws:
            if ":" in kw:
                ax, v = kw.split(":", 1)
                g.setdefault(ax, []).append(v.replace("_", " "))
        return g
    lg, dg = group(likes), group(dislikes)
    if lg:
        seg = [f"{ax.replace('_',' ')} of {' or '.join(vs)}" for ax, vs in lg.items()]
        parts.append("This user gravitates toward " + ", ".join(seg) + ".")
    if dg:
        seg = [f"{' or '.join(vs)} {ax.replace('_',' ')}" for ax, vs in dg.items()]
        parts.append("They tend to avoid " + ", ".join(seg) + ".")
    if not parts:
        parts.append("This user has flexible fashion preferences.")
    return " ".join(parts)


def generate_profile(user_idx, seed, provider_override=None):
    structured = sample_structured_attributes(seed)
    likes, dislikes = flatten_to_keywords(structured)
    prompt = build_narrative_prompt(likes, dislikes, user_idx)
    response = call_llm(prompt=prompt, stage="profile_generation",
                          system=NARRATIVE_SYSTEM, provider_override=provider_override)
    parsed = parse_json_response(response)
    if parsed and parsed.get("narrative_profile"):
        narrative = parsed["narrative_profile"].strip()
    else:
        narrative = narrative_fallback(likes, dislikes)
    return {
        "user_id": f"U{user_idx:03d}",
        "domain": "fashion",
        "structured_attributes": structured,
        "likes_keywords": likes,
        "dislikes_keywords": dislikes,
        "narrative_profile": narrative,
        "seed": seed,
    }


def run_pipeline(n_users, output_path, force=False, provider=None):
    log_step(f"Profile Generator — {n_users} users (provider={provider or 'default'})")

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

    for i in range(start_idx, n_users):
        seed = 1000 + i * 17
        print(f"\n  [{i+1}/{n_users}] seed={seed}")
        try:
            p = generate_profile(i + 1, seed, provider_override=provider)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=PHASE1_CONFIG["n_users_total"])
    parser.add_argument("--output", type=Path, default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--provider", type=str, default=None)
    args = parser.parse_args()
    run_pipeline(args.n_users, args.output, args.force, args.provider)


if __name__ == "__main__":
    main()
