"""
1 STAGE

Profile Generator v2 — Archetype-based, backward-designed from scenarios.

Instead of random sampling, profiles are constructed from 7 preference
archetypes × 3 variants = 21 users. Each variant guarantees cross-group
coverage in garment likes/dislikes for maximum scenario compatibility.
"""

import argparse
import random
import sys
from pathlib import Path

from .utils import call_llm, parse_json_response, save_jsonl, load_jsonl, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import PROFILES_DIR, PHASE1_CONFIG
from configs.profiles import PREFERENCE_ARCHETYPES, get_all_variants


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
3. Use everyday natural language — describe the person's taste as if introducing them.
4. Describe enduring taste only — NO situation/TPO/occasion information.
5. Length: 3-5 sentences.
6. Output ONLY a JSON object: {"narrative_profile": "..."}
"""


def build_narrative_prompt(likes, dislikes, persona_hint, user_idx):
    rng = random.Random(user_idx * 31 + 7)
    age = rng.choice(["late 20s", "early 30s", "mid 30s", "early 40s"])
    occupation = rng.choice(["office worker", "freelancer", "graduate student",
                             "designer", "engineer", "teacher", "consultant",
                             "researcher", "marketing manager", "architect"])
    return f"""Generate a narrative profile for the following user.

Demographic context: a {age} {occupation}.
Persona hint: {persona_hint}

Fashion likes: {', '.join(likes) if likes else '(none)'}
Fashion dislikes: {', '.join(dislikes) if dislikes else '(none)'}

Write 3-5 sentences. Every like and dislike must be reflected.
Do NOT include situation-specific information.

Output JSON:
{{"narrative_profile": "..."}}"""


def narrative_fallback(likes, dislikes):
    def group(kws):
        g = {}
        for kw in kws:
            if ":" in kw:
                ax, v = kw.split(":", 1)
                g.setdefault(ax, []).append(v.replace("_", " "))
        return g
    parts = []
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
    persona_hint = variant.get("persona_hint", "")

    prompt = build_narrative_prompt(likes, dislikes, persona_hint, user_idx)
    response = call_llm(prompt=prompt, stage="profile_generation",
                        system=NARRATIVE_SYSTEM,
                        provider_override=provider_override)
    parsed = parse_json_response(response)
    if parsed and parsed.get("narrative_profile"):
        narrative = parsed["narrative_profile"].strip()
    else:
        narrative = narrative_fallback(likes, dislikes)

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
    log_step(f"Profile Generator v2 — archetype-based, {n_users} users "
             f"(provider={provider or 'default'})")

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

    # Run compatibility check
    from .compatibility import get_compatible_instances, print_compatibility_report
    instances, stats = get_compatible_instances(profiles)
    print_compatibility_report(stats)
    print(f"  Total compatible instances: {len(instances)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int,
                        default=PHASE1_CONFIG["n_users_total"])
    parser.add_argument("--output", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--provider", type=str, default=None)
    args = parser.parse_args()
    run_pipeline(args.n_users, args.output, args.force, args.provider)


if __name__ == "__main__":
    main()
