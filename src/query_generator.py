"""
Query Generator v2 — One Axis Per Instance

Emits (query_text, active_axis, fixed_attrs, tpo_scenario) records.
Color-as-safety TPO exclusions enforced for active_axis=color.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from .utils import call_llm, parse_json_list_response, save_jsonl, load_jsonl, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    QUERIES_DIR, PROFILES_DIR, FASHION_TPO, FASHION_ATTRIBUTE_AXES,
    QUERY_TYPE_RATIO, PHASE1_CONFIG, PHASE1_AXES,
    COLOR_AS_SAFETY_TPO_EXCLUSIONS,
)


def _is_safety_excluded(scenario, active_axis):
    flat = set()
    for axis, sub in scenario.items():
        if isinstance(sub, dict):
            for v in sub.values():
                flat.add(str(v).lower())
    for protected_axis, blocked in COLOR_AS_SAFETY_TPO_EXCLUSIONS:
        if active_axis == protected_axis and (flat & blocked):
            return True
    return False


def sample_tpo_scenario(seed, active_axis, max_tries=10):
    rng = random.Random(seed)
    for _ in range(max_tries):
        scenario = {}
        for axis, sub_axes in FASHION_TPO.items():
            scenario[axis] = {}
            sub_keys = list(sub_axes.keys())
            n = rng.randint(1, len(sub_keys))
            for sk in rng.sample(sub_keys, n):
                scenario[axis][sk] = rng.choice(sub_axes[sk])
        if not _is_safety_excluded(scenario, active_axis):
            return scenario
    return scenario


def tpo_to_hint(scenario):
    parts = []
    for axis, sub in scenario.items():
        for k, v in sub.items():
            parts.append(f"{axis}.{k}={v}")
    return "; ".join(parts)


def sample_fixed_attrs(active_axis, profile, rng):
    fixed = {}
    structured = profile["structured_attributes"]

    if active_axis == "color":
        gc_likes = structured.get("garment_category", {}).get("likes", [])
        gc_dislikes = structured.get("garment_category", {}).get("dislikes", [])
        cand = [g for g in FASHION_ATTRIBUTE_AXES["garment_category"]
                if g not in gc_likes and g not in gc_dislikes]
        common = ["t_shirt", "shirt", "jacket", "coat", "trench_coat", "blouse",
                   "sweater", "pants", "dress", "blazer"]
        cand_common = [g for g in cand if g in common]
        fixed["garment_category"] = rng.choice(cand_common or cand or ["jacket"])
        fixed["pattern"] = "solid"

    elif active_axis == "pattern":
        gc_likes = structured.get("garment_category", {}).get("likes", [])
        gc_dislikes = structured.get("garment_category", {}).get("dislikes", [])
        cand = [g for g in FASHION_ATTRIBUTE_AXES["garment_category"]
                if g not in gc_likes and g not in gc_dislikes]
        common = ["t_shirt", "shirt", "jacket", "coat", "blouse", "sweater",
                   "dress", "blazer", "suit_jacket"]
        cand_common = [g for g in cand if g in common]
        fixed["garment_category"] = rng.choice(cand_common or cand or ["shirt"])
        col_likes = structured.get("color", {}).get("likes", [])
        col_dislikes = structured.get("color", {}).get("dislikes", [])
        neutral = [c for c in ["black", "white", "navy", "gray", "beige"]
                    if c not in col_dislikes]
        non_liked = [c for c in neutral if c not in col_likes]
        fixed["color"] = rng.choice(non_liked or neutral or ["black"])

    elif active_axis == "garment_category":
        col_likes = structured.get("color", {}).get("likes", [])
        col_dislikes = structured.get("color", {}).get("dislikes", [])
        neutral = [c for c in ["black", "white", "navy", "gray", "beige"]
                    if c not in col_dislikes and c not in col_likes]
        if not neutral:
            neutral = [c for c in ["black", "white", "navy", "gray", "beige"]
                        if c not in col_dislikes]
        fixed["color"] = rng.choice(neutral or ["black"])
        fixed["pattern"] = "solid"

    else:
        raise ValueError(f"Unknown active_axis: {active_axis}")

    return fixed


def is_tpo_meaningful(scenario, active_axis):
    flat = set()
    for axis, sub in scenario.items():
        if isinstance(sub, dict):
            for v in sub.values():
                flat.add(str(v).lower())

    if active_axis == "garment_category":
        keys = {"winter", "snowy", "cold", "summer", "hot_humid",
                 "formal", "wedding_venue", "ceremony_attendance",
                 "office", "work_meeting", "business_casual",
                 "exercise", "gym", "beach"}
        return bool(flat & keys)
    if active_axis == "color":
        keys = {"formal", "wedding_venue", "ceremony_attendance",
                 "business_casual", "office", "work_meeting",
                 "snowy", "winter"}
        return bool(flat & keys)
    if active_axis == "pattern":
        keys = {"formal", "wedding_venue", "ceremony_attendance",
                 "business_casual", "office", "work_meeting"}
        return bool(flat & keys)
    return True


QUERY_SYSTEM = """You write natural English queries that users would ask a fashion recommendation assistant. Output ONLY a JSON list of strings."""


def build_explicit_prompt(scenario, n):
    return f"""Generate {n} natural English fashion queries for this TPO scenario.
TPO must appear EXPLICITLY in query text (mention season/weather/place/occasion).

TPO scenario: {tpo_to_hint(scenario)}

Each query: 1-2 sentences. Ask for a recommendation.
Do NOT mention specific colors, materials, patterns, or styles.

Good: "What should I wear for a rainy outdoor event this fall?"
Bad:  "Recommend a navy raincoat."

Output JSON list: ["query1", ...]"""


def build_implicit_prompt(scenario, n):
    return f"""Generate {n} English fashion queries where TPO is conveyed
IMPLICITLY through a situation NAME.

TPO scenario hint: {tpo_to_hint(scenario)}

Good: "What should I wear to a friend's wedding?"
Bad:  "What should I wear to a formal indoor wedding?"

Output JSON list."""


def build_neutral_prompt(n):
    return f"""Generate {n} English fashion queries with NO TPO — purely about
the user's taste, no situation, weather, season, or occasion.

Good: "Pick something that fits my personal style."
Bad:  "Pick an office outfit."

Output JSON list."""


def generate_query_text(qtype, scenario, provider_override=None):
    if qtype == "explicit_tpo":
        prompt = build_explicit_prompt(scenario, 1)
    elif qtype == "implicit_tpo":
        prompt = build_implicit_prompt(scenario, 1)
    else:
        prompt = build_neutral_prompt(1)
    response = call_llm(prompt=prompt, stage="query_generation",
                          system=QUERY_SYSTEM, provider_override=provider_override)
    queries = parse_json_list_response(response)
    if queries and isinstance(queries, list) and len(queries) > 0:
        return str(queries[0])
    return None


def allocate_axes(profile, n_instances):
    structured = profile["structured_attributes"]
    eligible = []
    for axis in PHASE1_AXES:
        if axis in structured:
            prefs = structured[axis]
            if prefs.get("likes") and prefs.get("dislikes"):
                eligible.append(axis)
    if not eligible:
        return []

    target = PHASE1_CONFIG["axis_task_distribution"]
    weights = {ax: target.get(ax, 0) for ax in eligible}
    total = sum(weights.values()) or 1.0

    counts = {}
    remaining = n_instances
    for ax in eligible:
        c = int(round(n_instances * weights[ax] / total))
        counts[ax] = c
        remaining -= c
    while remaining > 0:
        for ax in eligible:
            counts[ax] += 1
            remaining -= 1
            if remaining <= 0:
                break

    out = []
    for ax, c in counts.items():
        out.extend([ax] * c)
    return out[:n_instances]


def allocate_query_types(n_instances, seed):
    counts = {}
    remaining = n_instances
    for qt, ratio in QUERY_TYPE_RATIO.items():
        c = int(n_instances * ratio)
        counts[qt] = c
        remaining -= c
    counts["explicit_tpo"] = counts.get("explicit_tpo", 0) + remaining
    out = []
    for qt, c in counts.items():
        out.extend([qt] * c)
    random.Random(seed).shuffle(out)
    return out


def generate_instances_for_user(profile, n_instances, provider_override=None):
    user_id = profile["user_id"]
    base_seed = abs(hash(user_id)) % 100000

    axes = allocate_axes(profile, n_instances)
    query_types = allocate_query_types(len(axes), base_seed + 7)

    instances = []
    for idx, (active_axis, qtype) in enumerate(zip(axes, query_types)):
        seed = base_seed + idx * 13
        try:
            fixed = sample_fixed_attrs(active_axis, profile, random.Random(seed))
        except Exception as e:
            print(f"      fix-attr failed: {e}")
            continue

        if qtype == "neutral":
            scenario = None
        else:
            scenario = None
            for try_i in range(6):
                cand = sample_tpo_scenario(seed + try_i, active_axis)
                if is_tpo_meaningful(cand, active_axis):
                    scenario = cand
                    break
            if scenario is None:
                continue

        try:
            text = generate_query_text(qtype, scenario or {}, provider_override)
        except Exception as e:
            print(f"      query gen failed: {e}")
            continue
        if not text:
            continue

        secondary = ["color"] if active_axis == "garment_category" else []

        instances.append({
            "query_id": f"{user_id}_Q{len(instances)+1:03d}",
            "user_id": user_id,
            "domain": "fashion",
            "query_type": qtype,
            "query_text": text,
            "tpo_scenario": scenario,
            "active_axis": active_axis,
            "tpo_axis": active_axis,
            "fixed_attrs": fixed,
            "secondary_differentiation_axis": secondary,
        })
    return instances


def run_pipeline(profile_path, output_path, n_per_user, force=False, provider=None):
    log_step(f"Query+Axis Generator — {n_per_user} inst/user (provider={provider or 'default'})")

    profiles = load_jsonl(profile_path)
    print(f"  Loaded {len(profiles)} profiles")

    if output_path.exists() and not force:
        existing = load_jsonl(output_path)
        done_users = {q["user_id"] for q in existing}
        all_q = existing
        todo = [p for p in profiles if p["user_id"] not in done_users]
        print(f"  Resuming: {len(done_users)} users done")
    else:
        all_q = []
        todo = profiles

    for i, prof in enumerate(todo):
        print(f"\n  [{i+1}/{len(todo)}] {prof['user_id']}")
        try:
            qs = generate_instances_for_user(prof, n_per_user, provider)
            all_q.extend(qs)
            print(f"    {len(qs)} instances generated")
            if (i + 1) % 5 == 0:
                save_jsonl(all_q, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")

    save_jsonl(all_q, output_path)
    print(f"\n  ✓ Saved {len(all_q)} instances")
    print(f"  axis dist: {dict(Counter(q['active_axis'] for q in all_q))}")
    print(f"  qtype dist: {dict(Counter(q['query_type'] for q in all_q))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", type=Path, default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--output", type=Path, default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--n_per_user", type=int, default=PHASE1_CONFIG["n_instances_per_user"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--provider", type=str, default=None)
    args = parser.parse_args()
    run_pipeline(args.profile_path, args.output, args.n_per_user, args.force, args.provider)


if __name__ == "__main__":
    main()
