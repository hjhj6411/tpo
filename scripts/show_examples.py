import json
import argparse
from pathlib import Path

PROFILES_PATH = "data/profiles/profiles.jsonl"
PLANS_PATH = "data/options/option_plans.jsonl"


def load_profiles(path):
    profiles = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            profiles[p["user_id"]] = p
    return profiles


def load_plans(path):
    plans = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            plans.append(json.loads(line))
    return plans


def attrs_to_text(attrs):
    color = attrs.get("color")
    pattern = attrs.get("pattern")
    garment = attrs.get("garment_category", "item")

    parts = []
    if color:
        parts.append(color)
    if pattern and pattern != "solid":
        parts.append(pattern.replace("_", " "))
    parts.append(garment.replace("_", " "))
    return " ".join(parts)


def profile_to_text(profile):
    # 우선 narrative 우선 사용, 없으면 structured fallback
    if profile.get("profile_text"):
        return profile["profile_text"]
    if profile.get("narrative"):
        return profile["narrative"]
    if profile.get("description"):
        return profile["description"]

    sa = profile.get("structured_attributes", {})
    chunks = []

    for axis in ["garment_category", "color", "pattern"]:
        prefs = sa.get(axis, {})
        likes = prefs.get("likes", [])
        dislikes = prefs.get("dislikes", [])

        if likes:
            chunks.append(f"likes {axis}: {', '.join(likes)}")
        if dislikes:
            chunks.append(f"dislikes {axis}: {', '.join(dislikes)}")

    return "; ".join(chunks) if chunks else "No profile text available."


def format_example(plan, profile):
    q = plan.get("query_text", "").strip()
    profile_text = profile_to_text(profile)

    A = attrs_to_text(plan["options"]["A"]["attributes"])
    B = attrs_to_text(plan["options"]["B"]["attributes"])
    C = attrs_to_text(plan["options"]["C"]["attributes"])
    D = attrs_to_text(plan["options"]["D"]["attributes"])

    text = []
    text.append(f"=== QUERY === {q}")
    text.append(f"=== USER PROFILE === {profile_text}")
    text.append("=== OPTIONS ===")
    text.append(f"A. {A}")
    text.append(f"B. {B}")
    text.append(f"C. {C}")
    text.append(f"D. {D}")
    text.append("Which option best fits the user's query and preferences? Answer with only the letter (A/B/C/D):")
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=str, default=PROFILES_PATH)
    parser.add_argument("--plans", type=str, default=PLANS_PATH)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--query_id", type=str, default=None)
    args = parser.parse_args()

    profiles = load_profiles(args.profiles)
    plans = load_plans(args.plans)

    if args.query_id:
        plans = [p for p in plans if p["query_id"] == args.query_id]

    shown = 0
    for plan in plans:
        uid = plan["user_id"]
        profile = profiles[uid]
        print(format_example(plan, profile))
        print()
        shown += 1
        if shown >= args.n:
            break


if __name__ == "__main__":
    main()