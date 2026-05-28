import json
from collections import defaultdict

PROFILES_PATH = "data/profiles/profiles.jsonl"
PLANS_PATH = "data/options/option_plans.jsonl"

profiles = {}
with open(PROFILES_PATH) as f:
    for line in f:
        p = json.loads(line)
        profiles[p["user_id"]] = p

results = defaultdict(int)
cases = defaultdict(list)

with open(PLANS_PATH) as f:
    for line in f:
        plan = json.loads(line)
        uid = plan["user_id"]
        active_axis = plan["active_axis"]
        fixed_attrs = plan.get("fixed_attrs", {})
        profile = profiles.get(uid, {})
        struct = profile.get("structured_attributes", {})

        for axis, val in fixed_attrs.items():
            if axis == active_axis:
                continue

            axis_pref = struct.get(axis, {})
            likes = set(axis_pref.get("likes", []))
            dislikes = set(axis_pref.get("dislikes", []))

            if val in dislikes:
                results["fixed_in_dislikes"] += 1
                cases["fixed_in_dislikes"].append({
                    "query_id": plan["query_id"],
                    "active_axis": active_axis,
                    "fixed_axis": axis,
                    "fixed_val": val,
                })
            elif val in likes:
                results["fixed_in_likes"] += 1
                cases["fixed_in_likes"].append({
                    "query_id": plan["query_id"],
                    "active_axis": active_axis,
                    "fixed_axis": axis,
                    "fixed_val": val,
                })
            else:
                results["fixed_neutral"] += 1

print("=== Fixed Attrs vs User Preference ===")
for k in ["fixed_in_dislikes", "fixed_in_likes", "fixed_neutral"]:
    print(f"  {k}: {results[k]}")

for bucket in ["fixed_in_dislikes", "fixed_in_likes"]:
    if cases[bucket]:
        print(f"\n{bucket} 샘플 (상위 5개):")
        for c in cases[bucket][:5]:
            print(f"  {c}")