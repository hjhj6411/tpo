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
noisy_cases = []

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
            likes = axis_pref.get("likes", [])
            dislikes = axis_pref.get("dislikes", [])

            if val in dislikes:
                results["fixed_in_dislikes"] += 1
                noisy_cases.append({
                    "query_id": plan["query_id"],
                    "active_axis": active_axis,
                    "fixed_axis": axis,
                    "fixed_val": val,
                    "status": "IN_DISLIKES"
                })
            elif val not in likes:
                results["fixed_not_in_likes"] += 1
                noisy_cases.append({
                    "query_id": plan["query_id"],
                    "active_axis": active_axis,
                    "fixed_axis": axis,
                    "fixed_val": val,
                    "status": "NOT_IN_LIKES"
                })
            else:
                results["fixed_in_likes"] += 1

print("=== Fixed Attrs vs User Preference ===")
for k, v in results.items():
    print(f"  {k}: {v}")

print(f"\n노이즈 케이스 샘플 (상위 5개):")
for c in noisy_cases[:5]:
    print(f"  {c}")