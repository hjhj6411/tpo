import json
from collections import defaultdict

PROFILES_PATH = "data/profiles/profiles.jsonl"
PLANS_PATH = "data/options/option_plans.jsonl"

profiles = {}
with open(PROFILES_PATH) as f:
    for line in f:
        p = json.loads(line)
        profiles[p["user_id"]] = p

errors = defaultdict(list)

with open(PLANS_PATH) as f:
    for line in f:
        plan = json.loads(line)
        uid = plan["user_id"]
        active_axis = plan["active_axis"]
        options = plan["options"]
        struct = profiles[uid]["structured_attributes"]
        likes = set(struct.get(active_axis, {}).get("likes", []))
        dislikes = set(struct.get(active_axis, {}).get("dislikes", []))
        qid = plan["query_id"]

        a_val = options["A"]["attributes"].get(active_axis)
        b_val = options["B"]["attributes"].get(active_axis)
        c_val = options["C"]["attributes"].get(active_axis)
        d_val = options["D"]["attributes"].get(active_axis)

        # A: must be in likes
        if a_val not in likes:
            errors["A_not_in_likes"].append({"qid": qid, "val": a_val})

        # B: must NOT be in likes
        if b_val in likes:
            errors["B_in_likes"].append({"qid": qid, "val": b_val})

        # C: must be in likes
        if c_val not in likes:
            errors["C_not_in_likes"].append({"qid": qid, "val": c_val})

        # D: must NOT be in likes
        if d_val in likes:
            errors["D_in_likes"].append({"qid": qid, "val": d_val})

        # A/C same value → 구분 불가
        if a_val == c_val:
            errors["AC_same_value"].append({"qid": qid, "val": a_val})

        # B/D same value → 구분 불가
        if b_val == d_val:
            errors["BD_same_value"].append({"qid": qid, "val": b_val})

        # 4개 중 중복값 존재
        vals = [a_val, b_val, c_val, d_val]
        if len(set(vals)) < 4:
            errors["duplicate_option_vals"].append({"qid": qid, "vals": vals})

print("=== Option Validity Check ===")
if not errors:
    print("  ✅ All options valid!")
else:
    for k, v in errors.items():
        print(f"  {k}: {len(v)} cases")
        for c in v[:3]:
            print(f"    {c}")