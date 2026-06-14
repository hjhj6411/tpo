"""
tests/test_exp_suite.py — mock verification of the whole exp suite.

No GPU, no servers, no network: FakeClient policies simulate models with known
behaviors (gold-picker, position-0-picker, TPO-picker, ...) and we assert each
experiment's analysis recovers exactly those behaviors. The two real failure
numbers from U001 plaid are a regression case for e4's scoring rules.

Run: cd pod_bench/exp && python tests/test_exp_suite.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import metrics
from common.clients import FakeClient
from common.data import CANON
from common.parsing import extract_choice
from common.permute import (base_perm, cyclic_orders, fixed_gold_order,
                            item_rng, verify_latin)
from common.prompts import IMG_MARK
from common.retrieval import apply_rule, build_query

import e1_position_bias.run as e1
import e2_bc_tradeoff.run as e2
import e3_input_conditions.run as e3
import e5_input_order.run as e5
import e6_dialogue_preference.run as e6

ok = 0
def check(name, cond, detail=""):
    global ok
    assert cond, f"FAIL {name}: {detail}"
    ok += 1
    print(f"  ✓ {name}")


def make_items(n=8):
    items = []
    for i in range(n):
        qid = f"U{i:03d}__scn__{'pattern' if i % 2 else 'color'}__{i:04d}"
        items.append({
            "query_id": qid,
            "active_axis": "pattern" if i % 2 else "color",
            "scenario_text": f"scenario {i}: snowy outdoor date",
            "profile_text": f"loves plaid and black (user {i})",
            "profile_images": [f"/img/profile_{i}.jpg"],
            "profile_entries": [
                {"explicit": f"This yellow shirt's color is my favorite ({i}).",
                 "implicit": f"This shirt's color is my favorite ({i}).",
                 "image": f"/img/e{i}_0.jpg"},
                {"explicit": f"I love this plaid pattern ({i}).",
                 "implicit": f"I love this pattern ({i}).",
                 "image": f"/img/e{i}_1.jpg"}],
            "options": {k: {"image": f"/img/{qid}_{k}.jpg"} for k in CANON},
            "gold": "A"})
    return items


def texts_of(messages):
    out = []
    for m in messages:
        for b in m["content"]:
            if b.get("type") == "text":
                out.append(b["text"])
    return " | ".join(out)


def imgs_of(messages):
    out = []
    for m in messages:
        for b in m["content"]:
            if b.get("type") == "image_url":
                out.append(b["image_url"]["url"].replace(IMG_MARK, ""))
    return out


# policies: simulate models with known behavior via task meta
def gold_picker(task, m, allowed):                 # perfect, unbiased model
    pos = task["order"].index(0)                   # canonical A
    return task["letters"][pos]

def pos0_picker(task, m, allowed):                 # pathological position bias
    return task["letters"][0]

def tpo_picker(task, m, allowed):                  # e2: always canonical B
    return task["letters"][task["pair"].index("B")]

CFG = {"seed": 17, "mode": "cyclic", "n_runs": 4, "condition": "both",
       "profile_mode": "text", "layout": "text_first",
       "conditions": ["both", "scenario_only", "profile_only", "none"],
       "p_list": [0.0, 1.0], "include_text_baseline": True}

items = make_items()

# ───────────────────────── permute ─────────────────────────
print("== permute ==")
base = base_perm(item_rng(17, "q1"))
orders = cyclic_orders(base, 4)
check("cyclic = latin square (every option visits every position once)",
      verify_latin(orders), orders)
check("seeded determinism", base == base_perm(item_rng(17, "q1")), "")
fg = [fixed_gold_order(base, 0, p) for p in range(4)]
check("fixed_gold: gold at requested position", all(o.index(0) == p
      for p, o in enumerate(fg)), fg)
d0 = [i for i in fg[0] if i != 0]
check("fixed_gold: distractor relative order constant (only gold moves)",
      all([i for i in o if i != 0] == d0 for o in fg), fg)

# ───────────────────────── e1 ─────────────────────────
print("\n== e1 position bias ==")
recs, an = e1.run(dict(CFG), items, FakeClient(gold_picker))
check("gold-picker: strict=1.0, luck_gap=0",
      an["overall"]["strict"]["acc"] == 1.0 and
      an["robust"]["luck_gap"] == 0.0, an["overall"])
check("gold-picker: acc-by-gold-pos spread 0",
      an["acc_by_gold_position"]["spread"] == 0.0, an["acc_by_gold_position"])

recs, an = e1.run(dict(CFG), items, FakeClient(pos0_picker))
pc = an["position_choice"]
check("pos0-picker: all choices at display A",
      pc["counts"][0] == sum(pc["counts"]) and pc["chi2"] > 50, pc)
g = an["acc_by_gold_position"]["by_gold_pos"]
check("pos0-picker: acc=1 only when gold at A",
      g["A"]["acc"] == 1.0 and all(g[l]["acc"] == 0.0 for l in "BCD"), g)
check("pos0-picker: cyclic mean acc = 0.25, robust = 0",
      an["robust"]["mean_acc"] == 0.25 and an["robust"]["robust_acc"] == 0.0,
      an["robust"])
cq = an["acc_by_gold_position"]["cochran_q"]
check("pos0-picker: Cochran Q flags position dependence",
      cq["Q"] > 10 and (cq["p"] is None or cq["p"] < 0.01), cq)

cfg_fg = dict(CFG); cfg_fg["mode"] = "fixed_gold"
recs, an = e1.run(cfg_fg, items, FakeClient(pos0_picker))
check("fixed_gold mode reproduces the A/B/C/D fixed-accuracy table",
      an["acc_by_gold_position"]["by_gold_pos"]["A"]["acc"] == 1.0,
      an["acc_by_gold_position"])

# ───────────────────────── e2 ─────────────────────────
print("\n== e2 bc tradeoff ==")
cfg2 = dict(CFG); cfg2["conditions"] = ["both", "scenario_only"]
recs, an = e2.run(cfg2, items, FakeClient(tpo_picker))
check("tpo-picker: P(B)=1.0 in every condition",
      all(b["overall"]["p_tpo"] == 1.0 for b in an["by_condition"].values()),
      an["by_condition"])
check("axis breakdown present (pattern/color)",
      set(an["by_condition"]["both"]["by_axis"]) == {"pattern", "color"},
      an["by_condition"]["both"]["by_axis"])
bal = an["side_balance"]["b_first_rate"]
check("B-side 50:50-ish under seeding", 0.2 <= bal <= 0.8, bal)
# cross-tab: fabricate a main run where model always chose canonical C
main_fake = [{"query_id": it["query_id"], "parsed": True, "strict": False,
              "choice_canon": "C"} for it in items]
_, an = e2.run(cfg2, items, FakeClient(tpo_picker), main_fake)
ct = an["cross_tab_vs_main_wrong"]
check("cross-tab: C-in-main vs B-in-BC -> agreement 0",
      ct["n_joint_wrong"] == len(items) and ct["direction_agreement"] == 0.0,
      ct)

# ───────────────────────── e3 ─────────────────────────
print("\n== e3 input conditions ==")
tasks = e3.build_tasks(items, CFG)
by_cond = {c: [t for t in tasks if t["condition"] == c]
           for c in CFG["conditions"]}
check("profile_only prompt has NO scenario text",
      all("Situation:" not in texts_of(t["messages"])
          and "User preferences:" in texts_of(t["messages"])
          for t in by_cond["profile_only"]), "")
check("none prompt has neither scenario nor profile",
      all("Situation:" not in texts_of(t["messages"])
          and "User preferences:" not in texts_of(t["messages"])
          and "loves wearing" not in texts_of(t["messages"])
          for t in by_cond["none"]), "")
check("same display order across conditions (paired design)",
      all(t["order"] == by_cond["both"][i]["order"]
          for c in CFG["conditions"] for i, t in enumerate(by_cond[c])), "")
recs, an = e3.run(dict(CFG), items, FakeClient(gold_picker))
check("gold-picker: strict=1 in every condition; mcnemar p=1",
      all(s["strict"]["acc"] == 1.0 for s in an["by_condition"].values())
      and all(v["p"] == 1.0 for v in an["mcnemar_vs_both"].values()), "")
check("profile_only sanity block emitted (tpo_at_chance flag)",
      "profile_only_sanity" in an and
      an["profile_only_sanity"]["tpo_at_chance"] in (True, False), an.keys())

# ───────────────────────── e5 ─────────────────────────
print("\n== e5 input order ==")
tasks = e5.build_tasks(items[:2], CFG)
tf = [t for t in tasks if t["layout"] == "text_first"][0]
imf = [t for t in tasks if t["layout"] == "img_first"][0]
t_tf, t_if = texts_of(tf["messages"]), texts_of(imf["messages"])
check("layouts produce different block orders, same content set",
      t_tf != t_if and sorted(t_tf.split(" | ")) == sorted(t_if.split(" | "))
      and imgs_of(tf["messages"]) == imgs_of(imf["messages"]), "")
check("img_first: options precede scenario",
      t_if.index("Option A:") < t_if.index("Situation:")
      and t_tf.index("Situation:") < t_tf.index("Option A:"), "")
recs, an = e5.run(dict(CFG), items, FakeClient(gold_picker))
check("gold-picker: both layouts 1.0, mcnemar p=1.0",
      an["by_layout"]["text_first"]["strict"]["acc"] == 1.0 and
      an["mcnemar_strict"]["p"] == 1.0, an)

# ───────────────────────── e6 ─────────────────────────
print("\n== e6 dialogue preference ==")
tasks = e6.build_tasks(items[:3], CFG)
p0 = [t for t in tasks if t["variant"] == "dialogue_p0"]
p1 = [t for t in tasks if t["variant"] == "dialogue_p1"]
check("p=0 -> all explicit sentences (attribute words present)",
      all("yellow" in texts_of(t["messages"]) for t in p0)
      and all(t["realized_implicit_frac"] == 0.0 for t in p0), "")
check("p=1 -> all implicit sentences (attribute words absent)",
      all("yellow" not in texts_of(t["messages"]) for t in p1)
      and all(t["realized_implicit_frac"] == 1.0 for t in p1), "")
check("dialogue final turn: scenario yes, profile block no",
      all("Situation:" in texts_of(t["messages"])
          and "User preferences:" not in texts_of(t["messages"])
          for t in p1), "")
check("assistant ack turns interleaved (no attribute leakage channel)",
      all(m["role"] == "assistant" for t in p1
          for m in t["messages"][1::2][:t["n_turns"]]), "")
recs, an = e6.run(dict(CFG), items, FakeClient(gold_picker))
check("gold-picker: every variant 1.0 + baseline gap 0",
      all(s["strict"]["acc"] == 1.0 for s in an["by_variant"].values())
      and an["explicit_vs_text_gap"] == 0.0, an)

# ───────────────────────── parsing / metrics ─────────────────────────
print("\n== parsing / metrics ==")
check("parsing variants",
      extract_choice("B", "ABCD")[0] == "B" and
      extract_choice(" (c) because...", "ABCD")[0] == "C" and
      extract_choice("The answer is D.", "ABCD")[0] == "D" and
      extract_choice("no letter here", "ABCD")[0] is None, "")
check("wilson sane", metrics.wilson(8, 10)[0] == 0.8 and
      metrics.wilson(8, 10)[1] < 0.8 < metrics.wilson(8, 10)[2], "")
check("mcnemar exact known value: b=8,c=2 -> p≈0.109",
      abs(metrics.mcnemar(8, 2)["p"] - 0.109375) < 1e-6, metrics.mcnemar(8, 2))
check("cochran_q zero on identical columns",
      metrics.cochran_q([[1, 1], [0, 0], [1, 1]])["Q"] == 0.0, "")
check("chi2_uniform flags concentration",
      metrics.chi2_uniform([40, 0, 0, 0])["chi2"] > 100, "")

# ───────────────────────── e4 scoring rules (regression) ──────────────
print("\n== e4 rules: U001 plaid regression ==")
# k1 = polka-dot tank (binary pcov fooled), k7 = windowpane (human best)
sims = [0.143, 0.121]
m = {"pcovB": [0.696, 0.727], "pcovT": [0.050, 0.727],
     "ccov": [0.696, 0.545]}
check("v1 prod_sim reproduces the failure (polka dot wins)",
      apply_rule("prod_sim", sims, m)[0] == 0, "")
check("v1 sum_norm also fails (sim noise inflated by minmax)",
      apply_rule("sum_norm", sims, m)[0] == 0, "")
check("cov_product fixes it (windowpane wins), beta=0.5 too",
      apply_rule("cov_product", sims, m, 1.0)[0] == 1 and
      apply_rule("cov_product", sims, m, 0.5)[0] == 1, "")
check("conjunctive: pcov=0 dies under cov_product despite ccov=1",
      apply_rule("cov_product", [0.2, 0.1],
                 {"pcovB": [1, 1], "pcovT": [0.0, 0.6], "ccov": [1.0, 0.8]},
                 1.0)[0] == 1, "")
check("sim_only ranks by similarity",
      apply_rule("sim_only", [0.1, 0.9, 0.5], 
                 {"pcovB": [0]*3, "pcovT": [0]*3, "ccov": [0]*3})[0] == 1, "")

# ───────────────────────── e4 query styles ─────────────────────────
q = build_query("product", "tank top", "black", "plaid", "pattern")
check("product query: person-free + all-over phrase",
      "person" not in q and "all-over plaid pattern" in q, q)
check("title query: bare attribute string",
      build_query("title", "tank top", "black", "plaid", "pattern")
      == "black plaid tank top", "")
check("person query preserved as v1 baseline",
      build_query("person", "coat", "beige", None, "color").startswith(
          "a fashion catalog photo of a person wearing a beige coat"), "")

print(f"\nALL {ok} CHECKS PASSED")
