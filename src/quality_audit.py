"""
Quality Audit + Final Assembly v2

(1) Assemble final benchmark with 3 profile variants and active_axis metadata.
(2) Vision-essentiality audit: blind LLM, captioner→LLM, full VLM.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from .utils import call_llm, call_vlm, save_jsonl, load_jsonl, save_json, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    FINAL_DIR, LABELS_DIR, PROFILES_DIR, OPTIONS_DIR,
    QUERIES_DIR, IMAGES_DIR, VISION_ESSENTIALITY_THRESHOLDS,
)


def build_final(profiles, queries, plans, collection, labels,
                  output_path, require_full_agreement=False):
    final = []
    for qid, plan in plans.items():
        if qid not in queries or qid not in collection or qid not in labels:
            continue
        coll = collection[qid]
        if not coll.get("all_collected"):
            continue
        homo = coll.get("homogeneity") or {}
        if not homo.get("passed", True):
            continue
        structure = coll.get("structure_preserved") or {}
        if structure and not structure.get("passed", True):
            continue
        lbl = labels[qid]
        if require_full_agreement and not lbl.get("all_options_agree", False):
            continue

        q = queries[qid]
        prof = profiles[q["user_id"]]
        options_final = {}
        for k in "ABCD":
            options_final[k] = {
                "image_path": coll["options"][k]["image_path"],
                "label": lbl["consensus"][k]["final_label"],
                "attributes": plan["options"][k]["attributes"],
                "image_source": coll["options"][k].get("source"),
            }
        correct = [k for k, v in options_final.items()
                    if v["label"] == "tpo_and_preference"]
        if len(correct) != 1:
            continue

        likes_str = ", ".join(prof["likes_keywords"]) or "(none)"
        dislikes_str = ", ".join(prof["dislikes_keywords"]) or "(none)"
        kw = f"Likes: {likes_str}. Dislikes: {dislikes_str}."

        final.append({
            "instance_id": qid, "user_id": q["user_id"], "domain": "fashion",
            "profile_variants": {
                "keyword_only": kw,
                "narrative_only": prof["narrative_profile"],
                "combined": prof["narrative_profile"] + " " + kw,
            },
            "query": {"text": q["query_text"], "type": q["query_type"],
                       "tpo_scenario": q.get("tpo_scenario") or {}},
            "active_axis": plan["active_axis"],
            "tpo_axis": plan.get("tpo_axis"),
            "fixed_attrs": plan.get("fixed_attrs"),
            "main_category": plan.get("main_category"),
            "options": options_final, "correct_option": correct[0],
            "homogeneity": homo, "structure_preserved": structure,
        })
    save_jsonl(final, output_path)
    return final


SOLVE_SYSTEM = """You are a fashion personalization assistant. Pick the BEST option (A/B/C/D)
that matches BOTH the user's enduring taste AND the situational requirement
of the query. Output ONLY the letter A, B, C, or D.
"""


def _parse_letter(text):
    for c in (text or "").upper():
        if c in "ABCD":
            return c
    return None


def blind_solve(instance, variant="combined"):
    opts = []
    for k in "ABCD":
        attrs = instance["options"][k]["attributes"]
        opts.append(f"  {k}: " + ", ".join(f"{a}={v}" for a, v in attrs.items()))
    prompt = f"""USER PROFILE:
  {instance['profile_variants'][variant]}

QUERY: "{instance['query']['text']}"

OPTIONS:
{chr(10).join(opts)}

Which option (A/B/C/D) is best? Answer with only the letter."""
    try:
        resp = call_llm(prompt, stage="blind_solver", system=SOLVE_SYSTEM,
                         max_tokens=8, temperature=0.0)
        return _parse_letter(resp)
    except Exception:
        return None


def caption_image(img_path):
    sys_msg = ("Describe this fashion item in 1-2 sentences: garment type, color, "
                "material, pattern, and any visible details.")
    try:
        return call_vlm(prompt="Describe this item.",
                          image_paths=[str(img_path)],
                          stage="captioner", system=sys_msg,
                          max_tokens=120, temperature=0.0)
    except Exception as e:
        return f"(caption failed: {e})"


def captioner_solve(instance, variant="combined"):
    caps = {}
    for k in "ABCD":
        p = instance["options"][k]["image_path"]
        caps[k] = caption_image(Path(p)) if p else "(missing)"
    opts = "\n".join(f"  {k}: {v}" for k, v in caps.items())
    prompt = f"""USER PROFILE:
  {instance['profile_variants'][variant]}

QUERY: "{instance['query']['text']}"

OPTIONS (image captions):
{opts}

Which option (A/B/C/D) is best? Answer with only the letter."""
    try:
        resp = call_llm(prompt, stage="blind_solver", system=SOLVE_SYSTEM,
                         max_tokens=8, temperature=0.0)
        return _parse_letter(resp)
    except Exception:
        return None


def full_vlm_solve(instance, variant="combined"):
    paths = [instance["options"][k]["image_path"] for k in "ABCD"]
    if any(p is None for p in paths):
        return None
    prompt = f"""USER PROFILE:
  {instance['profile_variants'][variant]}

QUERY: "{instance['query']['text']}"

The 4 images you see are options A, B, C, D in order.
Which option (A/B/C/D) is best? Answer with only the letter."""
    try:
        resp = call_vlm(prompt, image_paths=paths, stage="vlm_evaluator",
                         system=SOLVE_SYSTEM, max_tokens=8, temperature=0.0)
        return _parse_letter(resp)
    except Exception:
        return None


def run_audit(instances, n_audit=30, enable_vlm=True):
    log_step(f"Vision-Essentiality Audit (n={n_audit})")
    if not instances:
        return {"error": "no instances"}
    rng = random.Random(42)
    sample = rng.sample(instances, min(n_audit, len(instances)))
    blind_c, cap_c, vlm_c = [], [], []
    for i, inst in enumerate(sample):
        print(f"  [{i+1}/{len(sample)}] {inst['instance_id']}")
        correct = inst["correct_option"]
        try:
            b = blind_solve(inst)
            blind_c.append(b == correct)
            print(f"    blind: {b} (correct {correct})")
        except Exception as e:
            print(f"    blind err: {e}")
        if i < min(20, len(sample)):
            try:
                c = captioner_solve(inst)
                cap_c.append(c == correct)
                print(f"    caption→LLM: {c}")
            except Exception as e:
                print(f"    cap err: {e}")
        if enable_vlm:
            try:
                v = full_vlm_solve(inst)
                vlm_c.append(v == correct)
                print(f"    full VLM: {v}")
            except Exception as e:
                print(f"    vlm err: {e}")

    blind_a = sum(blind_c) / max(len(blind_c), 1)
    cap_a = sum(cap_c) / max(len(cap_c), 1)
    vlm_a = sum(vlm_c) / max(len(vlm_c), 1)
    th = VISION_ESSENTIALITY_THRESHOLDS
    audit = {
        "n_audited": len(sample),
        "blind_llm_accuracy": blind_a,
        "captioner_llm_accuracy": cap_a,
        "full_vlm_accuracy": vlm_a,
        "vision_on_off_gap": vlm_a - blind_a,
        "captioner_vs_vlm_gap": vlm_a - cap_a,
        "thresholds": th,
        "passed": (blind_a <= th["blind_llm_max_acc"]
                    and cap_a <= th["captioner_llm_max_acc"]
                    and vlm_a >= th["full_vlm_min_acc"]),
    }
    print(f"\n  blind:    {blind_a:.3f} (≤{th['blind_llm_max_acc']})")
    print(f"  caption→LLM: {cap_a:.3f} (≤{th['captioner_llm_max_acc']})")
    print(f"  full VLM: {vlm_a:.3f} (≥{th['full_vlm_min_acc']})")
    print(f"  vision-essentiality {'PASSED' if audit['passed'] else 'FAILED'}")
    return audit


def run_pipeline(profile_path, query_path, plan_path, collection_path,
                  label_path, final_output, audit_output,
                  n_audit=30, enable_vlm=True, require_full_agreement=False):
    log_step("Quality Audit + Final Assembly")
    profiles = {p["user_id"]: p for p in load_jsonl(profile_path)}
    queries = {q["query_id"]: q for q in load_jsonl(query_path)}
    plans = {p["query_id"]: p for p in load_jsonl(plan_path)}
    collection = {c["query_id"]: c for c in load_jsonl(collection_path)}
    labels = {l["query_id"]: l for l in load_jsonl(label_path)}
    print(f"  profiles={len(profiles)}, queries={len(queries)}, "
          f"plans={len(plans)}, collected={len(collection)}, labels={len(labels)}")

    instances = build_final(profiles, queries, plans, collection, labels,
                              final_output, require_full_agreement)
    print(f"  ✓ {len(instances)} final instances → {final_output}")
    print(f"  active_axis: {dict(Counter(i['active_axis'] for i in instances))}")
    print(f"  query_type:  {dict(Counter(i['query']['type'] for i in instances))}")

    audit = run_audit(instances, n_audit, enable_vlm)
    save_json(audit, audit_output)
    print(f"  ✓ Audit → {audit_output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", type=Path, default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--query_path", type=Path, default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--plan_path", type=Path, default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--collection_path", type=Path, default=IMAGES_DIR / "collection_log.jsonl")
    parser.add_argument("--label_path", type=Path, default=LABELS_DIR / "labels.jsonl")
    parser.add_argument("--final_output", type=Path, default=FINAL_DIR / "pod_bench.jsonl")
    parser.add_argument("--audit_output", type=Path, default=LABELS_DIR / "audit_metrics.json")
    parser.add_argument("--n_audit", type=int, default=30)
    parser.add_argument("--no_full_vlm", action="store_true")
    parser.add_argument("--require_full_agreement", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.profile_path, args.query_path, args.plan_path,
                  args.collection_path, args.label_path,
                  args.final_output, args.audit_output,
                  args.n_audit, not args.no_full_vlm,
                  args.require_full_agreement)


if __name__ == "__main__":
    main()
