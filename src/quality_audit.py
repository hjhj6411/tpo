"""
Quality Audit + Final Benchmark Assembly
=========================================

Two responsibilities:

  1. Assemble final benchmark JSONL from all upstream artifacts.
     Each instance carries THREE profile representations to support ablation:
       - keyword_profile: likes/dislikes keyword lists
       - narrative_profile: English paragraph
       - combined_profile: both

  2. Vision-essentiality audit on a sample:
       - Blind LLM:        attributes-only text input (upper bound for text)
       - Captioner→LLM:    image captions + text input (caption ceiling)
       - Full VLM:         images + text input (multimodal ceiling)
"""

import argparse
import random
from collections import Counter
from pathlib import Path

from .utils import (
    call_local_llm, call_local_vlm,
    save_jsonl, load_jsonl, save_json, log_step,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    FINAL_DIR, LABELS_DIR, PROFILES_DIR, OPTIONS_DIR,
    QUERIES_DIR, IMAGES_DIR, FREE_MODELS,
    VISION_ESSENTIALITY_THRESHOLDS,
)


# ─────────────────────────────────────────────
# Benchmark assembly
# ─────────────────────────────────────────────

def build_final(profiles: dict, queries: dict, plans: dict,
                 collection: dict, labels: dict,
                 output_path: Path,
                 require_full_agreement: bool = False) -> list:
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
                "search_query": plan["options"][k]["search_query"],
                "image_source": coll["options"][k].get("source"),
            }

        correct_keys = [k for k, v in options_final.items()
                        if v["label"] == "tpo_and_preference"]
        if len(correct_keys) != 1:
            continue

        # Profile variants for ablation
        likes_str = ", ".join(prof["likes_keywords"]) or "(none)"
        dislikes_str = ", ".join(prof["dislikes_keywords"]) or "(none)"
        keyword_profile = f"Likes: {likes_str}. Dislikes: {dislikes_str}."

        final.append({
            "instance_id": qid,
            "user_id": q["user_id"],
            "domain": "fashion",
            "profile_variants": {
                "keyword_only": keyword_profile,
                "narrative_only": prof["narrative_profile"],
                "combined": (
                    prof["narrative_profile"] + " " + keyword_profile
                ),
            },
            "query": {
                "text": q["query_text"],
                "type": q["query_type"],
                "tpo_scenario": q.get("tpo_scenario") or {},
            },
            "main_category": plan["main_category"],
            "options": options_final,
            "correct_option": correct_keys[0],
            "homogeneity": homo,
        })

    save_jsonl(final, output_path)
    return final


# ─────────────────────────────────────────────
# Vision-essentiality controls
# ─────────────────────────────────────────────

SOLVE_SYSTEM = """\
You are a fashion personalization assistant. Pick the BEST option (A/B/C/D)
that matches BOTH the user's enduring taste AND the situational requirement
of the query. Output ONLY the letter A, B, C, or D.
"""


def _parse_letter(text: str) -> str:
    for c in (text or "").upper():
        if c in "ABCD":
            return c
    return None


def blind_solve(instance: dict, profile_variant: str = "combined") -> str:
    """Text-only baseline: profile + query + option attributes (no images)."""
    opts_block = []
    for k in "ABCD":
        attrs = instance["options"][k]["attributes"]
        opts_block.append(f"  {k}: " + ", ".join(f"{a}={v}" for a, v in attrs.items()))
    prompt = f"""USER PROFILE:
  {instance['profile_variants'][profile_variant]}

QUERY: "{instance['query']['text']}"

OPTIONS:
{chr(10).join(opts_block)}

Which option (A/B/C/D) is best? Answer with only the letter."""

    cfg = FREE_MODELS["local_judge"]
    try:
        resp = call_local_llm(prompt, system=SOLVE_SYSTEM,
                              api_base=cfg["api_base"], model=cfg["model_name"],
                              max_tokens=8, temperature=0.0)
        return _parse_letter(resp)
    except Exception:
        return None


def caption_image(img_path: Path) -> str:
    sys_msg = "Describe this fashion item in 1-2 sentences: garment type, color, material, pattern, and any visible details. Output only the description."
    try:
        return call_local_vlm(
            prompt="Describe this item.", image_paths=[str(img_path)],
            system=sys_msg,
            api_base=FREE_MODELS["vlm_evaluator"]["api_base"],
            model=FREE_MODELS["vlm_evaluator"]["model_name"],
            max_tokens=120, temperature=0.0,
        )
    except Exception as e:
        return f"(caption failed: {e})"


def captioner_llm_solve(instance: dict, profile_variant: str = "combined") -> str:
    captions = {}
    for k in "ABCD":
        p = instance["options"][k]["image_path"]
        captions[k] = caption_image(Path(p)) if p else "(missing)"
    opts_block = "\n".join(f"  {k}: {v}" for k, v in captions.items())

    prompt = f"""USER PROFILE:
  {instance['profile_variants'][profile_variant]}

QUERY: "{instance['query']['text']}"

OPTIONS (described by image captions):
{opts_block}

Which option (A/B/C/D) is best? Answer with only the letter."""

    cfg = FREE_MODELS["local_judge"]
    try:
        resp = call_local_llm(prompt, system=SOLVE_SYSTEM,
                              api_base=cfg["api_base"], model=cfg["model_name"],
                              max_tokens=8, temperature=0.0)
        return _parse_letter(resp)
    except Exception:
        return None


def full_vlm_solve(instance: dict, profile_variant: str = "combined") -> str:
    paths = [instance["options"][k]["image_path"] for k in "ABCD"]
    if any(p is None for p in paths):
        return None

    prompt = f"""USER PROFILE:
  {instance['profile_variants'][profile_variant]}

QUERY: "{instance['query']['text']}"

The 4 images you see are options A, B, C, D in order.
Which option (A/B/C/D) is best? Answer with only the letter."""

    try:
        resp = call_local_vlm(
            prompt=prompt, image_paths=paths, system=SOLVE_SYSTEM,
            api_base=FREE_MODELS["vlm_evaluator"]["api_base"],
            model=FREE_MODELS["vlm_evaluator"]["model_name"],
            max_tokens=8, temperature=0.0,
        )
        return _parse_letter(resp)
    except Exception:
        return None


def run_audit(instances: list, n_audit: int = 50,
              enable_full_vlm: bool = True) -> dict:
    log_step(f"Vision-Essentiality Audit (n={n_audit})")
    if not instances:
        return {"error": "no instances"}

    rng = random.Random(42)
    sample = rng.sample(instances, min(n_audit, len(instances)))

    blind_correct, cap_correct, vlm_correct = [], [], []
    for i, inst in enumerate(sample):
        print(f"  [{i+1}/{len(sample)}] {inst['instance_id']}")
        correct = inst["correct_option"]
        try:
            b = blind_solve(inst)
            blind_correct.append(b == correct)
            print(f"    blind: {b} (correct {correct})")
        except Exception as e:
            print(f"    blind err: {e}")

        if i < min(20, len(sample)):
            try:
                c = captioner_llm_solve(inst)
                cap_correct.append(c == correct)
                print(f"    caption→LLM: {c}")
            except Exception as e:
                print(f"    caption err: {e}")

        if enable_full_vlm:
            try:
                v = full_vlm_solve(inst)
                vlm_correct.append(v == correct)
                print(f"    full VLM: {v}")
            except Exception as e:
                print(f"    vlm err: {e}")

    blind_acc = sum(blind_correct) / max(len(blind_correct), 1)
    cap_acc = sum(cap_correct) / max(len(cap_correct), 1)
    vlm_acc = sum(vlm_correct) / max(len(vlm_correct), 1)

    th = VISION_ESSENTIALITY_THRESHOLDS
    audit = {
        "n_audited": len(sample),
        "blind_llm_accuracy": blind_acc,
        "captioner_llm_accuracy": cap_acc,
        "full_vlm_accuracy": vlm_acc,
        "vision_on_off_gap": vlm_acc - blind_acc,
        "captioner_vs_vlm_gap": vlm_acc - cap_acc,
        "thresholds": th,
        "passed": (blind_acc <= th["blind_llm_max_acc"]
                   and cap_acc <= th["captioner_llm_max_acc"]
                   and vlm_acc >= th["full_vlm_min_acc"]),
    }
    print(f"\n  blind:    {blind_acc:.3f} (≤{th['blind_llm_max_acc']})")
    print(f"  caption→LLM: {cap_acc:.3f} (≤{th['captioner_llm_max_acc']})")
    print(f"  full VLM: {vlm_acc:.3f} (≥{th['full_vlm_min_acc']})")
    print(f"  vision-essentiality {'PASSED' if audit['passed'] else 'FAILED'}")
    return audit


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_pipeline(profile_path: Path, query_path: Path, plan_path: Path,
                 collection_path: Path, label_path: Path,
                 final_output: Path, audit_output: Path,
                 n_audit: int = 50, enable_full_vlm: bool = True,
                 require_full_agreement: bool = False):
    log_step("Quality Audit + Final Assembly")

    profiles = {p["user_id"]: p for p in load_jsonl(profile_path)}
    queries = {q["query_id"]: q for q in load_jsonl(query_path)}
    plans = {p["query_id"]: p for p in load_jsonl(plan_path)}
    collection = {c["query_id"]: c for c in load_jsonl(collection_path)}
    labels = {l["query_id"]: l for l in load_jsonl(label_path)}

    print(f"  Profiles: {len(profiles)}, Queries: {len(queries)}")
    print(f"  Plans: {len(plans)}, Collected: {len(collection)}, Labels: {len(labels)}")

    instances = build_final(profiles, queries, plans, collection, labels,
                             final_output, require_full_agreement)
    print(f"  ✓ {len(instances)} final instances → {final_output}")

    qtype_counts = Counter(i["query"]["type"] for i in instances)
    print(f"  Query type breakdown: {dict(qtype_counts)}")

    audit = run_audit(instances, n_audit, enable_full_vlm)
    save_json(audit, audit_output)
    print(f"  ✓ Audit metrics → {audit_output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--query_path", type=Path,
                        default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--plan_path", type=Path,
                        default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--collection_path", type=Path,
                        default=IMAGES_DIR / "collection_log.jsonl")
    parser.add_argument("--label_path", type=Path,
                        default=LABELS_DIR / "labels.jsonl")
    parser.add_argument("--final_output", type=Path,
                        default=FINAL_DIR / "pod_bench.jsonl")
    parser.add_argument("--audit_output", type=Path,
                        default=LABELS_DIR / "audit_metrics.json")
    parser.add_argument("--n_audit", type=int, default=30)
    parser.add_argument("--no_full_vlm", action="store_true")
    parser.add_argument("--require_full_agreement", action="store_true",
                        help="Drop instances where judges + rule + plan don't all agree")
    args = parser.parse_args()
    run_pipeline(args.profile_path, args.query_path, args.plan_path,
                 args.collection_path, args.label_path,
                 args.final_output, args.audit_output,
                 args.n_audit, not args.no_full_vlm,
                 args.require_full_agreement)


if __name__ == "__main__":
    main()
