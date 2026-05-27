"""
Quality Audit v2 — Final assembly + vision-essentiality gate.

Changes from v1:
  - scenario_id, scenario_archetype added to final record
  - Blind solver prompt includes scenario name for fair comparison
  - Per-archetype breakdown in essentiality stats
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .utils import (
    call_llm, call_vlm, parse_json_response,
    save_jsonl, load_jsonl, save_json, log_step,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    FINAL_DIR, LABELS_DIR, IMAGES_DIR, OPTIONS_DIR, QUERIES_DIR, PROFILES_DIR,
    VISION_ESSENTIALITY_THRESHOLDS,
)


# ── Blind LLM solver ─────────────────────────────────────

BLIND_SYSTEM = """You are a fashion recommendation expert. Given a user profile, a query,
and 4 options described only by text attributes, pick the BEST option.

The best option satisfies BOTH the user's personal preference AND the
situational requirements (weather, formality, occasion, etc.).

Output ONLY JSON: {"answer": "A" or "B" or "C" or "D", "rationale": "..."}"""


def _build_blind_prompt(record):
    profile = record["narrative_profile"]
    query = record["query_text"]
    scenario_name = record.get("scenario_name", "")
    likes = ", ".join(record.get("likes_keywords", [])) or "(none)"
    dislikes = ", ".join(record.get("dislikes_keywords", [])) or "(none)"

    opts = []
    for k in "ABCD":
        attrs = record["options"][k]["attributes"]
        opts.append(f"  {k}: " + ", ".join(f"{a}={v}" for a, v in attrs.items()))

    return f"""USER PROFILE: {profile}
  likes: {likes}
  dislikes: {dislikes}

QUERY: "{query}"
SCENARIO: {scenario_name}

OPTIONS (text attributes only — no images):
{chr(10).join(opts)}

Which option BEST matches both the user's taste and the situation?
Output JSON: {{"answer": "A"/"B"/"C"/"D", "rationale": "..."}}"""


def blind_solve(record, provider_override=None):
    prompt = _build_blind_prompt(record)
    try:
        resp = call_llm(prompt, stage="blind_solver", system=BLIND_SYSTEM,
                        provider_override=provider_override)
        parsed = parse_json_response(resp)
        if parsed and parsed.get("answer") in ("A", "B", "C", "D"):
            return parsed["answer"]
    except Exception:
        pass
    return None


# ── Captioner LLM solver ─────────────────────────────────

CAPTIONER_SYSTEM = """You are a fashion image captioner. Describe the garment in the image:
color, pattern, garment type, and style. Be concise (1-2 sentences).
Output ONLY the description text."""


def caption_image(image_path, provider_override=None):
    try:
        resp = call_vlm(
            prompt="Describe this fashion item: color, pattern, garment type, style.",
            image_paths=[str(image_path)],
            stage="captioner", system=CAPTIONER_SYSTEM,
            max_tokens=128, temperature=0.0,
            provider_override=provider_override,
        )
        return resp.strip() if resp else ""
    except Exception:
        return ""


CAPTIONER_SOLVER_SYSTEM = """You are a fashion recommendation expert. Given a user profile,
a query, and 4 options described by captions (not images), pick the BEST option.

Output ONLY JSON: {"answer": "A" or "B" or "C" or "D", "rationale": "..."}"""


def captioner_solve(record, captions, provider_override=None):
    profile = record["narrative_profile"]
    query = record["query_text"]
    scenario_name = record.get("scenario_name", "")
    likes = ", ".join(record.get("likes_keywords", [])) or "(none)"
    dislikes = ", ".join(record.get("dislikes_keywords", [])) or "(none)"

    opts = []
    for k in "ABCD":
        opts.append(f"  {k}: {captions.get(k, '(no caption)')}")

    prompt = f"""USER PROFILE: {profile}
  likes: {likes}
  dislikes: {dislikes}

QUERY: "{query}"
SCENARIO: {scenario_name}

OPTIONS (described by captions — no images):
{chr(10).join(opts)}

Which option BEST matches both the user's taste and the situation?
Output JSON: {{"answer": "A"/"B"/"C"/"D", "rationale": "..."}}"""

    try:
        resp = call_llm(prompt, stage="blind_solver",
                        system=CAPTIONER_SOLVER_SYSTEM,
                        provider_override=provider_override)
        parsed = parse_json_response(resp)
        if parsed and parsed.get("answer") in ("A", "B", "C", "D"):
            return parsed["answer"]
    except Exception:
        pass
    return None


# ── VLM solver ────────────────────────────────────────────

VLM_SYSTEM = """You are a fashion recommendation VLM. Given a user profile, a query,
and 4 product images, pick the BEST option.

Output ONLY JSON: {"answer": "A" or "B" or "C" or "D", "rationale": "..."}"""


def vlm_solve(record, image_paths, provider_override=None):
    profile = record["narrative_profile"]
    query = record["query_text"]
    scenario_name = record.get("scenario_name", "")
    likes = ", ".join(record.get("likes_keywords", [])) or "(none)"
    dislikes = ", ".join(record.get("dislikes_keywords", [])) or "(none)"

    prompt = f"""USER PROFILE: {profile}
  likes: {likes}
  dislikes: {dislikes}

QUERY: "{query}"
SCENARIO: {scenario_name}

The 4 images are options A, B, C, D (in order).
Which BEST matches both the user's taste and the situation?
Output JSON: {{"answer": "A"/"B"/"C"/"D", "rationale": "..."}}"""

    try:
        resp = call_vlm(prompt, image_paths=image_paths,
                        stage="vlm_evaluator", system=VLM_SYSTEM,
                        provider_override=provider_override)
        parsed = parse_json_response(resp)
        if parsed and parsed.get("answer") in ("A", "B", "C", "D"):
            return parsed["answer"]
    except Exception:
        pass
    return None


# ── Assembly + audit ──────────────────────────────────────

def assemble_final_record(label_rec, plan, query, profile, image_rec):
    """Merge all pipeline stages into one final benchmark record."""
    options = {}
    for k in "ABCD":
        opt_plan = plan["options"][k]
        img_info = image_rec["options"].get(k, {}) if image_rec else {}
        consensus = label_rec["consensus"][k]
        options[k] = {
            "attributes": opt_plan["attributes"],
            "search_query": opt_plan["search_query"],
            "label": consensus["final_label"],
            "plan_label": opt_plan["label"],
            "image_path": img_info.get("image_path"),
            "image_source": img_info.get("source"),
        }

    # Correct answer = tpo_and_preference
    correct = None
    for k, opt in options.items():
        if opt["label"] == "tpo_and_preference":
            correct = k
            break

    return {
        "instance_id": label_rec["query_id"],
        "query_id": label_rec["query_id"],
        "user_id": label_rec["user_id"],
        "scenario_id": label_rec.get("scenario_id", query.get("scenario_id")),
        "scenario_archetype": query.get("scenario_archetype"),
        "scenario_name": query.get("scenario_name"),
        "domain": "fashion",
        "query_type": query.get("query_type"),
        "query_text": query["query_text"],
        "active_axis": label_rec["active_axis"],
        "narrative_profile": profile["narrative_profile"],
        "preference_archetype": profile.get("preference_archetype"),
        "likes_keywords": profile.get("likes_keywords", []),
        "dislikes_keywords": profile.get("dislikes_keywords", []),
        "options": options,
        "correct_answer": correct,
        "all_labels_agree": label_rec.get("all_options_agree", False),
        "images_complete": image_rec["all_collected"] if image_rec else False,
        "homogeneity": image_rec.get("homogeneity") if image_rec else None,
    }


def run_pipeline(label_path, plan_path, query_path, profile_path, image_path,
                 output_path, essentiality_path, limit=0,
                 run_blind=True, run_captioner=True, run_vlm=True):
    log_step("Quality Audit v2 — assembly + vision-essentiality")

    labels = {r["query_id"]: r for r in load_jsonl(label_path)}
    plans = {p["query_id"]: p for p in load_jsonl(plan_path)}
    queries = {q["query_id"]: q for q in load_jsonl(query_path)}
    profiles = {p["user_id"]: p for p in load_jsonl(profile_path)}
    images = {r["query_id"]: r for r in load_jsonl(image_path)} if image_path.exists() else {}

    print(f"  labels={len(labels)}, plans={len(plans)}, "
          f"queries={len(queries)}, profiles={len(profiles)}, images={len(images)}")

    # Assemble
    records = []
    for qid, lbl in labels.items():
        if qid not in plans or qid not in queries:
            continue
        uid = lbl["user_id"]
        if uid not in profiles:
            continue
        img = images.get(qid)
        rec = assemble_final_record(lbl, plans[qid], queries[qid],
                                    profiles[uid], img)
        if rec["correct_answer"] is None:
            continue
        records.append(rec)

    if limit > 0:
        records = records[:limit]

    print(f"  Assembled {len(records)} final records")

    # Vision-essentiality probes
    blind_correct = 0
    captioner_correct = 0
    vlm_correct = 0
    n_tested = 0
    by_archetype = defaultdict(lambda: {"blind": 0, "cap": 0, "vlm": 0, "n": 0})

    for i, rec in enumerate(records):
        gt = rec["correct_answer"]
        arch = rec.get("scenario_archetype", "unknown")
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(records)}] essentiality probes...")

        if run_blind:
            ans = blind_solve(rec)
            rec["blind_answer"] = ans
            if ans == gt:
                blind_correct += 1
                by_archetype[arch]["blind"] += 1

        if run_captioner and rec.get("images_complete"):
            captions = {}
            for k in "ABCD":
                ip = rec["options"][k].get("image_path")
                if ip:
                    captions[k] = caption_image(ip)
                else:
                    captions[k] = ", ".join(f"{a}={v}" for a, v
                                            in rec["options"][k]["attributes"].items())
            rec["captions"] = captions
            ans = captioner_solve(rec, captions)
            rec["captioner_answer"] = ans
            if ans == gt:
                captioner_correct += 1
                by_archetype[arch]["cap"] += 1

        if run_vlm and rec.get("images_complete"):
            img_paths = [rec["options"][k]["image_path"] for k in "ABCD"]
            if all(img_paths):
                ans = vlm_solve(rec, img_paths)
                rec["vlm_answer"] = ans
                if ans == gt:
                    vlm_correct += 1
                    by_archetype[arch]["vlm"] += 1

        n_tested += 1
        by_archetype[arch]["n"] += 1

    save_jsonl(records, output_path)
    print(f"\n  ✓ Saved {len(records)} final records")

    # Essentiality report
    th = VISION_ESSENTIALITY_THRESHOLDS
    blind_acc = blind_correct / max(n_tested, 1)
    cap_acc = captioner_correct / max(n_tested, 1)
    vlm_acc = vlm_correct / max(n_tested, 1)

    essentiality = {
        "n_tested": n_tested,
        "blind_llm_acc": blind_acc,
        "captioner_llm_acc": cap_acc,
        "full_vlm_acc": vlm_acc,
        "thresholds": th,
        "passed": (blind_acc <= th["blind_llm_max_acc"]
                   and cap_acc <= th["captioner_llm_max_acc"]
                   and vlm_acc >= th["full_vlm_min_acc"]),
        "by_archetype": {},
    }
    for arch, s in by_archetype.items():
        n = max(s["n"], 1)
        essentiality["by_archetype"][arch] = {
            "n": s["n"],
            "blind_acc": s["blind"] / n,
            "captioner_acc": s["cap"] / n,
            "vlm_acc": s["vlm"] / n,
        }

    save_json(essentiality, essentiality_path)
    print(f"\n  Vision Essentiality:")
    print(f"    blind LLM:     {blind_acc:.3f} (≤{th['blind_llm_max_acc']})")
    print(f"    captioner LLM: {cap_acc:.3f} (≤{th['captioner_llm_max_acc']})")
    print(f"    full VLM:      {vlm_acc:.3f} (≥{th['full_vlm_min_acc']})")
    print(f"    passed: {essentiality['passed']}")
    for arch, s in essentiality["by_archetype"].items():
        print(f"    [{arch}] n={s['n']} blind={s['blind_acc']:.2f} "
              f"cap={s['captioner_acc']:.2f} vlm={s['vlm_acc']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", type=Path,
                        default=LABELS_DIR / "labels.jsonl")
    parser.add_argument("--plan_path", type=Path,
                        default=Path("data/options/option_plans.jsonl"))
    parser.add_argument("--query_path", type=Path,
                        default=Path("data/queries/queries.jsonl"))
    parser.add_argument("--profile_path", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--image_path", type=Path,
                        default=IMAGES_DIR / "collection_log.jsonl")
    parser.add_argument("--output", type=Path,
                        default=FINAL_DIR / "benchmark.jsonl")
    parser.add_argument("--essentiality_output", type=Path,
                        default=FINAL_DIR / "essentiality.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip_blind", action="store_true")
    parser.add_argument("--skip_captioner", action="store_true")
    parser.add_argument("--skip_vlm", action="store_true")
    args = parser.parse_args()
    run_pipeline(
        args.label_path, args.plan_path, args.query_path, args.profile_path,
        args.image_path, args.output, args.essentiality_output, args.limit,
        run_blind=not args.skip_blind, run_captioner=not args.skip_captioner,
        run_vlm=not args.skip_vlm,
    )


if __name__ == "__main__":
    main()
