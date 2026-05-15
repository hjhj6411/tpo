"""
Quality Audit
==============
Vision-essentiality 검증:

  1. Blind LLM control: 이미지 없이 텍스트(profile + query + 선지 search query)만 입력
     → 정답률이 충분히 낮아야 함 (≤ 40%)
  2. Captioner→LLM control: 각 선지 이미지를 captioning한 뒤 텍스트만 입력
     → 정답률이 caption 정보의 상한 (full VLM 대비 gap이 vision-essentiality 증거)

또한 최종 벤치마크 JSON을 생성합니다.

사용:
  python -m src.quality_audit

산출:
  data/labels/audit_metrics.json
  data/final/pod_bench.jsonl  (최종 벤치마크)
"""

import argparse
import random
from collections import Counter
from pathlib import Path

from .utils import (
    call_local_llm, call_local_vlm, parse_json_response,
    save_jsonl, load_jsonl, save_json, load_json, log_step,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    FINAL_DIR, LABELS_DIR, PROFILES_DIR, OPTIONS_DIR,
    QUERIES_DIR, IMAGES_DIR, FREE_MODELS,
    VISION_ESSENTIALITY_THRESHOLDS,
)


# ─────────────────────────────────────────────
# Blind LLM control
# ─────────────────────────────────────────────

BLIND_SYSTEM = """\
You are a personalization assistant. Given a user profile, a query, and 4 candidate options described in TEXT, choose the best option (A/B/C/D).

The 'best' option is one that matches both:
1. The user's enduring taste from their profile.
2. The situational requirement of the query.

Output ONLY the letter A, B, C, or D."""


def blind_llm_solve(profile: dict, plan: dict, query: dict,
                     judge_name: str = "local_qwen") -> str:
    """텍스트만으로 답을 추론. Vision-essential이라면 정답률이 낮아야 함."""
    options_text = []
    for opt_key in ["A", "B", "C", "D"]:
        opt = plan["options"][opt_key]
        # ⚠️ 텍스트 누출 가능 정보: search_query_en, distinguishing_attributes
        # 이걸 그대로 주면 LLM이 풀 수 있음 → 의도된 baseline
        attrs_str = ", ".join(
            f"{k}={v}" for k, v in opt["distinguishing_attributes"].items()
        )
        options_text.append(f"  {opt_key}: {attrs_str}")
    options_block = "\n".join(options_text)

    prompt = f"""USER PROFILE:
  {profile['narrative_profile']}

QUERY: "{query['query_text']}"

OPTIONS:
{options_block}

Which option (A/B/C/D) is the best choice?
Answer with only the letter."""

    cfg = FREE_MODELS["local_judge"]
    try:
        response = call_local_llm(prompt, system=BLIND_SYSTEM,
                                   api_base=cfg["api_base"],
                                   model=cfg["model_name"],
                                   max_tokens=8, temperature=0.0)
        for c in response.upper():
            if c in "ABCD":
                return c
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# Captioner→LLM control
# ─────────────────────────────────────────────

CAPTION_SYSTEM = """\
You are an image captioner. Describe the item in the image in 1-2 sentences,
mentioning visible attributes: category, color, silhouette, material, pattern, and details.

Be specific but neutral. Output only the caption."""


def caption_image(image_path: Path) -> str:
    """이미지를 캡셔닝."""
    try:
        return call_local_vlm(
            prompt="Describe this item in 1-2 sentences.",
            image_paths=[str(image_path)],
            system=CAPTION_SYSTEM,
            api_base=FREE_MODELS["vlm_evaluator"]["api_base"],
            model=FREE_MODELS["vlm_evaluator"]["model_name"],
            max_tokens=120, temperature=0.0,
        )
    except Exception as e:
        return f"(caption failed: {e})"


def captioner_llm_solve(profile: dict, query: dict,
                         image_paths: dict) -> str:
    """이미지를 캡셔닝한 뒤 LLM에게 텍스트로만 답을 묻는다."""
    captions = {}
    for opt_key in ["A", "B", "C", "D"]:
        if opt_key in image_paths:
            captions[opt_key] = caption_image(Path(image_paths[opt_key]))
        else:
            captions[opt_key] = "(missing)"

    options_block = "\n".join(
        f"  {k}: {v}" for k, v in captions.items()
    )

    prompt = f"""USER PROFILE:
  {profile['narrative_profile']}

QUERY: "{query['query_text']}"

OPTIONS (described by captions of their images):
{options_block}

Which option (A/B/C/D) is the best choice?
Answer with only the letter."""

    cfg = FREE_MODELS["local_judge"]
    try:
        response = call_local_llm(prompt, system=BLIND_SYSTEM,
                                   api_base=cfg["api_base"],
                                   model=cfg["model_name"],
                                   max_tokens=8, temperature=0.0)
        for c in response.upper():
            if c in "ABCD":
                return c
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# Full VLM (baseline)
# ─────────────────────────────────────────────

VLM_SOLVE_SYSTEM = """\
You are a personalization assistant. The user shows you their profile, a query, and 4 candidate items as images.

Choose the option (A/B/C/D) that best matches BOTH the user's enduring taste AND the situational requirement.

Output ONLY the letter A, B, C, or D."""


def full_vlm_solve(profile: dict, query: dict,
                    image_paths: dict) -> str:
    """Full VLM (이미지 + 텍스트)으로 답."""
    sorted_keys = ["A", "B", "C", "D"]
    paths = [str(image_paths[k]) for k in sorted_keys if k in image_paths]
    if len(paths) < 4:
        return None

    prompt = f"""USER PROFILE:
  {profile['narrative_profile']}

QUERY: "{query['query_text']}"

The 4 images shown are the candidate options A, B, C, D (in order).
Which option is the best choice?
Answer with only the letter A/B/C/D."""

    try:
        response = call_local_vlm(
            prompt=prompt,
            image_paths=paths,
            system=VLM_SOLVE_SYSTEM,
            api_base=FREE_MODELS["vlm_evaluator"]["api_base"],
            model=FREE_MODELS["vlm_evaluator"]["model_name"],
            max_tokens=8, temperature=0.0,
        )
        for c in response.upper():
            if c in "ABCD":
                return c
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# 최종 벤치마크 JSON 구성
# ─────────────────────────────────────────────

def build_final_benchmark(profiles: dict, queries: dict, plans: dict,
                           collection: dict, labels: dict,
                           output_path: Path) -> list[dict]:
    """모든 단계의 결과를 통합한 최종 인스턴스 생성.

    각 인스턴스가 가져야 할 정보:
      - user_id, profile.narrative
      - query (text + tpo_scenario + type)
      - 4 options (image_path + label)
      - 정답 (A 항상이지만 위치 셔플 처리는 평가 시점에)
    """
    final = []

    for query_id, plan in plans.items():
        if query_id not in queries:
            continue
        if query_id not in collection:
            continue
        if query_id not in labels:
            continue

        coll = collection[query_id]
        if not coll.get("all_collected"):
            continue

        # homogeneity 통과 여부
        homo = coll.get("homogeneity", {})
        if not homo.get("passed", True):
            continue

        # 라벨 검증 통과 여부
        lbl = labels[query_id]
        if not lbl.get("all_options_agree", False):
            # 부분 합의만으로도 통과시킬지 결정 — Phase 1에서는 일단 포함
            pass

        q = queries[query_id]
        prof = profiles[q["user_id"]]

        options_for_final = {}
        for opt_key in ["A", "B", "C", "D"]:
            options_for_final[opt_key] = {
                "image_path": coll["options"][opt_key]["image_path"],
                "label": lbl["consensus"][opt_key]["final_label"],
                "attributes": plan["options"][opt_key]["distinguishing_attributes"],
                "search_query": plan["options"][opt_key]["search_query_en"],
            }

        # 정답 찾기 (final_label == tpo_and_preference인 것)
        correct_keys = [k for k, v in options_for_final.items()
                       if v["label"] == "tpo_and_preference"]
        if len(correct_keys) != 1:
            # 정답이 모호하면 폐기
            continue

        final.append({
            "instance_id": query_id,
            "user_id": q["user_id"],
            "domain": q["domain"],
            "narrative_profile": prof["narrative_profile"],
            "query": {
                "text": q["query_text"],
                "type": q["query_type"],
                "tpo_scenario": q.get("tpo_scenario", {}),
            },
            "main_category": plan["main_category"],
            "options": options_for_final,
            "correct_option": correct_keys[0],
            "homogeneity": homo,
        })

    save_jsonl(final, output_path)
    return final


# ─────────────────────────────────────────────
# Audit 실행
# ─────────────────────────────────────────────

def run_audit(final_instances: list[dict],
              n_audit_samples: int = 50,
              enable_full_vlm: bool = True) -> dict:
    """Vision-essentiality audit."""
    log_step(f"Vision-Essentiality Audit (n={n_audit_samples})")

    if not final_instances:
        return {"error": "no final instances"}

    rng = random.Random(42)
    sample = rng.sample(final_instances, min(n_audit_samples, len(final_instances)))

    blind_results = []
    captioner_results = []
    vlm_results = []

    for i, inst in enumerate(sample):
        print(f"\n  [{i+1}/{len(sample)}] {inst['instance_id']}")
        correct = inst["correct_option"]

        # plan/profile/query 재구성 (간단히)
        plan = {
            "options": {k: {
                "distinguishing_attributes": v["attributes"],
                "search_query_en": v["search_query"],
                "label": v["label"],
            } for k, v in inst["options"].items()}
        }
        profile = {"narrative_profile": inst["narrative_profile"]}
        query = {"query_text": inst["query"]["text"]}
        image_paths = {k: v["image_path"] for k, v in inst["options"].items()
                       if v["image_path"]}

        # 1) Blind LLM
        try:
            ans = blind_llm_solve(profile, plan, query)
            blind_results.append(ans == correct)
            print(f"    blind: {ans} (correct: {correct}) → {'✓' if ans == correct else '✗'}")
        except Exception as e:
            print(f"    blind error: {e}")

        # 2) Captioner→LLM (시간 많이 듬, 작은 샘플만)
        if i < min(20, len(sample)):
            try:
                ans = captioner_llm_solve(profile, query, image_paths)
                captioner_results.append(ans == correct)
                print(f"    captioner→LLM: {ans} → {'✓' if ans == correct else '✗'}")
            except Exception as e:
                print(f"    captioner error: {e}")

        # 3) Full VLM
        if enable_full_vlm:
            try:
                ans = full_vlm_solve(profile, query, image_paths)
                vlm_results.append(ans == correct)
                print(f"    full VLM: {ans} → {'✓' if ans == correct else '✗'}")
            except Exception as e:
                print(f"    full VLM error: {e}")

    blind_acc = sum(blind_results) / max(len(blind_results), 1)
    cap_acc = sum(captioner_results) / max(len(captioner_results), 1)
    vlm_acc = sum(vlm_results) / max(len(vlm_results), 1)

    thresholds = VISION_ESSENTIALITY_THRESHOLDS

    audit = {
        "n_audited": len(sample),
        "blind_llm_accuracy": blind_acc,
        "captioner_llm_accuracy": cap_acc,
        "full_vlm_accuracy": vlm_acc,
        "blind_vs_full_gap": vlm_acc - blind_acc,
        "captioner_vs_full_gap": vlm_acc - cap_acc,
        "thresholds": thresholds,
        "vision_essentiality_passed": (
            blind_acc <= thresholds["blind_llm_max_acc"]
            and cap_acc <= thresholds["captioner_llm_max_acc"]
            and vlm_acc >= thresholds["full_vlm_min_acc"]
        ),
    }

    print(f"\n  Audit Summary:")
    print(f"    Blind LLM accuracy:       {blind_acc:.3f} (max allowed {thresholds['blind_llm_max_acc']})")
    print(f"    Captioner→LLM accuracy:   {cap_acc:.3f} (max allowed {thresholds['captioner_llm_max_acc']})")
    print(f"    Full VLM accuracy:        {vlm_acc:.3f} (min required {thresholds['full_vlm_min_acc']})")
    print(f"    Vision-essentiality: {'PASSED ✓' if audit['vision_essentiality_passed'] else 'FAILED ✗'}")

    return audit


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def run_pipeline(profile_path: Path, query_path: Path, plan_path: Path,
                 collection_path: Path, label_path: Path,
                 final_output: Path, audit_output: Path,
                 n_audit: int = 50,
                 enable_full_vlm: bool = True):
    """전체 quality audit 파이프라인 실행."""
    log_step("Quality Audit & Final Benchmark Assembly")

    profiles = {p["user_id"]: p for p in load_jsonl(profile_path)}
    queries = {q["query_id"]: q for q in load_jsonl(query_path)}
    plans = {p["query_id"]: p for p in load_jsonl(plan_path)}
    collection = {c["query_id"]: c for c in load_jsonl(collection_path)}
    labels = {l["query_id"]: l for l in load_jsonl(label_path)}

    print(f"  Profiles: {len(profiles)}, Queries: {len(queries)}")
    print(f"  Plans: {len(plans)}, Collected: {len(collection)}, Labels: {len(labels)}")

    # 최종 인스턴스 조립
    print("\n  Assembling final benchmark...")
    final_instances = build_final_benchmark(
        profiles, queries, plans, collection, labels, final_output,
    )
    print(f"  ✓ Assembled {len(final_instances)} final instances → {final_output}")

    # 도메인 통계
    domain_counts = Counter(i["domain"] for i in final_instances)
    qtype_counts = Counter(i["query"]["type"] for i in final_instances)
    print(f"  Domain breakdown: {dict(domain_counts)}")
    print(f"  Query type breakdown: {dict(qtype_counts)}")

    # Audit
    audit = run_audit(final_instances, n_audit, enable_full_vlm)
    save_json(audit, audit_output)
    print(f"\n  ✓ Audit metrics saved to {audit_output}")


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
    parser.add_argument("--n_audit", type=int, default=50)
    parser.add_argument("--no_full_vlm", action="store_true")
    args = parser.parse_args()

    run_pipeline(args.profile_path, args.query_path, args.plan_path,
                 args.collection_path, args.label_path,
                 args.final_output, args.audit_output,
                 args.n_audit, not args.no_full_vlm)


if __name__ == "__main__":
    main()
