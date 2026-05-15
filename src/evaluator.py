"""
Evaluator
==========
완성된 POD-Bench에서 VLM을 평가합니다.

평가 항목:
  1. 정답 정확도 (overall accuracy)
  2. 도메인별 / query-type별 정확도
  3. **혼동 행렬 (confusion matrix)**: 어떤 오답 패턴을 보이는가
     - B 선지 오답이 많으면 → TPO 과민 (PCogAlign 계열 실패)
     - C 선지 오답이 많으면 → 선호 과민 (Whose Boat? 계열 실패)
  4. Position bias 통제: 선지 위치를 셔플하여 2회 평가

사용:
  python -m src.evaluator --model qwen2.5-vl-7b

산출:
  data/evaluation/<model>/results.jsonl
  data/evaluation/<model>/metrics.json
"""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from .utils import (
    call_local_vlm, save_jsonl, load_jsonl, save_json, log_step,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import FINAL_DIR, FREE_MODELS


EVAL_DIR = FINAL_DIR.parent / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 평가 프롬프트
# ─────────────────────────────────────────────

EVAL_SYSTEM = """\
You are a personalization assistant. You see a user's profile, a query, and 4 candidate options as images.

Choose the option (A/B/C/D) that best matches BOTH:
1. The user's enduring taste from their profile
2. The situational requirement of the query

Output ONLY the letter A, B, C, or D."""


def build_eval_prompt(instance: dict, option_order: list[str]) -> str:
    """평가 프롬프트 생성. option_order는 셔플된 순서."""
    return f"""USER PROFILE:
  {instance['narrative_profile']}

QUERY: "{instance['query']['text']}"

The 4 images shown are options (in order): {', '.join(option_order)}.

Which option (A/B/C/D) is the best choice?
Answer with only the letter."""


# ─────────────────────────────────────────────
# 단일 인스턴스 평가
# ─────────────────────────────────────────────

def evaluate_instance(instance: dict, model_cfg: dict,
                      shuffle_seed: int) -> dict:
    """위치 셔플 + VLM 호출 + 결과 반환."""
    options = instance["options"]
    keys_original = ["A", "B", "C", "D"]

    # 셔플된 순서
    rng = random.Random(shuffle_seed)
    shuffled_order = list(keys_original)
    rng.shuffle(shuffled_order)

    # 셔플된 순서로 이미지 path 모음
    image_paths = [options[k]["image_path"] for k in shuffled_order]
    if any(p is None for p in image_paths):
        return {"error": "missing images"}

    # 모델에게 보여줄 'displayed letter' → 원본 key 매핑
    display_to_original = {
        chr(ord("A") + i): shuffled_order[i] for i in range(4)
    }

    prompt = build_eval_prompt(instance, ["A", "B", "C", "D"])

    try:
        response = call_local_vlm(
            prompt=prompt, image_paths=image_paths,
            system=EVAL_SYSTEM,
            api_base=model_cfg["api_base"],
            model=model_cfg["model_name"],
            max_tokens=8, temperature=0.0,
        )
    except Exception as e:
        return {"error": str(e)}

    # 응답 파싱
    predicted_letter = None
    for c in response.upper():
        if c in "ABCD":
            predicted_letter = c
            break

    if not predicted_letter:
        return {"error": "no valid letter", "response": response[:100]}

    predicted_original = display_to_original[predicted_letter]
    correct_original = instance["correct_option"]

    return {
        "instance_id": instance["instance_id"],
        "shuffled_order": shuffled_order,
        "predicted_displayed": predicted_letter,
        "predicted_original": predicted_original,
        "correct_original": correct_original,
        "correct": predicted_original == correct_original,
        "predicted_label": options[predicted_original]["label"],
        "raw_response": response[:200],
    }


# ─────────────────────────────────────────────
# 메트릭 계산
# ─────────────────────────────────────────────

def compute_metrics(results: list[dict], instances: dict) -> dict:
    """전체 결과로부터 다양한 메트릭 계산."""
    valid = [r for r in results if "error" not in r]
    n_valid = len(valid)
    if n_valid == 0:
        return {"error": "no valid results"}

    n_correct = sum(1 for r in valid if r["correct"])
    overall_acc = n_correct / n_valid

    # 도메인별
    domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    qtype_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    # 혼동 행렬: 정답 label vs 예측 label
    confusion = Counter()  # (correct_label, predicted_label) -> count

    # 오답 시 어떤 선지를 골랐는가
    wrong_picks = Counter()  # predicted_label when wrong -> count

    for r in valid:
        inst = instances[r["instance_id"]]
        domain = inst["domain"]
        qtype = inst["query"]["type"]

        domain_stats[domain]["total"] += 1
        qtype_stats[qtype]["total"] += 1
        if r["correct"]:
            domain_stats[domain]["correct"] += 1
            qtype_stats[qtype]["correct"] += 1

        correct_label = inst["options"][r["correct_original"]]["label"]
        predicted_label = r["predicted_label"]

        confusion[(correct_label, predicted_label)] += 1
        if not r["correct"]:
            wrong_picks[predicted_label] += 1

    # 진단 시그널: 오답 패턴 분석
    n_wrong = n_valid - n_correct
    diagnostic = {}
    if n_wrong > 0:
        for label in ["tpo_only", "preference_only", "neither"]:
            count = wrong_picks.get(label, 0)
            diagnostic[f"wrong_picked_{label}_ratio"] = count / n_wrong

    domain_acc = {
        d: s["correct"] / s["total"] for d, s in domain_stats.items()
    }
    qtype_acc = {
        q: s["correct"] / s["total"] for q, s in qtype_stats.items()
    }

    return {
        "n_total": len(results),
        "n_valid": n_valid,
        "n_errors": len(results) - n_valid,
        "overall_accuracy": overall_acc,
        "domain_accuracy": domain_acc,
        "query_type_accuracy": qtype_acc,
        "confusion_matrix": {f"{k[0]}→{k[1]}": v for k, v in confusion.items()},
        "wrong_pick_diagnostic": diagnostic,
        "interpretation": _interpret_diagnostic(diagnostic),
    }


def _interpret_diagnostic(diag: dict) -> str:
    """오답 패턴 해석."""
    tpo_only_ratio = diag.get("wrong_picked_tpo_only_ratio", 0)
    pref_only_ratio = diag.get("wrong_picked_preference_only_ratio", 0)
    neither_ratio = diag.get("wrong_picked_neither_ratio", 0)

    notes = []
    if tpo_only_ratio > 0.45:
        notes.append("TPO-overweighting: model prefers situational fit over user taste (similar to PCogAlign failure mode)")
    if pref_only_ratio > 0.45:
        notes.append("Preference-overweighting / mechanical FAE: model ignores TPO and picks based on persona only (similar to Whose Boat? / SynthesizeMe failure mode)")
    if neither_ratio > 0.30:
        notes.append("Random/confused: model fails on both axes")
    if not notes:
        notes.append("No strong directional bias in errors")

    return "; ".join(notes)


# ─────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────

def run_pipeline(benchmark_path: Path, model_key: str,
                 output_dir: Path, limit: int = 0,
                 n_seeds: int = 2):
    """전체 평가 실행."""
    log_step(f"Evaluator — model={model_key}")

    if model_key not in FREE_MODELS:
        # custom override 허용
        cfg = {
            "model_name": model_key,
            "api_base": "http://localhost:8000/v1",
        }
    else:
        cfg = FREE_MODELS[model_key]

    print(f"  Model: {cfg['model_name']}")
    print(f"  API: {cfg['api_base']}")

    instances_list = load_jsonl(benchmark_path)
    instances = {inst["instance_id"]: inst for inst in instances_list}

    if limit > 0:
        instances_list = instances_list[:limit]
        instances = {inst["instance_id"]: inst for inst in instances_list}

    print(f"  Loaded {len(instances_list)} instances")

    output_dir = output_dir / cfg["model_name"].replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in range(n_seeds):
        log_step(f"  Seed {seed} (position shuffle)")
        seed_results = []

        for i, inst in enumerate(instances_list):
            print(f"  [{i+1}/{len(instances_list)}] {inst['instance_id']}", end="\r")
            try:
                result = evaluate_instance(inst, cfg, shuffle_seed=seed * 1000 + i)
                result["seed"] = seed
                seed_results.append(result)
            except Exception as e:
                seed_results.append({"instance_id": inst["instance_id"],
                                     "error": str(e), "seed": seed})

            # 매 50개마다 저장
            if (i + 1) % 50 == 0:
                save_jsonl(seed_results, output_dir / f"results_seed{seed}.jsonl")

        save_jsonl(seed_results, output_dir / f"results_seed{seed}.jsonl")
        all_results.extend(seed_results)

        # seed별 메트릭
        metrics = compute_metrics(seed_results, instances)
        save_json(metrics, output_dir / f"metrics_seed{seed}.json")
        print(f"\n  Seed {seed} acc: {metrics['overall_accuracy']:.3f}")
        print(f"  Diagnostic: {metrics['interpretation']}")

    # 전체 메트릭
    overall_metrics = compute_metrics(all_results, instances)
    save_json(overall_metrics, output_dir / "metrics_overall.json")

    # Position bias 분석: 2 seed 간 일관성
    if n_seeds >= 2:
        per_instance_seeds = defaultdict(list)
        for r in all_results:
            if "error" not in r:
                per_instance_seeds[r["instance_id"]].append(r["correct"])

        consistent = sum(1 for v in per_instance_seeds.values()
                         if len(v) >= 2 and len(set(v)) == 1)
        total_with_2 = sum(1 for v in per_instance_seeds.values() if len(v) >= 2)
        consistency = consistent / max(total_with_2, 1)
        overall_metrics["position_consistency"] = consistency
        print(f"\n  Position consistency (2 seeds): {consistency:.3f}")
        save_json(overall_metrics, output_dir / "metrics_overall.json")

    print(f"\n  ✓ Saved results to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_path", type=Path,
                        default=FINAL_DIR / "pod_bench.jsonl")
    parser.add_argument("--model", type=str, default="vlm_evaluator",
                        help="Key in FREE_MODELS or HF model name")
    parser.add_argument("--output_dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=2)
    args = parser.parse_args()

    run_pipeline(args.benchmark_path, args.model,
                 args.output_dir, args.limit, args.n_seeds)


if __name__ == "__main__":
    main()
