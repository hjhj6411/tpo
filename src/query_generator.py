"""
Query Generator
================
각 user profile에 대해 다양한 TPO를 가진 query를 생성합니다.

생성 비율 (configs.config.QUERY_TYPE_RATIO):
  - explicit_tpo: 30% — "비 오는 날 외출용으로 무엇을 가져갈까?"
  - implicit_tpo: 25% — "야외 결혼식에 입을 옷을 추천해줘"
  - visual_tpo:   30% — 모호 텍스트 + 별도 TPO 컨텍스트 이미지
  - neutral:      15% — TPO 없음, 순수 선호 질문

각 query는 사용자의 structured_attributes와 *독립적*으로 생성됩니다
(query는 상황만 정의; 선호는 profile에서 가져옴).

사용:
  python -m src.query_generator --n_per_user 20

산출:
  data/queries/queries.jsonl
"""

import argparse
import random
from pathlib import Path

from .utils import (
    call_gpt5_mini, parse_json_response, parse_json_list_response,
    save_jsonl, load_jsonl, log_step,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    QUERIES_DIR, PROFILES_DIR, TPO_BY_DOMAIN,
    QUERY_TYPE_RATIO, PHASE1_CONFIG,
)


# ─────────────────────────────────────────────
# TPO 시나리오 샘플러
# ─────────────────────────────────────────────

def sample_tpo_scenario(domain: str, seed: int) -> dict:
    """TPO 축에서 한 시나리오 샘플링."""
    rng = random.Random(seed)
    tpo = TPO_BY_DOMAIN[domain]

    scenario = {}
    for axis, sub_axes in tpo.items():
        scenario[axis] = {}
        # 각 sub_axis에서 1개씩 샘플 (부분만 채워도 됨)
        sub_keys = list(sub_axes.keys())
        n_to_sample = rng.randint(1, len(sub_keys))
        sampled_keys = rng.sample(sub_keys, n_to_sample)
        for sk in sampled_keys:
            scenario[axis][sk] = rng.choice(sub_axes[sk])

    return scenario


def tpo_to_text(scenario: dict, domain: str) -> str:
    """TPO scenario를 자연어 힌트로 변환."""
    parts = []
    for axis, sub_dict in scenario.items():
        for sub_key, value in sub_dict.items():
            parts.append(f"{sub_key}={value}")
    return ", ".join(parts)


# ─────────────────────────────────────────────
# Query 생성 프롬프트
# ─────────────────────────────────────────────

QUERY_GEN_SYSTEM = """\
You are a query writer for a personalization benchmark.
You write natural Korean queries that users would ask a shopping/recommendation assistant.

Output ONLY a JSON list of strings."""


def build_explicit_tpo_prompt(domain: str, scenario: dict, n_queries: int) -> str:
    """명시적 TPO query 생성: TPO 정보가 query 텍스트에 직접 등장."""
    scenario_hint = tpo_to_text(scenario, domain)
    domain_kr = "패션" if domain == "fashion" else "인테리어"

    return f"""다음 TPO 시나리오에 대한 자연스러운 한국어 query를 {n_queries}개 생성하세요.
TPO 정보가 query 텍스트에 명시적으로 포함되어야 합니다.

도메인: {domain_kr}
TPO 시나리오 힌트: {scenario_hint}

요구사항:
- 각 query는 1-2문장, 자연스러운 한국어
- TPO 요소(시간/장소/상황)가 query에 명시적으로 등장
- 사용자가 추천을 요청하는 형태
- 너무 구체적인 색상/스타일은 언급하지 말 것 (그건 profile에서 옴)

좋은 예시:
  "비 오는 가을 야외 행사에 입고 갈 아우터 추천해줘"
  "여름 휴가 때 해변 산책용 신발 골라줘"

나쁜 예시:
  "푸른색 우비 추천해줘"  ← 색상 명시 금지
  "옷 추천해줘"           ← TPO 없음

출력 형식 (JSON list):
["query1", "query2", ...]"""


def build_implicit_tpo_prompt(domain: str, scenario: dict, n_queries: int) -> str:
    """함축적 TPO query: 상황 키워드만으로 TPO 추론 필요."""
    scenario_hint = tpo_to_text(scenario, domain)
    domain_kr = "패션" if domain == "fashion" else "인테리어"

    return f"""다음 TPO 시나리오에 대한 자연스러운 한국어 query를 {n_queries}개 생성하세요.
TPO 정보가 query 텍스트에 *함축적으로* 포함되어야 합니다.
즉, 시간/날씨/격식 등을 직접 말하지 않고, *상황의 이름*만 언급합니다.

도메인: {domain_kr}
TPO 시나리오 힌트: {scenario_hint}

좋은 예시:
  "친구 결혼식 입을 옷 추천해줘"    ← '결혼식'에서 격식/실내/낮 등을 추론
  "주말 캠핑 갈 때 옷 어떤 게 좋을까?"  ← '캠핑'에서 야외/활동성 등 추론

나쁜 예시:
  "비 오는 날에 갈 결혼식 옷"  ← 너무 명시적
  "옷 추천해줘"                ← 상황 없음

출력 형식 (JSON list):
["query1", "query2", ...]"""


def build_visual_tpo_prompt(domain: str, scenario: dict, n_queries: int) -> str:
    """시각적 TPO query: 텍스트는 모호, TPO는 별도 이미지로 제공.

    Phase 1에서는 query만 생성. TPO 컨텍스트 이미지는 별도 단계에서.
    """
    domain_kr = "패션" if domain == "fashion" else "인테리어"

    return f"""다음 도메인에 대한 *모호한* 한국어 query를 {n_queries}개 생성하세요.
이 query는 별도의 TPO 컨텍스트 이미지와 함께 모델에게 제시됩니다.
따라서 query 텍스트만으로는 TPO를 알 수 없어야 합니다.

도메인: {domain_kr}

좋은 예시:
  "여기 갈 때 입을 옷 추천해줘"        ← '여기'가 이미지로만 명시됨
  "이 분위기에 맞는 가구 찾아줘"
  "이런 곳에서 신을 신발 골라줘"

나쁜 예시:
  "결혼식 옷 추천해줘"      ← 상황을 텍스트로 말함
  "비 오는 날 옷"          ← TPO 노출

출력 형식 (JSON list):
["query1", "query2", ...]"""


def build_neutral_prompt(domain: str, n_queries: int) -> str:
    """TPO-중립 query: 순수 선호만 묻는 대조군."""
    domain_kr = "패션" if domain == "fashion" else "인테리어"

    return f"""다음 도메인에 대한 TPO 정보가 없는 순수 선호 query를 {n_queries}개 생성하세요.

도메인: {domain_kr}

좋은 예시:
  "내 취향에 맞는 옷 추천해줘"
  "내가 좋아할 만한 의자 골라줘"
  "이 중에 어떤 게 내 스타일에 맞을까?"

나쁜 예시:
  "사무실용 옷"  ← TPO 있음
  "비 올 때 입을 옷"  ← TPO 있음

출력 형식 (JSON list):
["query1", "query2", ...]"""


# ─────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────

def generate_queries_for_user(profile: dict, n_per_user: int) -> list[dict]:
    """단일 user에 대해 n개의 query 생성."""
    domain = profile["domain"]
    user_id = profile["user_id"]
    base_seed = abs(hash(user_id)) % 100000

    # 유형별 개수 계산
    type_counts = {}
    remaining = n_per_user
    for qtype, ratio in QUERY_TYPE_RATIO.items():
        count = int(n_per_user * ratio)
        type_counts[qtype] = count
        remaining -= count
    # 나머지는 explicit에
    type_counts["explicit_tpo"] += remaining

    all_queries = []

    for qtype, count in type_counts.items():
        if count == 0:
            continue

        # batch로 생성 (한 번에 3-4개씩)
        batch_size = min(count, 4)
        generated = 0

        while generated < count:
            this_batch = min(batch_size, count - generated)

            if qtype == "explicit_tpo":
                scenario = sample_tpo_scenario(domain, base_seed + generated)
                prompt = build_explicit_tpo_prompt(domain, scenario, this_batch)
                tpo_field = scenario
            elif qtype == "implicit_tpo":
                scenario = sample_tpo_scenario(domain, base_seed + generated + 100)
                prompt = build_implicit_tpo_prompt(domain, scenario, this_batch)
                tpo_field = scenario
            elif qtype == "visual_tpo":
                scenario = sample_tpo_scenario(domain, base_seed + generated + 200)
                prompt = build_visual_tpo_prompt(domain, scenario, this_batch)
                tpo_field = scenario  # 이미지로 표현될 TPO
            else:  # neutral
                prompt = build_neutral_prompt(domain, this_batch)
                tpo_field = None

            try:
                response = call_gpt5_mini(
                    prompt=prompt, system=QUERY_GEN_SYSTEM,
                    max_tokens=512, temperature=0.8,
                )
                queries = parse_json_list_response(response)
                if not queries:
                    print(f"      Parse failed for {qtype}, skipping batch")
                    generated += this_batch
                    continue

                for q in queries[:this_batch]:
                    all_queries.append({
                        "query_id": f"{user_id}_Q{len(all_queries)+1:03d}",
                        "user_id": user_id,
                        "domain": domain,
                        "query_type": qtype,
                        "query_text": q,
                        "tpo_scenario": tpo_field,
                    })
                generated += len(queries[:this_batch])

            except Exception as e:
                print(f"      Error generating {qtype} batch: {e}")
                generated += this_batch  # 진행은 계속

    return all_queries[:n_per_user]


def run_pipeline(profile_path: Path, output_path: Path, n_per_user: int,
                 force: bool = False):
    """전체 query 생성 파이프라인 실행."""
    log_step(f"Query Generator — {n_per_user} queries/user")

    profiles = load_jsonl(profile_path)
    print(f"  Loaded {len(profiles)} profiles")

    if output_path.exists() and not force:
        existing = load_jsonl(output_path)
        existing_users = {q["user_id"] for q in existing}
        print(f"  Resuming: {len(existing_users)} users already have queries")
        all_queries = existing
        profiles_to_do = [p for p in profiles if p["user_id"] not in existing_users]
    else:
        all_queries = []
        profiles_to_do = profiles

    for i, profile in enumerate(profiles_to_do):
        print(f"\n  [{i+1}/{len(profiles_to_do)}] {profile['user_id']} ({profile['domain']})")
        try:
            queries = generate_queries_for_user(profile, n_per_user)
            all_queries.extend(queries)
            print(f"    Generated {len(queries)} queries")

            # 매 5명마다 저장
            if (i + 1) % 5 == 0:
                save_jsonl(all_queries, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    save_jsonl(all_queries, output_path)
    print(f"\n  ✓ Saved {len(all_queries)} queries to {output_path}")
    return all_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--output", type=Path,
                        default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--n_per_user", type=int,
                        default=PHASE1_CONFIG["n_queries_per_user"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_pipeline(args.profile_path, args.output, args.n_per_user, args.force)


if __name__ == "__main__":
    main()
