"""
Profile Generator
==================
50명의 user profile을 생성합니다.

각 profile은 두 부분으로 구성:
  1. narrative_profile: 모델에 입력되는 자연어 서술 (명확하게 작성)
  2. structured_attributes: 내부 ground-truth, 라벨링에 사용 (모델에 노출 X)

사용:
  python -m src.profile_generator --n_users 50

산출:
  data/profiles/profiles.jsonl
"""

import argparse
import random
from pathlib import Path

from .utils import (
    call_gpt5_mini, parse_json_response,
    save_jsonl, load_jsonl, log_step,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    PROFILES_DIR, DOMAINS, ATTRIBUTES_BY_DOMAIN,
    PHASE1_CONFIG,
)


# ─────────────────────────────────────────────
# Structured attribute 샘플러
# ─────────────────────────────────────────────

def sample_structured_attributes(domain: str, seed: int) -> dict:
    """도메인 attribute taxonomy에서 일관성 있는 선호/회피 조합을 샘플링.

    핵심 원칙:
      - 각 attribute 카테고리에서 1-2개를 'prefer', 1-2개를 'avoid'로 표시
      - 같은 카테고리 내에서 모순되지 않도록 (예: prefer 'minimalist' & avoid 'heavy_decoration')
      - 도메인별 자연스러운 상관관계 반영 (예: minimalist는 plain detail과 함께)
    """
    rng = random.Random(seed)
    taxonomy = ATTRIBUTES_BY_DOMAIN[domain]

    # 스타일 정체성 먼저 결정 (다른 attribute의 종속 변수)
    style_options = taxonomy["style_identity"]
    primary_style = rng.choice(style_options)

    structured = {"primary_style": primary_style}

    # 스타일에 따른 일관성 있는 선호 생성
    if primary_style == "minimalist":
        biases = {
            "color_palette": {"prefer": ["muted_neutral", "deep_dark", "monochrome"],
                             "avoid": ["high_saturation"]},
            "silhouette": {"prefer": ["fitted", "structured"],
                          "avoid": ["oversized"]},
            "detail_load": {"prefer": ["plain"],
                           "avoid": ["heavy_decoration"]},
            "pattern": {"prefer": ["solid"],
                       "avoid": ["large_pattern", "graphic_print"]},
        } if domain == "fashion" else {
            "color_palette": {"prefer": ["cool_minimal", "scandinavian_light"],
                             "avoid": ["bold_colored"]},
            "form": {"prefer": ["clean_geometric"],
                    "avoid": ["ornate_detailed"]},
        }
    elif primary_style == "streetwear":
        biases = {
            "color_palette": {"prefer": ["high_saturation", "monochrome"],
                             "avoid": ["pastel_soft"]},
            "silhouette": {"prefer": ["relaxed", "oversized"],
                          "avoid": ["fitted"]},
            "detail_load": {"prefer": ["heavy_decoration", "subtle_detail"],
                           "avoid": ["plain"]},
            "pattern": {"prefer": ["graphic_print", "large_pattern"],
                       "avoid": []},
        }
    elif primary_style == "classic":
        biases = {
            "color_palette": {"prefer": ["muted_neutral", "deep_dark"],
                             "avoid": ["high_saturation", "pastel_soft"]},
            "silhouette": {"prefer": ["fitted", "structured"],
                          "avoid": ["oversized"]},
            "detail_load": {"prefer": ["subtle_detail"],
                           "avoid": ["heavy_decoration"]},
            "material": {"prefer": ["natural_fiber", "leather_suede"],
                        "avoid": ["synthetic"]},
        }
    elif primary_style == "bohemian":
        biases = {
            "color_palette": {"prefer": ["pastel_soft", "muted_neutral"],
                             "avoid": ["monochrome"]} if domain == "fashion"
                            else {"prefer": ["earthy_natural", "warm_wood"],
                                  "avoid": ["cool_minimal"]},
            "pattern": {"prefer": ["small_pattern", "large_pattern"]} if domain == "fashion" else {},
            "material": {"prefer": ["natural_fiber"] if domain == "fashion" else ["rattan_woven", "natural_wood"]},
        }
    elif primary_style == "scandinavian" and domain == "interior":
        biases = {
            "color_palette": {"prefer": ["scandinavian_light", "cool_minimal"],
                             "avoid": ["bold_colored"]},
            "material": {"prefer": ["natural_wood", "fabric_upholstered"],
                        "avoid": ["metal_industrial"]},
            "form": {"prefer": ["clean_geometric", "curved_organic"],
                    "avoid": ["ornate_detailed"]},
        }
    elif primary_style == "industrial" and domain == "interior":
        biases = {
            "color_palette": {"prefer": ["cool_minimal", "warm_wood"],
                             "avoid": ["pastel_soft", "bold_colored"]},
            "material": {"prefer": ["metal_industrial", "leather"],
                        "avoid": ["rattan_woven"]},
            "form": {"prefer": ["clean_geometric"],
                    "avoid": ["ornate_detailed"]},
        }
    else:
        # 기본: 랜덤하지만 카테고리별 최소 1개 prefer, 1개 avoid
        biases = {}
        for cat, options in taxonomy.items():
            if cat == "style_identity":
                continue
            shuffled = list(options)
            rng.shuffle(shuffled)
            biases[cat] = {
                "prefer": shuffled[:1 + rng.randint(0, 1)],
                "avoid": shuffled[-1:],
            }

    # 모든 attribute 카테고리에 대해 prefer/avoid 결정 (biases에 없는 건 약하게)
    for cat, options in taxonomy.items():
        if cat == "style_identity":
            continue
        if cat in biases:
            structured[cat] = {
                "prefer": biases[cat].get("prefer", []),
                "avoid": biases[cat].get("avoid", []),
            }
        else:
            shuffled = list(options)
            rng.shuffle(shuffled)
            structured[cat] = {
                "prefer": shuffled[:1],
                "avoid": [],
            }

    return structured


# ─────────────────────────────────────────────
# Narrative 생성 프롬프트
# ─────────────────────────────────────────────

NARRATIVE_SYSTEM = """\
You are a user profile writer for a personalization benchmark.
You write clear, specific user profiles in Korean that express the user's taste.

CRITICAL RULES:
- Write in Korean.
- Be SPECIFIC about preferences (colors, silhouettes, materials, style).
- The profile must allow a reader to predict the user's choice when shown items.
- Do NOT include any TPO/situation-specific information (no "for office", no "on rainy days").
  The profile describes ENDURING taste only.
- Length: 3-5 sentences, ~80-150 Korean characters.
- Output ONLY a JSON object."""


def build_narrative_prompt(domain: str, structured: dict, demographic_seed: int) -> str:
    """structured attributes로부터 narrative profile 생성 프롬프트."""
    rng = random.Random(demographic_seed)

    # 간단한 demographic context (취향과 독립적)
    age_band = rng.choice(["20대 후반", "30대 초반", "30대 후반", "40대 초반"])
    occupation_type = rng.choice([
        "직장인", "프리랜서", "대학원생", "디자이너", "엔지니어",
        "기획자", "교사", "연구원",
    ])

    # structured를 자연어 힌트로 변환
    hints = []
    primary = structured.get("primary_style", "")
    if primary:
        hints.append(f"메인 스타일: {primary}")

    for cat, vals in structured.items():
        if cat == "primary_style":
            continue
        if isinstance(vals, dict):
            prefer = vals.get("prefer", [])
            avoid = vals.get("avoid", [])
            if prefer:
                hints.append(f"{cat} 선호: {', '.join(prefer)}")
            if avoid:
                hints.append(f"{cat} 회피: {', '.join(avoid)}")

    hints_text = "\n  ".join(hints)

    domain_kr = "패션 (의류/액세서리)" if domain == "fashion" else "인테리어 (가구/홈데코)"

    return f"""다음 사용자의 narrative profile을 작성해주세요.

도메인: {domain_kr}
인구 정보: {age_band} {occupation_type}

내부 ground-truth 선호 정보:
  {hints_text}

위 정보를 자연스러운 한국어 서술로 풀어주세요.
사용자의 본질적 취향만 기술하고, 특정 상황(TPO)은 포함하지 마세요.

출력 형식 (JSON):
{{
  "narrative_profile": "..."
}}"""


# ─────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────

def generate_profile(user_idx: int, domain: str, seed: int) -> dict:
    """단일 user profile 생성."""
    structured = sample_structured_attributes(domain, seed)

    prompt = build_narrative_prompt(domain, structured, seed + 1000)
    response = call_gpt5_mini(
        prompt=prompt, system=NARRATIVE_SYSTEM,
        max_tokens=512, temperature=0.8,
    )
    parsed = parse_json_response(response)

    if parsed is None or "narrative_profile" not in parsed:
        # fallback: 구조화된 정보를 단순 변환
        narrative = _structured_to_text_fallback(structured, domain)
    else:
        narrative = parsed["narrative_profile"]

    return {
        "user_id": f"U{user_idx:03d}",
        "domain": domain,
        "narrative_profile": narrative,
        "structured_attributes": structured,
        "seed": seed,
    }


def _structured_to_text_fallback(structured: dict, domain: str) -> str:
    """LLM 실패 시 구조화 정보를 단순 한국어로 변환."""
    parts = []
    primary = structured.get("primary_style", "")
    if primary:
        parts.append(f"{primary} 스타일을 선호한다.")

    all_prefers = []
    all_avoids = []
    for cat, vals in structured.items():
        if isinstance(vals, dict):
            all_prefers.extend(vals.get("prefer", []))
            all_avoids.extend(vals.get("avoid", []))

    if all_prefers:
        parts.append("특히 " + ", ".join(all_prefers[:4]) + " 같은 속성을 좋아한다.")
    if all_avoids:
        parts.append(", ".join(all_avoids[:3]) + " 같은 요소는 피한다.")

    return " ".join(parts)


def run_pipeline(n_users: int, output_path: Path, force: bool = False):
    """전체 profile 생성 파이프라인 실행."""
    log_step(f"Profile Generator — {n_users} users")

    if output_path.exists() and not force:
        existing = load_jsonl(output_path)
        if len(existing) >= n_users:
            print(f"  Already have {len(existing)} profiles. Skipping. Use --force to regenerate.")
            return existing
        print(f"  Resuming from {len(existing)} existing profiles")
        profiles = existing
        start_idx = len(existing)
    else:
        profiles = []
        start_idx = 0

    # 도메인별 할당
    n_fashion = int(n_users * DOMAINS["fashion"]["weight"])
    n_interior = n_users - n_fashion
    domain_list = (["fashion"] * n_fashion) + (["interior"] * n_interior)
    random.Random(42).shuffle(domain_list)

    for i in range(start_idx, n_users):
        domain = domain_list[i]
        seed = 1000 + i * 17

        print(f"\n  [{i+1}/{n_users}] {domain} | seed={seed}")
        try:
            profile = generate_profile(i + 1, domain, seed)
            profiles.append(profile)
            print(f"    {profile['user_id']}: {profile['narrative_profile'][:80]}...")

            # 매 5개마다 중간 저장
            if (i + 1) % 5 == 0:
                save_jsonl(profiles, output_path)

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    save_jsonl(profiles, output_path)
    print(f"\n  ✓ Saved {len(profiles)} profiles to {output_path}")
    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int,
                        default=PHASE1_CONFIG["n_users_total"])
    parser.add_argument("--output", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_pipeline(args.n_users, args.output, args.force)


if __name__ == "__main__":
    main()
