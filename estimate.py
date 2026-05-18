# cost_estimate.py — 실행 전 토큰 추정
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")  # gpt-5-mini 전용 인코더 없으므로 근사

# gpt-5-mini 단가 (2025년 8월 출시 기준)
INPUT_PRICE  = 0.25 / 1_000_000   # $0.25 / 1M input tokens [cite:81]
OUTPUT_PRICE = 2.50 / 1_000_000   # $2.50 / 1M output tokens (reasoning 포함)

# Phase 1 파이프라인 호출 수
CALLS = {
    "profile_generation":   50,          # 50명
    "query_generation":     50 * 5,      # 50명 × 배치 5회
    "option_planning":      50 * 20,     # 1000 queries
    "label_verifier_judge": 1000,        # 1000 queries × GPT judge 1회
}

# 프롬프트별 평균 토큰 추정 (길이 샘플링)
AVG_TOKENS = {
    "profile_generation":   {"in": 400,  "out": 600},
    "query_generation":     {"in": 300,  "out": 200},
    "option_planning":      {"in": 600,  "out": 800},
    "label_verifier_judge": {"in": 500,  "out": 200},
}

total_cost = 0
for step, n_calls in CALLS.items():
    t = AVG_TOKENS[step]
    cost = n_calls * (t["in"] * INPUT_PRICE + t["out"] * OUTPUT_PRICE)
    print(f"{step:30s}: {n_calls:5d} calls → ${cost:.2f}")
    total_cost += cost

print(f"\nPhase 1 총 예상 비용: ${total_cost:.2f}")