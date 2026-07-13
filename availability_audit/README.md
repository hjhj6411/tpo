# Availability Audit — retrievability-constrained option construction

`striped trench coat` 같은 희소 조합을 사람이 "없을 법하다"고 blacklist하지 않는다.
대신 **고정된 POD-Bench corpus와 현재 image construction pipeline에서, 품질 기준을
만족하는 서로 다른 이미지를 충분히 확보할 수 있는가**를 실측하고, 그 실측치로
planner를 제약한다. 논문 표기: *retrievability-constrained option construction* /
*corpus-aware 2×2 assembly*.

## 정책 (2단계 구조)

1. **Option 생성 전** — corpus availability를 실측하여 planner가 4-option 조합
   전체가 지원되는 tuple만 선택하게 한다 (주 방식).
2. **Retrieval 후** — 실패하면 다른 feasible tuple로 backfill하고, 끝까지
   구성되지 않는 instance만 reason code와 함께 제거한다 (최종 안전장치).

## "없는 옷"의 세 가지 경우 — 반드시 분리한다

| 경우 | 원인 | 대응 |
|---|---|---|
| **A. 조합 자체가 희소** (leopard puffer jacket) | corpus에 없음 | red 등급 → planner에서 제외 |
| **B. 존재하지만 complex query가 못 찾음** (striped trench coat) | FashionSigLIP의 복합 쿼리 한계 | garment-first bank로 회수 → 유지 |
| **C. retrieval 성공, SAM3 mask 실패** | prompt/이미지 표현 불일치 | mask alias 개선 → 유지 |

현재 pipeline은 mask 실패를 0점 처리하므로 B/C가 A로 오분류된다. 따라서
"complex query에서 valid candidate 0개"만으로 조합을 제거하면 안 된다.

## 핵심 원칙: availability 측정에 complex query를 쓰지 않는다

Garment-only(+alias union)로 넓게 회수한 뒤, 검증기를 **candidate 쪽에** 돌려서
"corpus 안의 trench coat 중 striped가 몇 개인가"를 잰다. complex query retrieval
성능과 무관한 측정치가 된다.

```
Step 1  garment-only retrieval        canonical + synonym alias union, top 300–1000
Step 2  VLM garment 검증               closed-vocabulary 분류 (기존 collector와 동일 vocab)
Step 3  pattern/color 분류             /patch-coverage (6-way argmax), /patch-color-coverage (13-way)
        → availability_catalog.jsonl  (image_id, garment, pattern, color, scores, mask_status)
```

숫자는 단순 retrieval 수가 아니라 **검증을 통과한 distinct product 수**다.

## 두 개의 support를 따로 저장한다

- **corpus_support** — garment VLM 통과 + full-image(또는 대체 방식) pattern 확인.
  "옷이 corpus에 존재하는가".
- **pipeline_yield** — 현재 SAM3 pipeline에서 mask까지 성공해 실제 option으로
  쓸 수 있는 수. "지금 파이프라인이 뽑아낼 수 있는가".

| Pair | corpus | maskable | 판단 |
|---|---:|---:|---|
| striped trench coat | 12 | 10 | 사용 가능 |
| striped puffer jacket | 6 | 1 | 조합은 존재 — masking 개선 대상 |
| leopard sweatshirt | 2 | 2 | 실제로 희소 |
| polka-dot puffer | 0 | 0 | 제거 |

이 구분 없이 모두 제거하면 benchmark가 segmentation 모델의 편향에 맞춰 축소된다.

## 등급 (pilot 후 임계값 확정)

- **Green** — 자유 사용. `verified unique images >= 8–10`, mask success >= 60%.
- **Yellow** — 제한 사용(최대 사용 횟수 cap, top-k 확대/alias 확장 시도). `3–7`.
- **Red** — multimodal main set 제외. `0–2` 또는 대부분 seg/retrieval 실패.

고정 임계값 대신 사용 계획 기반 최소치:

```
N_min(combo) = ceil(planned_uses / max_reuse_per_image) + safety_buffer
```

예: 8회 필요, 이미지당 재사용 2회 허용 → 최소 4 + buffer 2–3.

## 4-option 전체가 지원되어야 한다 (2×2 rectangle 조건)

pattern-active instance에서 (liked=striped, disliked=leopard,
compat=trench_coat, incompat=puffer_jacket)이면 다음 네 edge가 전부 필요하다:

```
striped trench coat / leopard trench coat / striped puffer / leopard puffer
```

planner 통합 지점 (`construction/option_planner.py`):

```python
from availability_audit.feasibility import option_set_is_retrievable, rarity_cost
```

- 현재 planner는 (a,b) value와 garment pair를 **따로** 고른다 → availability
  도입 시 `(a, b, compat_g, incompat_g, fixed_attr)`를 하나의 tuple로 보는
  **joint assignment**로 바꾼다. feasible tuple만 후보로 만들고, 기존 목적함수
  (counterbalance + confusability + diversity)에 rarity 항을 더한다:

```
cost = balance + confusability + repetition + λ · rarity
set_support = min(s_A, s_B, s_C, s_D)
rarity_cost = 1 / (set_support + ε)
```

## 비활성 축 선택 규칙

**Color-active** (pattern은 fixed): ① neutral+TPO-safe면 `solid` 우선 ②
solid가 like/dislike면 availability 높은 neutral pattern ③ 충분한 support가
없으면 **unfixed** ④ rare pattern을 억지로 고정하지 않는다. (두 번째
preference/TPO signal을 만들지 않는 것이 명시보다 우선 — 기존
`query_generator._sample_preference_neutral_value` 설계와 충돌 없음.)

**Pattern-active** (color는 fixed): 네 garment×pattern 조합 모두에서 충분히
존재하는 색을 고른다:

```
c* = argmax_{c ∈ neutral colors} min( S(g_ok,p_like,c), S(g_ok,p_dis,c),
                                      S(g_bad,p_like,c), S(g_bad,p_dis,c) )
```

공통으로 충분한 색이 없으면 그 2×2 tuple 자체를 제외한다.
→ `feasibility.pick_fixed_color()`.

## Post-retrieval backfill

실패한 option 한 개만 삭제하면 2×2가 깨진다. 순서:

```
1차 tuple로 A/B/C/D retrieval → 일부 실패
→ 같은 query에서 다음 feasible tuple로 backfill (다른 garment pair 또는 다른 a/b)
→ 최대 R회 실패 시 instance 전체 제거 + reason code
```

Reason codes: `unsupported_combination`, `insufficient_unique_candidates`,
`garment_retrieval_failure`, `sam3_mask_failure`, `pattern_verification_failure`,
`color_verification_failure`, `set_level_gender_conflict`, `duplicate_image_conflict`.

## Filtering 후 재검증 (새 confound 방지)

희소 pair를 삭제하면 pattern↔garment 상관이 강해질 수 있다 (leopard는 dress에만,
puffer는 항상 solid → 모델이 profile 없이 garment만 보고 pattern을 추정).
최종 image-realizable subset에 대해 반드시 재확인:

- pattern별 등장 garment 수 / garment별 포함 pattern 수
- pattern별 TPO-compatible/incompatible 역할 균형
- active value의 A/B 빈도 균형 유지 여부
- preference-blind exploit accuracy 증가 여부
- → `scripts/validate_options.py`를 최종 subset에 재실행

## 구현 순서

| Phase | 내용 | 코드 |
|---|---|---|
| 1 | garment-first candidate bank (retrieve → VLM garment → pattern/color 분류) | `build_candidate_bank.py` |
| 2 | support matrix 2종 (`corpus_support.json`, `pipeline_yield.json`) + 등급 | `support_matrix.py` |
| 3 | planner feasibility hard constraint + rarity soft penalty | `feasibility.py` → `option_planner.py` 수정 |
| 4 | collection backfill (catalog 내 다음 candidate → 다음 feasible tuple) | collector 수정 |
| 5 | 최종 subset 재검증 (integrity/balance/reuse/correlation/blind-exploit) | `validate_options.py` 재실행 |

## Pilot (최소 실험)

전체 도입 전에 `audit_config.PILOT_RARE_PAIRS`(10) + `PILOT_COMMON_PAIRS`(10)에
대해 다음을 비교한다: ① complex query top-100 valid 수 ② garment-bank top-500
post-classification valid 수 ③ alias union 효과 ④ SAM3 mask 성공률 ⑤ 최종
verified distinct 수.

```bash
# 서버 필요: fsiglip/serve_fsiglip_knn.py (:1235), vLLM VLM (:8002)
python -m availability_audit.build_candidate_bank --stage retrieve --pairs pilot
python -m availability_audit.build_candidate_bank --stage verify-garment --pairs pilot
python -m availability_audit.build_candidate_bank --stage classify --pairs pilot
python -m availability_audit.support_matrix               # → 등급표 출력 + json 저장
```

이 결과로 complex query 문제 / corpus 문제 / SAM3 문제를 분리하고, green/yellow/red
임계값을 확정한 뒤 전체 20 garment × 6 pattern(=120 pair)으로 확장한다.
색상은 1560개 조합을 직접 감사하지 않고, 각 pair에서 실제로 관측된 color 분포를
catalog에서 집계한다.
