# Garment 텍스트 쿼리 재정립 설계서 (굵은 통합안)

> 상태: **제안 (미적용)** · 작성일 2026-07-06
> 목적: garment 카테고리의 "우산어 vs 특정어" 혼용 때문에 정답 아이템이
> garment 게이트에서 몰살당하는 문제를 해결한다.

---

## 1. 문제 정의

`search_query`는 `"{color} {pattern} {garment}"` 한 줄로 생성된다
(`option_to_text` → `retrieval_query`). 여기서 쓰는 garment 명사가 파이프라인의
두 단계에서 의미가 달라진다.

| 단계 | garment 해석 | 예: `striped white jacket` |
|---|---|---|
| SAM3 + SigLIP 검색 | **상위어(우산)** — blazer·windbreaker·bomber 다 끌어옴 | 줄무늬 블레이저가 상위 랭크로 진입 |
| VLM garment 검증 | **좁은 특정 클래스**, `equiv={jacket}` | `pred=blazer` → **FAIL** |

실측(`q00004 ... striped_white_jacket`): 20개 중 **17개가 garment_fail**,
그중 다수가 `pat≈0.95 col≈1.0`인 완벽한 줄무늬 블레이저였다.

근본 원인: garment 어휘에 **시각적으로 구분 불가하거나(blazer↔suit jacket),
포함관계인(coat⊃trench coat, jacket⊃windbreaker)** 쌍들이 서로 배타 클래스로
취급되고 있다. 검증 어휘(`GARMENT_EQUIV_GROUPS`)가 이들을 병합하지 않기 때문.

---

## 2. 설계 원칙

1. **쿼리 taxonomy와 검증 equiv set은 한 쌍으로 정한다.** 쿼리 문자열만 고쳐도
   equiv set이 같은 입도로 안 맞으면 fail은 그대로 남는다.
2. **VLM이 사진만으로 신뢰성 있게 구분 못 하는 쌍은 하나의 canonical로 병합한다**
   (굵은 통합). 단, TPO 변별에 필요한 굵은 대비(blazer vs hoodie 등)는 유지한다.
3. **병합은 TPO 대비를 절대 깨지 않아야 한다** — 한 쿼리 안에서 compatible /
   incompatible 양쪽에 같은 canonical의 서로 다른 원소가 동시에 오면 안 된다.

---

## 3. 최종 garment taxonomy (canonical 13개)

| canonical | 흡수하는 원소(raw) | 비고 |
|---|---|---|
| **blazer** | blazer, suit jacket | 사진상 거의 구분 불가 → 병합 |
| **coat** | coat, trench coat | trench는 coat의 하위 유형 → 병합 |
| **jacket** | jacket, windbreaker | 얇은 캐주얼 셸 우산어 (필요시 bomber/denim jacket 추가) |
| **fleece** | fleece | puffer와 겹치지 않는 soft outerwear 용어로 유지 |
| **shirt** | shirt, blouse | *이미 병합됨*(현행 `GARMENT_EQUIV_GROUPS`) |
| t_shirt | t shirt, tee | 현행 유지 |
| tank_top | tank top, camisole, crop top | 현행 유지 |
| sweater | sweater, pullover, cardigan, knitwear | 현행 유지 |
| hoodie | hoodie, sweatshirt | 현행 유지 |
| dress | dress, gown | 현행 유지 |
| skirt | skirt, mini/long skirt | 현행 유지 |
| jeans | jeans, denim | 현행 유지 |
| shorts | shorts | 유지 |

병합 후 카테고리별 플랜 사용량(참고):
`blazer 1210 · t_shirt 1082 · tank_top 936 · shorts 906 · shirt 820 · coat 760 ·
dress 630 · jacket 594 · hoodie 480 · fleece 470 · sweater 408 · skirt 244 · jeans 112`

### ⚠️ 중요: 이 통합안에서 `jacket`과 `blazer`는 **여전히 별개**다
- 따라서 예제 `q00004 (striped white jacket)`의 블레이저들은 **계속 FAIL**이며,
  이는 **의도된 정상 동작**이다 (극한 추위 시나리오에서 얇은 정장 블레이저는
  "jacket"의 유효 답이 아님).
- 굵은 통합이 실제로 살려내는 것은 **그룹 내부 동의어 실패**다:
  - target=blazer 인데 아이템이 suit jacket → (기존 FAIL) → **PASS**
  - target=coat 인데 아이템이 trench coat → **PASS**
  - target=jacket 인데 아이템이 windbreaker → **PASS**

---

## 4. 검증 equiv set 신규 정의

`fsiglip/collect_topk_sam3_fsiglip_patch_rank_vlm_garment.py` 의
`GARMENT_EQUIV_GROUPS`(현재 834–843줄)에 **아우터 3개 그룹만 추가**한다.
`GARMENT_VOCAB`(808–827줄)은 세부 라벨(suit jacket / trench coat / windbreaker)을
이미 포함하므로 **변경 불필요** — VLM이 세부어를 출력하면 equiv 그룹이 canonical로
접어준다.

```python
GARMENT_EQUIV_GROUPS = [
    {"tank top", "sleeveless top", "camisole top", "camisole", "crop top"},
    {"t shirt", "tee", "tee shirt"},
    {"shirt", "blouse"},
    {"jeans", "denim pants", "denim"},
    {"skirt", "mini skirt", "long skirt"},
    {"dress", "gown"},
    {"sweater", "pullover", "knit sweater", "cardigan", "knitwear"},
    {"hoodie", "sweatshirt"},
    # ── 신규(굵은 통합) ────────────────────────────────
    {"blazer", "suit jacket"},
    {"coat", "trench coat"},
    {"jacket", "windbreaker"},          # 필요시 "bomber jacket", "denim jacket" 추가
]
```

동시에 800–807줄 / 829–833줄 주석(“outerwear를 SEPARATE로 유지한다”)을 수정해야
한다 — 이제 아우터를 **coarse canonical 3개(blazer/coat/jacket) + fleece**로
접되, 이들 4개 간 굵은 대비는 유지한다는 문구로.

---

## 5. TPO 안전성 검증 (완료)

병합맵 `{suit_jacket→blazer, trench_coat→coat, windbreaker→jacket, blouse→shirt}`
하에서, **한 컨텍스트 안에서 같은 canonical의 서로 다른 원소가 compatible ↔
incompatible 양쪽에 동시 등장하는 충돌**을 두 레벨에서 스캔했다:

| 스캔 대상 | 결과 |
|---|---|
| `configs/scenarios.py`의 53개 시나리오 **garment 팔레트** (compat/incompat 원본) | **충돌 0건** |
| 생성된 `option_plans.jsonl` 전체 (실제 샘플된 쿼리) | **충돌 0건** |

→ 굵은 통합은 소스(시나리오 팔레트)에서도, 산출물(플랜)에서도 어떤 TPO 대비도
깨지 않는다. (예: `_FIELD_C`는 jacket·windbreaker를 **둘 다 compatible** 쪽에만
두므로 canon 후 하나로 접혀도 무해.)

---

## 5-1. 실제 파이프라인: garment 단어의 흐름 (construction/)

"텍스트 쿼리"는 `construction/` STAGE 2–3에서 만들어진다. garment 어휘의 출처와
경로는 다음과 같다:

```
configs/scenarios.py                     # 출처: _COLD_C/_HEAT_I 등 garment 팔레트
   └─ garment_category.{compatible, incompatible}
        │
construction/compatibility.py            # check_axis_compatibility() (84–87줄)
   └─ compatible_garments / incompatible_garments  (유저 garment 선호 뺀 neutral만)
        │
construction/query_generator.py          # 각 원소를 query 레코드에 실어보냄
        │
construction/option_planner.py
   ├─ _pick_garment()  (57줄)            # compat 1개 + incompat 1개 랜덤 선택
   └─ attrs_to_search_query()  (62줄)    # "{pattern≠solid} {color} {garment.replace('_',' ')}"
        │
   data/options/option_plans.jsonl       # 최종 search_query 문자열
```

즉 **search_query의 garment 단어 = 시나리오 팔레트 원소를 그대로 `_`→공백 치환한 것**
(`suit_jacket`→"suit jacket", `trench_coat`→"trench coat"). "명확한 쿼리"로 바꾸려면
이 흐름의 어딘가에서 canonical 정규화를 넣어야 한다(§6 안 B).

---

## 6. "텍스트 쿼리" 자체를 바꿀지 — 2가지 적용 범위

병합은 **검증 시점(verdict)** 에서 일어나므로, 쿼리 문자열/플랜을 반드시 바꿀
필요는 없다. 두 안 중 선택:

### 안 A — 검증만 변경 (권장, 최소·가역)
- `GARMENT_EQUIV_GROUPS`에 3개 그룹 추가 + 주석 수정. **끝.**
- `option_plans.jsonl`, `scenarios.py` 무변경 → **재생성 불필요.**
- 기존 원본 카테고리(suit_jacket/trench_coat/windbreaker)는 그대로 두되 verdict에서
  canonical로 접힘. 세부어 쿼리는 오히려 검색 정밀도에 유리(우산어보다 구체적).
- 단점: 플랜 안에 `suit_jacket`과 `blazer`가 별도 쿼리로 공존 → 사실상 동일 클래스가
  두 개의 쿼리로 표기되는 표기 중복이 남음.

### 안 B — construction 단계에서 garment_category를 canonical로 정규화 (후속, 재생성 필요)
쿼리 문자열과 저장된 `garment_category` 속성을 모두 canonical로 통일 → 가장 "명확한
쿼리". **권장 주입 지점은 `construction/compatibility.py`** (garment 단어가 처음
개별 원소로 확정되는 곳):

```python
# construction/utils.py 등에 canonical 맵 1곳
GARMENT_CANON = {"suit_jacket": "blazer", "trench_coat": "coat",
                 "windbreaker": "jacket", "blouse": "shirt"}
def canonical_garment(g): return GARMENT_CANON.get(g, g)

# construction/compatibility.py check_axis_compatibility() 84–87줄:
compatible_garments   = sorted({canonical_garment(g) for g in _neutral_values(
                            garment_constraint["compatible"], garment_likes, garment_dislikes)})
incompatible_garments = sorted({canonical_garment(g) for g in _neutral_values(
                            garment_constraint["incompatible"], garment_likes, garment_dislikes)})
```
(유저 garment 선호 `garment_likes/dislikes`도 빼기 전에 canon하면 완전 일관.)

- 이렇게 하면 `option_planner.attrs_to_search_query`는 손대지 않아도 canonical
  단어로 쿼리가 생성됨. `configs/scenarios.py` 팔레트는 **원본 유지 가능**(정규화가
  construction 레이어에서 일어나므로) — 팔레트를 직접 고칠 필요 없음.
- 영향: `option_plans.jsonl` **재생성** + 다운스트림 랭킹 **재실행**.
- 장점: 쿼리 = 검증 = 단일 canonical, 표기 완전 일치.
- 단점: windbreaker/trench 고유 검색 신호 손실, 전체 재생성 비용.

> 권장 순서: **먼저 안 A로 적용·검증** → 결과 보고 필요하면 안 B로 정규화.

---

## 7. 영향받는 파일 목록

| 파일 | 안 A | 안 B |
|---|---|---|
| `fsiglip/collect_topk_sam3_fsiglip_patch_rank_vlm_garment.py` (`GARMENT_EQUIV_GROUPS`, 주석) | **수정** | 수정 |
| `construction/compatibility.py` (compatible/incompatible_garments 정규화) | 무변경 | **수정(주입점)** |
| `construction/utils.py` (`GARMENT_CANON` 맵 추가) | 무변경 | **추가** |
| `construction/option_planner.py` (`attrs_to_search_query`) | 무변경 | 무변경(자동 반영) |
| `configs/scenarios.py` (garment 팔레트) | 무변경 | 무변경(원본 유지) |
| `data/options/option_plans.jsonl` | 무변경 | **재생성** (`python -m construction.option_planner --force`) |
| `data/retrieval/.../ranked_images`, `topk_patches` (랭킹 산출물) | 재실행 시 갱신 | **재실행** |

> 참고: STAGE 2(`query_generator.py`)부터 재생성해야 완전 일관 —
> `queries.jsonl`의 `compatible_garments`/`incompatible_garments`가 canonical이 되려면
> STAGE 2→3 순서로 재실행.

---

## 8. 재생성 필요 여부 요약

- **안 A**: 코드 1파일(`GARMENT_EQUIV_GROUPS`) 수정 → garment 검증만 바뀜. 플랜 재생성
  **불필요**, 랭킹만 재실행하면 신규 equiv가 반영됨.
- **안 B**: `construction/` 정규화 후 STAGE 2→3 재생성 + 랭킹 재실행 **필요**.

---

## 9. 미결 결정사항 (승인 요청)

1. **적용 범위**: 안 A(검증만) 로 먼저 갈지, 곧장 안 B(플랜 정규화)까지 갈지.
2. **`jacket` 우산 범위**: `{jacket, windbreaker}` 로 둘지, `bomber jacket`/
   `denim jacket` 도 vocab+equiv에 추가할지.
3. **`fleece` 처리**: puffer와 분리되는 독립 soft outerwear로 유지할지.
