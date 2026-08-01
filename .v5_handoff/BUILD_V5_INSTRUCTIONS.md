# 작업 지시: `wacv_scenario_v5` 생성

`hjhj6411/tpo` 저장소에서 POD-Bench의 다음 변형 `wacv_scenario_v5`를 만든다.
변경은 다섯 가지이며, 각각 **왜 하는지**와 **어떻게 바꾸는지**를 아래에 적었다.

작업 범위는 **Stage 1–3 (프로필 → 질의 → 옵션 플랜)** 과 트랙 용어 개명이다.
GPU 작업(이미지 재수집, SAM3 스크리닝, 재주석)과 이미지 폴더 실체화는 포함되지 않는다.

참조 구현으로 `v5_reference.patch`가 함께 제공된다. 그대로 적용해도 되고 아래
명세대로 직접 구현해도 되지만, **§8의 수용 기준은 반드시 만족해야 한다.**

---

## 0. 배경 — 지금 무엇이 잘못되어 있나

`wacv_scenario_v4`는 옵션 플랜 3,027개를 만들었다. 그런데 사람이 1,661개 셀을
전수 검수해 이미지 라이브러리(`annotation/attribute_library.json`)를 확정한 뒤
대조해 보니, **네 선지 A/B/C/D 모두에 이미지를 붙일 수 있는 문항은 1,012개(33.4%)**
뿐이었다. 원인은 두 가지다.

### 원인 1 — 만들고 나서 버리는 구조

옵션 생성기가 이미지 라이브러리를 전혀 모르는 채로 조합을 정하고, 나중에 이미지가
없는 문항을 버렸다. 문제는 2,015개를 잃은 것만이 아니다. **버리는 과정이 무작위가
아니라는 점**이 더 심각하다. 실제로 트랙 비율이 62:38 → 69:31로 틀어졌고 축 비율도
함께 무너졌다. 생성 단계에서 정교하게 맞춰 놓은 카운터밸런스·혼동쌍 회피·사용자별
다양성이 사후 삭제로 전부 훼손된 것이다.

### 원인 2 — 배경축 빈칸 803개

각 문항은 세 축(색·의류·무늬) 중 하나를 **선호축**(사용자가 좋아하는 값 vs 싫어하는
값), 하나를 **위반축**(상황에 맞는 값 vs 안 맞는 값)으로 쓰고, 남는 하나가
**배경축**이 된다. 배경축 값은 A/B/C/D 네 선지에 동일하게 들어간다.

v4에서는 이 배경축에 값이 없는 문항이 803개 있었다. 값이 없으면 이미지를 검색할
`color|garment|pattern` 키를 만들 수 없으므로 문항 전체가 무용지물이 된다. 그 결과
**6개 시나리오가 문항 0개로 전멸**했다: 장례식, 추모식, 법정, 졸업식, 그린스크린
촬영, TV 인터뷰. 빈칸이 생긴 근본 원인은 §3에서 설명한다.

### 목표 상태

| 항목 | v4 | v5 목표 |
|---|---:|---:|
| 네 선지 모두 이미지 존재 | 1,012 (33.4%) | **~2,571 (97%↑)** |
| 문항 0개 시나리오 | 6 | **0** |
| 12명 미만 시나리오 | 24 | **0** |

---

## 1. 절대 지켜야 할 제약

- **`data_wacv_scenario_v4/`를 수정·삭제하지 말 것.** 역사적 산출물로 보존한다.
  모든 출력은 `data_wacv_scenario_v5/`로 간다.
- **암묵적 기본값 금지.** "무늬가 비어 있으면 조용히 solid로 친다" 같은 폴백을 이미지
  조립이나 평가 단계에 넣지 말 것. 값 결정은 반드시 Stage 1–3에서 명시적으로
  이루어지고 검증기를 통과해야 한다 (`docs/PROCESS.md` §6).
- **선호축 A/B는 언제나 liked vs disliked.** 중립값이 A/B 대비에 들어가는 경로를
  만들지 말 것. 이 규칙이 깨지면 축마다 측정 기준이 달라진다.
- **결정론 유지.** 같은 시드로 두 번 돌리면 세 산출 파일의 SHA256이 일치해야 한다.
- 기존 검증기·리포트 스크립트의 시그니처를 바꾸지 말 것. 그대로 통과해야 한다.

---

## 2. 변경 A — 이미지 가용 셀만 후보로 (`construction/option_planner.py`)

### 왜

원인 1을 고친다. 핵심은 **"만들고 나서 버리기"를 "만들 때부터 피하기"로 바꾸는** 것.

가장 중요한 점: **이것은 사후 필터가 아니라 후보 제약이어야 한다.** 문항을 다 만든
뒤 걸러내면 v4와 똑같은 결과가 된다. 각 배정 함수의 후보 루프 안에서, 점수를 계산
**전에** 걸러야 한다. 그래야 카운터밸런스·혼동쌍 회피·사용자별 다양성 목적함수가
**실현 가능한 범위 안에서** 계속 최적화된다.

참고로 "아마존 코퍼스에 있는 옷만 쓴다"는 제약은 v4에도 이미 걸려 있었다. 다만 그
제약이 파이프라인 **끝**에서 작동해 균형을 파괴했을 뿐이다. 이 변경은 새 제약을
도입하는 것이 아니라, 이미 존재하던 제약을 균형 로직이 볼 수 있는 위치로 옮기는 것이다.

### 어떻게

`annotation/attribute_library.json`을 생성 입력으로 받아 `status == "available"`인
셀 키 집합을 만든다. 스키마는 `"color|garment_category|pattern" -> {status, images[]}`
이며 available 셀은 1,237개다.

```python
def _cells_ok(triples):
    """triples: 네 선지의 (color, garment, pattern).
    필터가 꺼져 있거나 네 셀이 모두 available이면 True.
    세 축이 다 정해지지 않은 후보는 그냥 통과시킨다(변경 B가 별도로 다룸)."""
```

적용 위치는 여섯 곳:

| 함수 | 검사 대상 |
|---|---|
| `assign_garment_pairs` | `(compat_garment, incompat_garment)` × A/B 선호값 |
| `assign_violation_values` | `(garment, compat_value, incompat_value)` × A/B 선호값 |
| `assign_garment_active_values` | `(compat_value, incompat_value)` × A/B 의류 |
| `_value_violation_feasible` | 위반축 회전 가능성 판정 |
| `_garment_active_violation_feasible` | 동일 |
| `plan_option_variant` 폴백 경로 | 배정이 없을 때의 무작위 선택도 가용 조합에서만 |

앞의 세 곳은 A/B 선호값을 알아야 네 셀을 구성할 수 있으므로 `ab_values`를 참조하도록
시그니처를 맞춘다. **`assign_ab_values`(선호축 A/B 값을 정하는 함수)는 건드리지
않는다.** A/B 값과 그 카운터밸런스는 v4와 완전히 동일하게 유지되어야 한다.

### CLI

```
--cell-library PATH      # annotation/attribute_library.json
```
플래그가 없으면 필터는 꺼진 채 v4와 동일하게 동작해야 한다.

---

## 3. 변경 B — `solid`를 무늬 축의 기준선으로

이번 작업에서 개념적으로 가장 중요한 변경이다.

### 왜 — 무늬 축에만 영(null) 수준이 있다

옷은 반드시 무언가를 입어야 하고, 색은 반드시 무언가를 띤다. 그러나 **무늬는 "없음"이
가능하고, 그것이 `solid`다.**

v4는 이 영 수준을 나머지 다섯 무늬와 동급의 값으로 취급해, 모든 사용자에게 `solid`에
대한 호/불호를 배정했다. 실험 설계로 말하면 **대조 조건을 처치 수준 중 하나로 넣은
것**이다. 결과는 두 가지였다.

**결과 1 — 배경축 빈칸 803개.** 배경축 값은 "사용자에게 취향 중립이면서 상황에도
안전한 값"이어야 한다. 그런데 무늬 배정 규칙(잔잔한 무늬 solid·striped에서 1+1)
때문에 **24명 전원이 solid와 striped를 모두 취향값으로 가졌다.** 배경축 후보 목록은
`[solid, striped, checkered]`뿐이었으므로(화려한 무늬는 두 번째 시각 단서가 되지
않도록 의도적으로 제외), checkered마저 취향값인 사용자는 후보가 0이 되어 빈칸이
남았다. 803개 중 799개가 무늬 빈칸이다.

**결과 2 — 격식 상황에서 무늬 위반 문항이 0개.** 이쪽이 벤치마크 서사상 더 뼈아프다.
무늬가 위반축이 되려면 "상황에 맞으면서 취향 중립인 무늬"와 "상황에 안 맞으면서 취향
중립인 무늬"가 둘 다 필요하다. 격식 드레스코드가 허용하는 무늬는 사실상
{solid, striped}인데 이 둘이 모든 사용자의 취향값이므로, **적합-중립 후보가 어떤
사용자에게도 존재하지 않았다.** 실측하면 해당 질의 679개 중 무늬를 위반축으로 쓸 수
있는 것이 **0개**다. 즉 v4는 "장례식에 표범무늬는 부적절하다"는 문항을 구조적으로 단
하나도 만들 수 없었다. 이 벤치마크가 보여주고 싶은 가장 대표적인 대비인데도.

`solid`를 기준선으로 되돌리면 두 문제가 동시에 풀린다. 그리고 `solid`는 **무늬를
제약하는 29개 시나리오 전부에서 적합**하고 **이미지 확보율 100%(272/272)** 이므로
예외 처리가 필요 없다. 반면 v4가 배경으로 쓰던 화려한 무늬는 leopard 57%,
polka_dot 60%로 확보율이 낮아 배경 때문에 문항이 죽는 일이 잦았다.

### B-1. `construction/profile_generator.py`

```python
PATTERN_BASELINE   = "solid"
PATTERN_PREFERENCE = ["striped", "checkered", "floral", "polka_dot", "leopard"]
```

- 무늬 선호는 `PATTERN_PREFERENCE` 5종에 대해서만 **2 likes + 2 dislikes** 배정.
  `solid`는 절대 배정하지 않는다 → 전 사용자 공통 중립.
- **쿼터(2+2)와 사용자별 중립 개수(2개)는 v4와 동일하다.** 바뀌는 것은 배정 대상
  어휘가 6종 → 5종이 되고, 중립 2개 중 하나가 항상 `solid`로 고정된다는 점뿐이다.
- 배정 방식: v4가 색축에 이미 쓰는 **전역 least-used-first**. 좋아함/싫어함 중 어느
  쪽이 먼저 뽑을지 프로필 인덱스 홀짝으로 번갈아 가게 해, 한쪽이 상대의 선택에 계속
  제약받는 것을 막는다(이렇게 하지 않으면 전역 편차가 1을 넘는다).
- **RNG draw를 소비하지 말 것.** 기존 무늬 배정 블록도 RNG를 쓰지 않는다. 소비하면
  뒤따르는 의류·색 배정이 전부 어긋나 프로필이 달라진다.
- 기존 어서션(`PATTERN_QUIET` 1+1, `PATTERN_EXPRESSIVE` 1+1)을 다음으로 교체:
  - `solid`가 어떤 프로필의 likes/dislikes에도 없을 것
  - likes/dislikes가 `PATTERN_PREFERENCE`의 부분집합일 것
  - 5종 각각의 전역 like/dislike 횟수 편차가 1 이하일 것
- `PATTERN_QUIET` / `PATTERN_EXPRESSIVE` 상수는 v4 재현용으로 남겨둔다(삭제 금지).

### B-2. `construction/option_planner.py`

`plan_option_variant` 초입, `fixed_attrs`를 만든 직후:

```python
if _SOLID_BASELINE and "pattern" not in (active_axis, violation_axis):
    fixed_attrs["pattern"] = "solid"
```

**중립 무늬 중에서 고르는 것이 아니라 항상 solid로 고정한다.** 이유는 두 가지다.
첫째, 사용자마다 남는 중립 유무늬가 다르므로(U001은 striped, U002는 leopard) 그것을
쓰면 문항마다 배경의 성질이 달라진다. 둘째, 유무늬는 이미지 확보율이 낮아 배경
때문에 문항이 죽는다.

배경축 값은 A/B/C/D에 **동일하게** 들어가므로 대비가 아니라 상수이며, 어느 선지가
정답인지 구별하는 데 기여할 수 없다.

### CLI

```
--solid-baseline
```
플래그가 없으면 꺼진 채 v4와 동일 동작.

---

## 4. 변경 C — 시나리오 단위 짝 균형 (`assign_ab_values`)

### 왜

사용자별 짝 순환(`user_pair`)은 정상 작동하지만, **같은 시나리오 안에서 여러
사용자가 같은 짝으로 몰리는 것**을 막는 항이 없었다. 사용자 각각의 순환은 멀쩡한데
시나리오 단면으로 자르면 편중이 보이는 상황이다.

### 어떻게

`scen_pair[(scenario_id, frozenset({a, b}))]` 카운터를 추가하고 후보 정렬
우선순위를 다음으로 둔다:

```
user_pair  →  scen_pair  →  pair_use  →  per-user value use  →  net  →  random
```

**전역 균형(`pair_use`)을 사용자별 항보다 앞에 두지 말 것.** 기존 docstring에 이유가
적혀 있다 — 한 사용자가 같은 A값만 계속 받게 되는 붕괴가 관측된 적이 있다.

---

## 5. 변경 D — `wedding_reception` 무늬 제약 수정 (`configs/scenarios.py`)

### 왜 — 근거는 **관습**이며, 커버리지가 아니다

v7 개정에서 `floral`을 `leopard`와 함께 부적합으로 분류했다. 그러나 **미국 주류
관습에서 꽃무늬 하객 복장은 결혼식 피로연의 일반적인 선택**이며, 이를 "loud print"로
분류할 근거가 없다. 근거는 세 가지다.

1. v10에서 이미 같은 성격의 수정을 했다 — 비즈니스 4개 시나리오와 골프에서 전면적
   floral 금지를 완화했다. 결혼식만 남은 것이 오히려 일관되지 않다.
2. 인접 시나리오 `social_gallery_opening`은 floral을 이미 허용하고 있다.
3. 꽃무늬 하객 드레스는 봄·여름 웨딩 에티켓 가이드에서 표준적으로 권장된다.

> **문서화 시 반드시 지킬 것:** 개정 이력과 논문에는 **관습 근거만** 적는다.
> 커버리지 효과(12명 미만 시나리오 1개 → 0개)는 **결과이지 동기가 아니다.**
> 순서를 뒤집어 쓰면 "표본이 모자라서 관습 판정을 바꿨다"로 읽히고, 이 벤치마크의
> 핵심 방어선(README: "TPO 판정은 저자의 취향이 아니라 물리적 위험·기능적 불가능·
> 널리 공유된 관습에 근거한다")이 무너진다. 리뷰어가 개정 이력을 읽는다고 가정하라.

### 어떻게

```diff
-  'pattern': {'compatible': ['solid', 'striped'],
-              'incompatible': ['leopard', 'floral']},
+  'pattern': {'compatible': ['solid', 'striped', 'floral'],
+              'incompatible': ['leopard', 'polka_dot']},
```

**`wedding_reception` 항목에만 적용한다.** 동일한 무늬 블록이 파일 안에 3곳 있으므로
`scenario_id`로 범위를 좁혀 치환할 것. 다른 wedding/celebration 시나리오는 이번에
건드리지 않는다.

`polka_dot`을 부적합에 추가하는 것은 부적합 목록을 2개로 유지하기 위한 형식적
조치다. 격식 피로연에서 물방울무늬가 leopard보다 덜 튄다고 볼 근거는 약하므로,
**부적합을 `leopard` 하나만 남기는 선택지도 검토 대상으로 두고 저자에게 확인할 것.**

---

## 6. 변경 E — 트랙 용어 개명: `physical` → `functional`, `dress_code` → `normative`

### 왜

현재 두 트랙 이름은 **층위가 어긋나 있다.** `Physical`은 제약의 *원인*(물리 법칙)을
가리키고, `Dress-code`는 제약의 *형식*(복장 규정)을 가리킨다. 서로 대칭이 아니다.
게다가 "dress code"는 일상어로 "복장 규정 전반"을 뜻해 physical 시나리오까지 포함하는
것처럼 읽히고, "physical"은 옷의 물성(면, 신축성)으로 오해될 여지가 있다.

`Functional / Normative`는 둘 다 **제약의 성격**을 같은 층위에서 가리킨다. "옷이 그
상황에서 기능하는가" vs "규범에 부합하는가". 그리고 이 쌍은 README가 이미 선언한 TPO
판정 근거 — "물리적 위험·기능적 불가능·널리 공유된 관습" — 과 직접 대응한다. 앞의
둘이 functional, 뒤가 normative다. 트랙 이름이 판정 근거의 분류와 일치하면 리뷰어에게
설계 일관성으로 읽힌다.

### 어떻게

트랙 이름은 상수로 중앙화되어 있어 개명이 깔끔하다. `configs/scenarios.py`:

```python
TRACK_FUNCTIONAL = "functional"      # was TRACK_PHYSICAL = "physical"
TRACK_NORMATIVE  = "normative"       # was TRACK_DRESS_CODE = "dress_code"

FUNCTIONAL_ARCHETYPES = frozenset({...})   # was PHYSICAL_ARCHETYPES
NORMATIVE_ARCHETYPES  = frozenset({...})   # was DRESS_CODE_ARCHETYPES
```

**개명 대상:**

| 파일 | 내용 |
|---|---|
| `configs/scenarios.py` | 상수 4개, `track_for_archetype()`, 트랙 정의 주석(160–165행), S-checks |
| `construction/option_planner.py` | `TRACK_PHYSICAL`/`TRACK_DRESS_CODE` import 및 리포트 루프 |
| `construction/compatibility.py` | docstring |
| `construction/query_generator.py` | docstring |
| `construction/profile_generator.py` | docstring |
| `scripts/multimodal_eval.py` | `--track` 인자 help 및 파싱 |
| `scripts/text_only_eval.py` | 동일 |
| `README.md`, `construction/README.md`, `docs/PROCESS.md` | 서술 전반 |

**주의 — 무차별 치환 금지.** `physical`이라는 단어는 트랙 이름이 아닌 곳에도 많이
쓰인다. 시나리오 `justification` 문장들("physically unbearable", "physically
impossible")과 `annotation/serve_annotator.py`의 "physical run"(트랙이 아니라 실행
이력을 뜻함)은 **그대로 두어야 한다.** 반드시 다음 순서로 진행할 것:

1. `grep -rn "physical\|dress_code\|PHYSICAL\|DRESS_CODE"` 로 전체 출현 지점 나열
2. 각 지점이 트랙을 가리키는지 사람이 판단
3. 트랙을 가리키는 것만 변경
4. `before_configs/` 디렉터리는 과거 스냅샷이므로 **건드리지 않는다**

**과거 산출물 호환:** `data_wacv_scenario_v3/`, `v4/`의 `option_plans.jsonl`은
`track` 필드에 `"physical"`/`"dress_code"`를 갖고 있다. v3/v4를 다시 읽는 스크립트가
있다면 로드 시점에 매핑하는 얇은 호환 레이어를 두고, **v5 이후 산출물에는 새 이름만
쓴다.** 새 이름과 옛 이름이 같은 파일 안에 섞이지 않게 할 것.

---

## 7. 최종 규칙 명세 (구현 후 이 명세와 대조할 것)

### 7-1. 프로필 구성

| 축 | 좋아함 | 싫어함 | 구성 | 중립 |
|---|---:|---:|---|---:|
| 의류 (23종) | 4 | 4 | ANCHOR 7종에서 2+2, FREE 16종에서 2+2 | **15** |
| 색 (13종) | 3 | 3 | **전체에서 전역 균형 3+3 (ANCHOR 티어 없음)** | **7** |
| 무늬 (6종) | 2 | 2 | **유무늬 5종에서 2+2, `solid`는 배정 제외** | **2** |

색축 주의: v4 문서에는 "격식 공통 허용색 3종에서 1+1"이라고 적혀 있으나 **현재 코드는
그 티어를 이미 제거했다**(코드 주석의 "The old 3-anchor color split" 참조). 실제
프로필을 보면 사용자마다 black/gray/navy를 0~3개까지 제각각 가진다(예: U004는 0개).
문서를 코드에 맞춰 갱신할 것이며, **색 배정 로직을 되돌리지 말 것.**

무늬 중립 2개 = `solid`(전원 공통) + 유무늬 1종(사용자마다 다름).

### 7-2. Functional 트랙 (위반축은 항상 의류)

시나리오가 의류만 제약하므로 색·무늬는 순수 선호 축이 된다.

- 선호축 = **색** → A/C는 좋아하는 색, B/D는 싫어하는 색. 배경 무늬 = **항상 `solid`**
- 선호축 = **무늬** → A/C·B/D는 유무늬 2+2 중 1+1. 배경 색 = 중립 색 7개 중 하나
- 어느 경우든 `solid`는 선호축 A/B에 **절대 등장하지 않는다**

### 7-3. Normative 트랙 (선호축은 3축 모두 가능)

시나리오가 2~3개 축을 제약하므로 배경축이 제약 대상일 수 있다. 남는 축의 처리:

| 배경축 | 처리 |
|---|---|
| **무늬** | **항상 `solid`.** 시나리오 제약 여부와 무관 — `solid`는 무늬를 제약하는 29개 시나리오 전부에서 적합하므로 안전하다 |
| **색** | 시나리오가 색을 제약하면 `시나리오 허용색 ∩ 사용자 중립색`에서 하나. **교집합이 비면 문항 생성 실패**(억지로 채우지 않는다). 제약하지 않으면 중립색 하나 |
| **의류** | 공유 의류로 지정 (빈칸 문제 없음) |

### 7-4. 불변식 (검증기가 확인해야 할 것)

- 선호축: A·C = 사용자가 **좋아하는** 값, B·D = **싫어하는** 값, A=C, B=D
- 위반축: A·B = 시나리오 **적합** 값, C·D = **부적합** 값, 그리고 **두 값 모두 그
  사용자에게 취향 중립**
- 배경축: 네 선지 **동일** + 취향 중립 + 시나리오에 어긋나지 않음
- 선호축 ≠ 위반축
- Functional 트랙: 위반축은 반드시 의류, 선호축은 색 또는 무늬, 시나리오가 색·무늬를
  제약하지 않음

---

## 8. 실행 및 수용 기준

### 실행

```bash
export POD_VARIANT=wacv_scenario_v5

python -m construction.profile_generator --force
python -m construction.query_generator   --force
python -m construction.option_planner    --force \
    --cell-library annotation/attribute_library.json \
    --solid-baseline

python -m configs.scenarios          # S-checks
python -m configs.profiles           # P-checks
python -m scripts.validate_options
python -m scripts.report_track_balance
python -m tests.test_option_validator_mutations
```

`scripts/verify_release.sh`가 v5를 재생성·비교하도록 갱신하고, 해시 목록에
`annotation/attribute_library.json`을 **생성 입력**으로 추가한다:

```
attribute_library.json  72cc8665d6f92d143f850b751bba1767c342a2e9b857292e6738666bea86baae
```

### 무결성 기준 (하나라도 실패하면 중단하고 보고)

| 항목 | 기준 |
|---|---|
| `validate_options` 구조/선호 위반 | **0** |
| `validate_options` TPO 위반 | **0** |
| `test_option_validator_mutations` | **48/48 detected** |
| 파이프라인 2회 실행 SHA256 | 세 파일 모두 **일치** |
| 선호축 A/B에 `solid` 등장 | **0회** |
| 배경축 값이 취향 중립이 아닌 문항 | **0** |
| 배경축 값이 네 선지에서 다른 문항 | **0** |

### 규모·커버리지 기준 (참조 구현 기준, ±3% 허용)

| 항목 | v4 | v5 목표 |
|---|---:|---:|
| 생성 문항 | 3,027 | ~2,641 |
| **네 선지 모두 이미지 존재** | 1,012 (33.4%) | **~2,571 (97%↑)** |
| 문항 0개 시나리오 | 6 | **0** |
| 12명 미만 시나리오 | 24 | **0** |
| 24명 전원 커버 | 3/59 | **33/59↑** |
| functional 선호축 색:무늬 | 42:58 | **~55:45** |
| 격식 14개 시나리오의 무늬 위반축 | 0 | **250↑** |

부활해야 하는 시나리오(v4에서 전부 0문항): `mourn_funeral`, `mourn_memorial`,
`civic_court_appearance`, `celebration_graduation`, `stage_greenscreen_shoot`,
`stage_tv_interview`

### 참조 SHA256 (참조 구현, seed 42)

```
profiles.jsonl      7d1cda17eccf3d73337b50bc7ed36f63e96214193930a9ea37760e187a0afd71
queries.jsonl       cc87f1c2281a52d9fe7f28676358a85963e4a17428b9eb619776210528f27671
option_plans.jsonl  813e9ab11d955794aa8eaf0b389970dc12166db8c094b124f1d8442773241f29
```

주: 이 해시는 **트랙 개명 이전** 참조 구현의 값이다. 개명 후에는 `track` 필드 값이
바뀌므로 `option_plans.jsonl` 해시가 달라진다. 판정은 **재실행 일치(결정론)와 위
지표**로 한다. 직접 구현하는 경우 tie-break 순서 차이로도 해시가 달라질 수 있다.

---

## 9. 문서 갱신

### `configs/scenarios.py` 상단 개정 이력

```
v14 (wacv_scenario_v5) — wedding_reception 무늬 제약 수정:
  v7에서 floral을 leopard와 함께 부적합으로 분류했으나, 꽃무늬 하객 복장은 미국
  주류 관습에서 결혼식 피로연의 일반적 선택이며 이 분류에는 관습적 근거가 없었다.
  floral을 compatible로 이동하고 polka_dot을 incompatible에 추가한다.
  v10에서 비즈니스·골프 시나리오의 전면적 floral 금지를 완화한 것과 같은 성격의
  수정이며, 인접 시나리오 social_gallery_opening의 분류와도 일치한다.

v14 (wacv_scenario_v5) — 트랙 용어 개명: physical → functional,
  dress_code → normative. 기존 쌍은 층위가 어긋나 있었다(제약의 원인 vs 제약의
  형식). 새 쌍은 둘 다 제약의 성격을 같은 층위에서 가리키며, README가 선언한 TPO
  판정 근거("물리적 위험·기능적 불가능" / "널리 공유된 관습")와 직접 대응한다.
```

### `construction/profile_generator.py` 상단

§3의 영 수준 근거를 주석으로 남긴다. 왜 `solid`가 배정에서 빠지는지, v4에서 무엇이
잘못됐는지를 후임자가 읽고 이해할 수 있어야 한다.

### `README.md`

- 트랙 이름 전면 갱신 (Functional / Normative)
- 프로필 쿼터 표를 §7-1대로 갱신. 특히 **색축의 ANCHOR 티어 서술을 삭제**하고,
  무늬는 "유무늬 5종에서 2+2, `solid`는 기준선으로 예약(전 사용자 중립)"으로
- 사용자별 중립값: 의류 15종 / 색 7종 / 무늬 2종(`solid` + 유무늬 1종)
- 파이프라인 설명에 "Stage 3은 `annotation/attribute_library.json`을 입력으로 받는다"
  추가
- 현재 변형을 `wacv_scenario_v5`로, 해시 갱신

### `docs/wacv_scenario_v5_report.md` (신규)

v3/v4 리포트와 같은 형식. §8의 지표 표와 §10의 한계를 포함할 것.

### `docs/PROCESS.md`

§7의 "다음 재현 가능 단계"를 갱신한다. 1번(803문항 처리 결정)과 2번(1,012 유지 vs
재생성)은 이 작업으로 **해결됨**으로 표시하고, 남은 항목은 manifest 동결 → 이미지
폴더 실체화 → preflight → 트랙 분리 평가다.

---

## 10. 논문에 반드시 서술할 한계 (알고리즘으로 고칠 수 없음)

이 네 가지는 균형 알고리즘으로 완화할 수 없음을 실측으로 확인했다. 숨기지 말고
원인과 함께 서술하는 편이 방어에 유리하다.

1. **격식 14개 시나리오에서 무늬는 선호축이 될 수 없다.** 허용 무늬가
   {solid, striped}뿐이고 `solid`가 중립이므로 like/dislike 짝이 성립하지 않는다.
   이는 상황의 성질이다 — 무늬 선택지가 없는 자리에서는 무늬 취향을 표현할 여지도
   없다(README의 "겨울의 탱크탑" 논리와 동일). 무늬 선호는 선택지가 열려 있는 21개
   시나리오에서 측정된다. **같은 시나리오에서 무늬는 대신 규범 위반축으로 기능하며,
   v4에서 구조적으로 0개였던 문항이 v5에서 생성된다.**

2. **무늬 위반축이 "solid → X"로 수렴한다(최빈 약 47%).** 해당 질의 438개 **전부**
   적합-중립 후보가 `solid` 단일이다. 균형 항을 추가해도 분산할 후보가 없다. 격식
   드레스코드가 허용하는 무늬가 사실상 무지뿐이기 때문이며, 데이터 아티팩트가 아니라
   도메인의 성질이다.

3. **normative 색 선호축 최빈 짝 약 7.7%(navy/white).** 5명의 사용자에게 집중되며,
   해당 문항들이 가졌던 대안 짝은 평균 1.5개(최소 1개)였다. 격식 상황의 허용 색이
   좁아 표현 가능한 선호 대비 자체가 제한된다.

4. **축별 쿼터는 의도적으로 비대칭이다:** 의류 4+4(23종 중), 색 3+3(13종 중),
   무늬 2+2(유무늬 5종 중, `solid`는 기준선). 어휘 크기와 영 수준의 유무가 축마다
   다르기 때문이며, **축별 지표를 결코 통합하지 않는 이유**이기도 하다.

### 논문 문장 초안

> 무늬 축은 다른 두 축과 달리 영(null) 수준을 가진다. `solid`는 무늬 값이 아니라
> 무늬의 부재이며, 요인 설계에서 대조 조건을 처치 수준 중 하나로 취급하지 않는 것과
> 같은 이유로 선호는 다섯 개의 유무늬 값에 대해서만 정의하고 `solid`는 기준선으로
> 유지한다. 무늬가 선호축도 규범축도 아닌 문항에서는 네 선지 모두 무지로 고정된다.

> 옵션 생성기는 사람이 검수한 이미지 셀 목록(해시 고정)을 입력으로 받아, 네 선지
> 모두에 이미지가 존재하는 조합만 채택한다. 이는 사후 필터가 아니라 후보 제약이므로
> 균형 목적함수가 실현 가능한 범위 안에서 최적화된다.

> 시나리오를 두 트랙으로 분리한다. **Functional** 트랙에서는 위반이 물리적 위험이나
> 기능적 불가능을 초래하며 제약은 의류 종류에만 걸린다(색·무늬는 순수 선호 축).
> **Normative** 트랙에서는 위반이 사회적 부적절을 초래하며 의류·색·무늬 세 축 모두
> 제약될 수 있다. 두 트랙은 별도로 구성하고 별도로 채점하며 지표를 통합하지 않는다.

---

## 11. 이번 작업에서 하지 말 것

- `data_wacv_scenario_v4/`, `data_wacv_scenario_v3/`, `annotation/`,
  `availability_audit/`, `before_configs/` 수정
- GPU 작업 — 이미지 재수집, SAM3 스크리닝, 재주석
- `wedding_reception` 외 다른 시나리오의 제약 변경
- 평가 프롬프트(`EVAL_FRAME_CLAUSE`, `EVAL_PRIORITY_CLAUSE_*`) 수정
- 이미지 폴더 실체화 — 별도 단계다
- 색 배정 로직에 ANCHOR 티어 재도입
- `assign_ab_values`의 A/B 선호값 결정 로직 변경

## 12. 후속 작업 (별도 세션)

1. 미주석 `solid` 셀 27개 추가 주석 → 문항 추가 확보 (반나절)
2. `plan_id` manifest 동결 후 밸런스·카운터밸런스 리포트를 **manifest 기준으로**
   재산출 (3,027 기준 리포트를 복사하지 말 것)
3. `data_wacv_scenario_v5/images/<plan_id>/{A,B,C,D}.jpg` 실체화. 경로는 반드시
   `attribute_library.json`을 통해서만 해석할 것 — `images_final/`에 superseded
   파일이 99개 있으므로 **폴더 glob 금지**
4. `scripts.multimodal_eval.preflight_images` 후 트랙 분리 평가
