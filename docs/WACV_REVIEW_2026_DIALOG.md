# POD-Bench — Part II: Multimodal Dialog × ABCD 설계 리뷰 (2026-08-02)

Part I(`docs/WACV_REVIEW_2026.md`)는 "명시적 프로필 + ABCD"를 최종 과제로 가정하고
쓴 리뷰다. 저자 확인 결과 **최종 주제는 `multimodal dialog × ABCD`이고, 명시적
프로필은 실험 조건 축(setting) 중 하나**다. 이 문서는 그 전제 위에서 다시 쓴
리뷰다.

Part I의 §3–§12는 그대로 유효하다(결과 부재, 지표 종속성, 통계 부재, MCQ 편향,
이미지 교란, 문화 타당성, 사실관계 오류). **Part I §2 "personalization 정의가
약하다"만 이 문서로 대체된다.**

---

## 0. 방향 전환에 대한 총평

**이 전환은 옳다. 그리고 지금까지 한 작업이 하나도 버려지지 않는다.**

- 명시적 프로필 조건은 폐기되는 게 아니라 **oracle ceiling (상한 조건)** 이 된다.
  "선호를 문자 그대로 줬을 때조차 X%"라는 숫자는 대화 조건의 해석 기준선이므로,
  Part I이 지적한 모든 구성 작업(2×2 불변식, counterbalance, availability
  constraint, mutation test)이 **그대로 필요하고 그대로 쓰인다.**
- 더 중요한 것: 이 설계는 대화 개인화 벤치마크가 거의 갖지 못한 자산을 갖는다 —
  **잠재 변수(사용자 선호)의 ground truth가 정의상 알려져 있다.** 프로필을 먼저
  정하고 대화를 역설계하기 때문이다. PrefEval·PersonaMem·Latent-Preference 계열은
  선호 라벨이 어노테이터 판단이거나 시나리오 서술이지만, 여기서는 **13색 / 5패턴 /
  23의류의 폐쇄 어휘 위에서 정확히 채점 가능한 잠재 변수**다. 이게 셀링 포인트다.

**동시에, 리스크는 사라진 게 아니라 이동했다.** Part I §2의 리스크("프로필이 답을
문자열로 준다")는 이제 **"대화가 답을 문자열로 준다"** 로 옮겨간다. 대화를 LLM으로
생성하면 십중팔구 `"I really love checkered patterns"`가 나오고, 그 순간
multimodal dialog 조건은 narrative-profile 조건과 정보량이 동일해진다.
**이 문서의 절반은 그 실패를 어떻게 측정하고 막는가에 대한 것이다.**

---

## 1. 이 설계의 novelty는 정확히 어디에 있는가

리뷰어는 "multimodal dialog + MCQ = 기존 조합"으로 읽으려 한다. 조합으로 읽히지
않으려면 **아래 두 가지 중 최소 하나**를 실제 측정으로 보여야 한다.

### N1. 증거가 이미지에만 있는 대화 (evidence-in-image)

같은 선호 증거를 세 형태로 제시하는 **조작 변수**를 만든다.

| 조건 | 대화 속 증거 형태 | 속성이 어디에 있나 |
|---|---|---|
| **T** | `"체크무늬 셔츠 하나 샀는데 진짜 마음에 들어"` | 텍스트 |
| **I** | `[체크무늬 셔츠 이미지] + "이거 샀는데 진짜 마음에 들어"` | **이미지에만** |
| **T+I** | 이미지 + 속성 명시 텍스트 | 양쪽 (중복) |

**I 조건의 정확도가 T 조건보다 유의하게 낮다** — 이 한 줄이 논문의 헤드라인이 될
수 있다. "VLM은 대화 속 시각 증거로부터 사용자 선호를 추론하지 못한다"는 주장은
MMPB에도, PrefEval에도, SIMMC 2.0에도, Latent-Preference 벤치에도 없다.
그리고 이 벤치는 그걸 **정확히 채점할 수 있는 유일한 구조**를 이미 갖고 있다
(폐쇄 어휘 + 인간 검증된 셀 이미지).

여기서 결정적인 자산: `annotation/attribute_library.json`의 1,237개 셀은
**색·패턴·의류가 인간에 의해 검증된** 이미지다. 즉 I 조건의 "이미지에 그 속성이
실제로 있다"는 것이 **이미 보장되어 있다.** 이건 대부분의 연구가 못 하는 것이다.

### N2. 추론(recognize)과 활용(follow)의 분리

PrefEval의 제목이 "Do LLMs **Recognize** Your Preferences?"인 것은 우연이 아니다.
대화 기반 개인화는 두 능력의 합성이다:

```
대화 → [Stage 1: 선호 추론] → 잠재 프로필 → [Stage 2: 상황과 조율] → ABCD
```

이 벤치는 **Stage 2에 이미 2×2 구조를 갖고 있으므로**, Stage 1을 붙이면
**두 단계를 분리 채점할 수 있는 드문 구조**가 된다. §5에서 구체화한다.

### 이 둘이 없으면

"SIMMC 2.0 스타일 대화 + MMPB 스타일 개인화 + MCQ" 조합으로 읽힌다.
N1·N2 중 최소 하나는 **main table에 들어가야 한다.**

---

## 2. 최대 위협 — Evidence Validity (대화가 정답을 결정하는가)

문항 하나의 정답 A는 `(active axis의 liked value a, disliked value b)`로 결정된다.
대화 조건에서 이 문항이 유효하려면 **대화가 "a를 좋아함"과 "b를 싫어함"을 모두
담고 있어야** 한다. 세 가지 실패 모드가 있고, 각각 측정량이 필요하다.

### F1. Leakage — 너무 명시적

대화에 정답 어휘가 그대로 등장 → narrative profile과 동일. novelty 소멸.

**측정: Canonical Lexical Leakage Rate.**
대화 텍스트에서 `configs/config.py`의 정규 어휘(및 렌더링 별칭 — "leopard print",
"formal shirt" 등)가 **선호 극성과 함께** 문자 그대로 등장하는 비율.
조건별로 보고:
- explicit 대화: 높은 게 정상 (설계 의도)
- implicit 대화: **이게 낮다는 것이 implicit 조건의 정의**여야 한다.
  "implicit"이라고 이름 붙이고 어휘가 그대로 나오면 그건 explicit이다.

이 검사는 이미 v9에서 240개 시드에 대해 한 번 한 작업이다
(README "v9: explicit/implicit sharpening after a full 240-seed audit — implicit
seeds never state the constraint (8 leaky seeds rewritten)"). **같은 감사를 대화에
대해 자동화된 형태로 다시 하라.** 그때는 8건을 손으로 찾았지만 대화는 수천 개가
될 것이다.

### F2. Under-determination — 증거 부족

대화에 그 문항의 (a, b) 축 증거가 없으면 그 문항은 대화 조건에서 **풀 수 없는
문항**이다. 이런 문항이 섞이면 정확도가 낮아지는데, 그건 "어려운 벤치"가 아니라
**"틀린 벤치"** 다. 리뷰어는 이 둘을 구분해 달라고 요구한다.

**측정: Evidence Sufficiency Label (문항 단위, 3값).**

각 `(dialog, plan)` 쌍에 대해:
- `both` — 대화가 `a=like`와 `b=dislike` 증거를 모두 담음
- `one` — 한쪽만
- `none` — 둘 다 없음 (이 문항은 대화 조건에서 chance)

이 라벨은 **main table의 계층화 변수**이자 **문항 포함 기준**이다.
`both`만으로 main 숫자를 내고, `one`/`none`은 별도 진단으로 보고하라.

### F3. TPO leakage — 새로 생기는 위험, 놓치기 쉬움

대화에 `"지난번 장례식엔 검정 정장 입었어"` 같은 발화가 들어가면, 그건 선호가
아니라 **상황 규범(TPO)** 을 알려준다. 그러면 C/D를 배제하는 근거가 시나리오
지식이 아니라 대화가 되어 **2×2의 축 분리가 무너진다.**

**규칙: 대화는 선호 증거만 담고, 상황 규범 증거는 담지 않는다.**
**측정:** 대화에 시나리오 archetype 관련 어휘(`configs/scenarios.py`의 시나리오
이름·시드 어휘)가 등장하는 비율. 0에 가까워야 한다.

이건 Part I §5의 "지표가 종속이다"보다 심각한 문제다 — 지표는 재계산하면 되지만
축 오염은 데이터를 다시 만들어야 한다. **대화 생성 프롬프트를 쓰기 전에 이 제약을
먼저 못 박아라.**

### 셋을 묶는 하나의 게이트: Profile Recovery Rate (PRR)

세 실패 모드를 한 번에 진단하는 단일 측정이 있다.

> **강한 LLM(또는 사람)에게 대화만 주고, 폐쇄 어휘 위에서 사용자 프로필을
> 복원시킨다. 정답은 알려져 있다 (`profiles.jsonl`).**

- **PRR이 100%에 가까우면** → F1(leakage). 대화가 프로필을 그대로 말하고 있다.
- **PRR이 우연 수준이면** → F2(under-determination). 대화에 증거가 없다.
- **적정 구간(예: 60–85%)** → 유효한 implicit 증거.

PRR은 **데이터 품질 지표이자 동시에 §5의 Stage-1 성능 지표**다. 하나의 측정으로
두 목적을 만족하므로, 이걸 파이프라인의 필수 게이트로 넣는 것을 강하게 권한다.

보고 형태:
- axis별 (color / pattern / garment)
- **polarity별 (like / dislike)** ← §5에서 설명하듯 여기서 새 발견이 나올 가능성
- 조건별 (T / I / T+I, explicit / implicit)
- Macro-F1 + per-value confusion

---

## 3. 대화 생성 방법론 — 리뷰어가 반드시 묻는 5가지

### Q1. 누가 생성했나? 평가 대상과 겹치는가?

LLM 생성 대화를 LLM으로 평가하면 **self-preference bias**가 생긴다. GPT-계열이
쓴 대화는 GPT-계열이 더 잘 읽는다.

**완화책 (택1 이상):**
- 생성 모델을 평가 대상에서 제외 (가장 깨끗하지만 좋은 모델을 못 씀)
- **2개 이상의 서로 다른 계열로 생성하고, 생성자별 정확도 차이를 보고**
  ← 권장. "생성자 효과"가 모델 간 차이보다 작다는 것을 보이면 방어 완료.
  이건 표 한 개짜리 실험이고, 리뷰어가 가장 좋아하는 형태의 방어다.
- 인간 paraphrase 단계 추가 (SIMMC 2.0이 쓴 방식: simulator 생성 → 인간 재작성).
  비용이 크지만 표본 일부(예: 10%)에만 적용해 **"인간 재작성본에서의 정확도가
  동일한가"** 를 보이는 것으로 충분하다.

### Q2. 자연스러운가?

문헌상 zero-shot LLM 대화는 자연스러운 리듬·감정 표현·턴 교대가 부족하다.
**인간 평가 5차원**(coherence, fluency, consistency, relevance, naturalness)을
표본 200개에 대해 2명 이상이 평정하고 **Cohen's κ**를 보고하는 것이 표준이다.

### Q3. dislike 증거를 어떻게 자연스럽게 넣을 것인가 — **가장 어려운 설계 문제**

사람은 `"나는 주황색이 싫어"`라고 잘 말하지 않는다. 그런데 이 벤치는 B/D가
**반드시 disliked value**를 쓰도록 설계되어 있다(`README` "Strict 2×2 with real
dislikes"). 즉 **dislike 증거는 필수인데 가장 부자연스럽다.**

게다가 문헌이 경고하는 그대로다: zero-shot user simulator는 지나치게 순응적이고
**부정적·저항적 발화를 체계적으로 과소 표현한다.** 이 벤치는 그 약점이 정확히
급소에 오는 구조다.

**설계 제안 — dislike 증거의 자연스러운 형태 사다리:**

| 수준 | 형태 | 예 |
|---|---|---|
| E (explicit) | 직접 부정 | "주황색은 안 좋아해" |
| I1 | 회피 서술 | "그 색은 나한테 잘 안 어울리더라" |
| I2 | 부정 경험 | "저번에 산 주황색 니트는 결국 한 번도 안 입었어" |
| I3 | 행동 증거 | "[주황 스웨터 이미지] 이거 결국 반품했어" ← **I 조건과 결합** |
| I4 | 대조 선택 | "[두 이미지] 둘 중엔 왼쪽" (반복 관찰로 추론) |

I3/I4는 §1의 evidence-in-image 조건과 자연스럽게 결합되고, **행동 로그로부터의
선호 추론**이라는 추천 시스템의 실제 문제 설정과 정확히 일치한다.
I4(반복 선택 관찰)까지 가면 이건 사실상 **implicit feedback 기반 선호 학습**을
VLM 대화 문맥에서 평가하는 것이 되고, 그건 확실히 새로운 과제다.

**단, I4는 evidence sufficiency 판정이 어려워진다** — 몇 번의 선택이 "증거"인가에
대한 기준이 필요하다. 초판은 I1–I3까지만 쓰고 I4는 future work로 두는 것을 권한다.

### Q4. 대화가 특정 문항의 정답을 겨냥해 만들어졌는가?

**가장 위험한 설계 실수: 문항별로 대화를 생성하는 것.**
그러면 "대화가 답을 알고 만들어졌다"는 공격을 막을 수 없다.

**권고: 대화는 유저 단위로 생성한다.**
- 한 유저의 대화는 그 유저의 **모든** plan에서 재사용된다.
- 대화 생성 시점에 어떤 시나리오·어떤 문항이 쓰일지 모른다 → 구조적으로 무결.
- 생성 비용도 `24 × N`으로 끝난다 (문항별이면 2,641 × N).
- 대신 §2 F2의 evidence sufficiency 라벨이 **반드시** 필요해진다 (특정 문항의
  (a,b) 축 증거가 그 유저 대화에 없을 수 있으므로).

이 트레이드오프는 논문에 명시적으로 써라. "우리는 문항 조건부 생성을 의도적으로
피했고, 그 대가로 증거 충분성 라벨을 도입했다"는 서술은 리뷰어에게 매우 잘 읽힌다.

### Q5. 재현 가능한가?

Stage 1–3은 결정적(seed 42, SHA256 bit-identical)이라는 것이 이 프로젝트의 자랑인데,
**LLM 대화 생성은 비결정적이다.** 원칙 충돌을 먼저 선언하라.

> **Stage 0 (dialog synthesis)은 비결정적이며, 산출물 자체를 아티팩트로 동결하고
> 해시한다. Stage 1–3의 결정성 보장은 Stage 0 산출물을 입력으로 고정한 뒤에 성립한다.**

기록할 것: 생성 모델 ID, 프롬프트 버전(현 `PROMPT_VERSION` 관행 확장), temperature,
대화 파일 SHA256. `annotation/attribute_library.json`을 "generation input"으로
핀 고정한 v5의 방식이 **그대로 재사용 가능한 선례**다.

---

## 4. 실험 조건 사다리 — 이게 논문의 main table이다

프로필을 "축"으로 쓴다면, 그 축은 이런 모양이어야 한다.

| ID | 선호 정보 제공 방식 | 역할 |
|---|---|---|
| **P0** | 없음 | 하한. 설계상 active-value prior = 0.50 |
| **P1** | oracle 구조화 속성 (`structured_attributes`) | **상한 ceiling** |
| **P2** | oracle narrative (현재 조건) | 형식 민감도 |
| **P3** | 대화 · explicit · 텍스트 | |
| **P4** | 대화 · implicit · 텍스트 | |
| **P5** | 대화 · explicit · **이미지 증거** | **N1 핵심** |
| **P6** | 대화 · implicit · **이미지 증거** | **가장 어려움** |
| **P7** | P4/P6 + 무관 대화 N턴 삽입 | 지속성 (PrefEval 재현) |
| **P8** | 대화 + "먼저 프로필을 요약한 뒤 답하라" | two-stage 상한 |

**핵심 파생 지표 3개:**

1. **Personalization Gap = P1 − P6.**
   "선호를 문자열로 주는 것"과 "대화에서 추론하는 것"이 다른 문제임을 보이는 수치.
   이 gap이 작으면 논문의 전제가 무너지므로, **가장 먼저 파일럿으로 확인할 값**이다.
   (파일럿 권고: 유저 4명 × 문항 100개로 P1과 P6만 먼저 재본다. gap이 5%p 미만이면
   대화 설계를 다시 해야 한다는 신호다. 전체 생성 전에 반드시 하라.)

2. **Modality Gap = P3 − P5, P4 − P6.**
   같은 증거를 텍스트로 줄 때와 이미지로 줄 때의 차이 = N1의 헤드라인.

3. **Explicitness Gap = P3 − P4, P5 − P6.**
   PrefEval의 explicit/implicit 축을 시각 도메인으로 옮긴 것.

**P2가 이미 있다는 것이 큰 이점이다** — 지금까지의 모든 작업이 이 표의 P1·P2
행이고, 새로 만들 것은 P3–P8뿐이다.

---

## 5. 2단계 분해 — 이 논문의 핵심 그림이 될 것

```
대화 → [Stage 1] 잠재 프로필 추론 → [Stage 2] 상황과 조율 → ABCD
        └ PRR (§2)                  └ ABCD accuracy
```

### 채점 방식

- **Stage 1 = PRR** (대화만 주고 폐쇄 어휘로 프로필 복원, macro-F1)
- **Stage 2a = ABCD accuracy given oracle profile** (= P1)
- **Stage 2b = ABCD accuracy given the model's own recovered profile** (= P8)
- **End-to-end = P6**

### 2×2 진단표 — 이게 그림 하나로 들어가야 한다

| | ABCD 높음 | ABCD 낮음 |
|---|---|---|
| **PRR 높음** | 정상 | **추론은 되는데 활용을 못 함** ← PrefEval의 "recognize ≠ follow"를 시각 도메인에서 재현 |
| **PRR 낮음** | 우연/shortcut 의심 — **반드시 조사** | 추론 자체 실패 |

우하단은 예상 가능하지만, **우상단(PRR 높고 ABCD 낮음)** 이 관측되면 그것이 이
논문에서 가장 인용될 결과다. 그리고 좌하단(PRR 낮은데 ABCD 높음)은 **shortcut의
직접 증거**이므로 데이터 결함 탐지기로도 쓰인다.

### like/dislike 비대칭 — 조사하면 발견이 나올 가능성이 높은 지점

B/D는 **disliked value**를 쓴다. 즉 dislike를 추론하지 못하면 A와 B를 구분할 수
없다. 그런데 §3-Q3에서 봤듯 dislike는 표현도 어렵고 시뮬레이터도 과소 표현한다.

**예측 (검증 가치 있는 가설): PRR(like) ≫ PRR(dislike).**

이게 사실이면 "모델은 사용자가 무엇을 좋아하는지는 배우지만 무엇을 싫어하는지는
배우지 못한다"는 문장이 되고, 이건 추천 시스템·개인화 전반에 걸친 함의가 있다.
**측정 비용은 PRR을 극성별로 나눠 보고하는 것뿐이다 — 사실상 공짜다.**

---

## 6. 대화 길이·위치 — multimodal이라서 더 세게 나올 결과

PrefEval의 핵심 발견: **10턴(~3k 토큰)만 지나도 선호 준수 정확도가 10% 아래로
떨어진다** (텍스트 기준).

멀티모달에서는 이게 **훨씬 빨리** 온다고 예측할 근거가 있다. 이미지 1장이
수백~수천 비전 토큰을 소모하므로, 같은 "턴 수"에서 컨텍스트 압박이 몇 배다.

**조작 변수 3개를 분리해서 보고하라 — 이게 이 절의 핵심이다:**

| 변수 | 왜 분리해야 하나 |
|---|---|
| 증거–질의 사이 **턴 수** | 대화 구조의 효과 |
| 증거–질의 사이 **텍스트 토큰 수** | PrefEval과 직접 비교 가능 |
| 증거–질의 사이 **삽입된 이미지 장수** | **이게 novelty** |

세 번째가 핵심이다. "이미지 k장이 끼어들면 선호를 잊는가"를 턴 수·토큰 수와
**독립적으로** 보인 연구를 나는 찾지 못했다. 턴 수를 고정하고 이미지 수만 늘리는
설계가 가능하므로(무관 턴에 이미지를 넣거나 빼거나), 깨끗한 조작이 된다.

---

## 7. 새로 생기는 shortcut — 반드시 감사해야 할 것

### S1. 대화 이미지 ↔ 옵션 이미지 픽셀 매칭 (**최우선 위험**)

대화에서 보여준 "좋아하는 아이템" 이미지와 옵션 A의 이미지가 **같은 셀에서
나오면**, 모델은 선호를 추론할 필요 없이 **시각 유사도 매칭**으로 A를 찾는다.
극단적으로 같은 상품 사진이면 완전히 풀린다.

**여기서 실무적 문제가 있다:** `annotation/attribute_library.json`은 셀당 선택
이미지가 **1장**이다 (1,237셀 × 1장). 대화용 이미지를 같은 셀에서 뽑으려면
**추가 이미지가 필요하다.**

**선택지:**
1. `screen_sam3_candidates.jsonl`의 미선택 후보(19,012장 중)를 쓴다 →
   **인간 검증이 없다.** 대화 증거의 속성이 틀리면 잠재 변수가 오염된다. 위험.
2. **셀당 2장씩 재주석** (`pick_n=2`). 어노테이터 UI가 이미
   `mode=select, pick_n=1, show_n=10`으로 파라미터화되어 있으므로 구조 변경이
   아니라 설정 변경이다. **권장.** 대화용/옵션용 이미지 풀을 명시적으로 분리하고,
   `attribute_library.json`에 `role: option | dialog` 필드를 둔다.
3. 대화 증거를 옵션과 **다른 의류**로만 준다 (색/패턴 선호는 다른 옷에서 관찰).
   실제로 이게 가장 자연스럽기도 하다 — "체크무늬를 좋아한다"는 증거가 꼭 같은
   셔츠일 필요가 없다. **2번과 병행하면 가장 강하다.**

**감사 항목:** 모든 (대화 이미지, 옵션 이미지) 쌍의 perceptual hash / 임베딩
유사도 분포. 상위 꼬리를 반드시 눈으로 확인하고, 유사도 상위 문항에서의 정확도가
높지 않은지 검사하라 (높으면 shortcut 확정).

### S2. 대화 길이 자체가 단서

증거 턴이 항상 대화 끝에 있거나, 대화 길이가 정답과 상관되면 안 된다.
**증거 턴 위치를 무작위화하고, 위치와 정확도의 상관을 보고하라.**

### S3. 대화 스타일이 유저를 식별

같은 유저의 대화가 항상 같은 말투면, 모델이 스타일로 유저를 기억할 수 있다
(train/test 분리가 없는 벤치라 직접적 누출은 아니지만, 문항 간 정보 전이가 생긴다).
문항을 독립적으로 평가한다면 문제없지만, **각 문항이 독립 세션임을 명시**하라.

---

## 8. 규모 문제 — 지금이 유저를 늘릴 마지막 타이밍

대화를 주제로 삼는 순간 **유저 수가 병목**이 된다. 리뷰어:

> "A personalization benchmark with 24 users. Are the reported differences
> properties of personalization, or of these 24 profiles?"

Part I §7에서 지적한 cluster bootstrap이 여기서 더 심각해진다 — 클러스터가
`plan_id ⊂ query_id ⊂ (user_id, dialog_id)`로 한 겹 더 깊어지고, **최상위 클러스터가
24개뿐**이면 사실상 유저 수준 추론의 자유도가 23이다.

**좋은 소식: 유저 확장은 싸다.**
`construction/profile_generator.py`의 `generate_rule_variants(seed=42, n_variants=24)`는
`n_variants`가 단순 루프 파라미터이고, 배정은 전역 least-used-first 카운터 기반이다.
`run_pipeline`도 `n_users > len(all_variants)`일 때 경고하며 클램프할 뿐이다.
즉 **유저 100명 이상은 한 파라미터 변경 + 전역 균형 assertion 재확인**이지
재설계가 아니다. 제약은 조합론이 아니라 cell-liveness predicate(1,237셀)인데,
23의류 × 13색 × 5패턴 위에서 4+4 / 3+3 / 2+2 배정의 조합 수는 충분히 크다.

**권고: 대화 생성 착수 전에 유저를 100–200명으로 늘려라.**
대화를 만든 뒤에 늘리면 대화를 전부 다시 만들어야 한다. 순서가 되돌릴 수 없다.

부수 효과로 Part I §7-3의 "seed 3개 재생성 → 랭킹 안정성"도 같이 해결된다.

---

## 9. Related work 재배치

Part I §14-2의 표를 이 방향에 맞게 다시 그린다.

| 벤치마크 | 선호 표현 | 대화 | 시각 증거 | 상황 충돌 | 잠재변수 GT |
|---|---|---|---|---|---|
| SIMMC 2.0 (EMNLP'21) | task-oriented 대화 내 요구 | ✓ (11k 대화) | ✓ (장면) | ✗ | 대화 상태 |
| MMDialog / PhotoChat | ✗ | ✓ | ✓ | ✗ | ✗ |
| MMPB (NeurIPS'25) | 개념 주입 + 다중턴 | ✓ | ✓ | ✗ | 개념 ID |
| PerVL-Bench (WACV'26) | 개념 | — | ✓ | ✗ | 개념 ID |
| PrefEval (ICLR'25) | 명시/암시 선호, 장문맥 | ✓ | ✗ | ✗ | 선호 라벨 |
| Latent Preference (2510.17132) | 대화로 잠재 속성 발굴 | ✓ | ✗ | ✗ | 잠재 변수 |
| AlpsBench (SIGIR'26) | 실대화 기억 | ✓ | ✗ | ✗ | |
| PersonaMem / RealPref | 장기 이력 | ✓ | ✗ | ✗ | |
| CulturalVQA | ✗ | ✗ | ✓ | 문화 지식 | 다중 정답 |
| **ours** | **멀티모달 대화** | **✓** | **✓ (증거로서)** | **✓ (2×2)** | **✓ 역설계** |

**빈 열이 두 개다: "상황 충돌"과 "잠재변수 GT".** 이 두 열이 동시에 채워지는
행이 이 논문뿐이라는 것을 intro 한 문단으로 쓰면 novelty 서술이 끝난다.

특히 마지막 열을 강조하라: 대화 개인화 벤치는 대부분 "정답 선호"가 어노테이터
판단이거나 서술이지만, 이 연구는 **프로필을 먼저 정하고 대화를 역생성**하므로
잠재 변수가 정의상 알려져 있고 폐쇄 어휘 위에서 정확히 채점된다.
**이건 "우리가 대화를 합성했다"는 약점을 강점으로 뒤집는 논리다.** 합성했기
때문에 채점할 수 있다.

---

## 10. 실행 순서 권고

순서가 중요하다. 되돌릴 수 없는 결정이 앞에 있다.

```
[0] 파일럿 (1주) ─ 되돌릴 수 있는 유일한 구간
    · 유저 4명 × 문항 100개
    · P1(oracle) vs P6(dialog·implicit·image)만 측정
    · Personalization Gap ≥ 5%p 확인
    · PRR 적정 구간(60–85%) 확인
    → gap이 없으면 대화 설계를 다시. 여기서 멈출 수 있어야 한다.

[1] 되돌릴 수 없는 결정 (파일럿 통과 후 즉시)
    · 유저 24 → 100~200 확장               ← 대화 생성 전에 반드시
    · 셀당 이미지 2장 재주석 (option/dialog 풀 분리)  ← S1 차단
    · 대화 생성 규약 확정 (선호만, TPO 금지 / 유저 단위 / 증거 형태 사다리)

[2] 대화 생성 + 자동 감사
    · 생성 모델 2종 이상
    · Leakage / TPO-leakage / Evidence sufficiency 자동 라벨링
    · 대화 SHA256 동결, 생성 메타 기록

[3] 인간 검증
    · 대화 자연스러움 5차원 × 200개 × 2명 (Cohen's κ)
    · 인간 PRR (대화만 보고 프로필 복원) — 인간 상한
    · Part I §9의 dress-code 라벨 검증과 같은 세션에서 처리

[4] 본 실험
    · P0–P8 × 2 트랙 × ≥8 모델
    · PRR (axis × polarity × 조건)
    · 턴/토큰/이미지 수 스케일링
    · cluster bootstrap (user > dialog > query > plan)

[5] shortcut 감사
    · 대화-옵션 이미지 유사도 분포
    · 증거 턴 위치 × 정확도 상관
    · Part I §10의 4옵션 세트 저수준 교란
```

**[0]을 건너뛰지 마라.** 이 설계의 전제(P1과 P6이 다르다)가 틀리면 [1] 이후는
전부 매몰비용이 된다. 반대로 파일럿에서 gap이 크게 나오면 그 숫자 하나로 남은
전 과정의 정당성이 확보된다.

---

## 11. Part I에서 바뀌는 것 / 그대로인 것

**대체됨**
- Part I §2 ("personalization 정의가 약하다") → 이 문서 전체.
  단, **P1/P2 조건이 논문에 남는 한 §2의 지적은 "그 조건에 한해" 여전히 유효**하다.
  P1/P2를 "oracle ceiling"이라고 명명하는 것이 그 답이다.

**그대로 유효 (오히려 더 중요해짐)**
- §3 결과 부재 — 조건이 P0–P8로 늘어나 실험량이 3배가 된다. 더 급하다.
- §4 vision-essentiality — 이제 **옵션 이미지와 대화 이미지 두 곳**에서 필요.
- §5 지표 종속성 (TPO+pref = 1+strict−P(D)) — 조건이 늘수록 4-way 분포 보고가 필수.
- §6 프롬프트가 절차를 알려줌 — 대화 조건에서는 `EVAL_PRIORITY_CLAUSE`에 더해
  "대화에서 선호를 추론하라"는 지시를 줄지 말지가 **추가 ablation 축**이 된다.
- §7 통계 — 클러스터가 한 겹 깊어짐. 유저 24명이면 최상위 자유도 23.
- §8 MCQ 편향 / CircularEval — 그대로.
- §9 dress-code 인간 검증 — 그대로. §10의 [3]에서 함께 처리.
- §10 이미지 교란 — 그대로 + S1(대화-옵션 유사도) 추가.
- §11 지각 상한 probe — 그대로. **PRR의 전제조건**이기도 하다(옵션 속성을 못 보면
  선호를 알아도 못 고른다).
- §12 사실관계 오류 (MMPB/NaturalBench 규모) — 즉시 수정.

---

## 12. 추가 참고 문헌 (Part I §16에 더해)

**멀티모달 대화**
- SIMMC 2.0: A Task-oriented Dialog Dataset for Immersive Multimodal Conversations (EMNLP 2021) — https://aclanthology.org/2021.emnlp-main.401/
- MMDialog: A Large-scale Multi-turn Dialogue Dataset Towards Multi-modal Open-domain Conversation — https://arxiv.org/pdf/2211.05719
- DialogCC: An Automated Pipeline for Creating High-Quality Multi-Modal Dialogue Dataset — https://arxiv.org/pdf/2212.04119
- VDialogUE: A Unified Evaluation Benchmark for Visually-grounded Dialogue — https://arxiv.org/pdf/2309.07387
- BI-MDRG: Bridging Image History in Multimodal Dialogue Response Generation (ECCV 2024) — https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04591.pdf

**대화로부터의 선호 추론**
- Do LLMs Recognize Your Latent Preferences? A Benchmark for Latent Information Discovery — https://arxiv.org/abs/2510.17132
- AlpsBench: An LLM Personalization Benchmark for Real-Dialogue Memorization and Preference Alignment (SIGIR 2026) — https://doi.org/10.1145/3805712.3808634
- Toward Personalized LLM-Powered Agents: Foundations, Evaluation, and Future Directions — https://arxiv.org/pdf/2602.22680

**합성 대화의 타당성 / user simulator**
- User Simulation in the Era of Generative AI: User Modeling, Synthetic Data Generation, and System Evaluation — https://arxiv.org/pdf/2501.04410
- Large Language Models as User-Agents for Evaluating Task-Oriented-Dialogue Systems — https://arxiv.org/pdf/2411.09972
- ChatChecker: Dialogue System Testing Through Non-cooperative User Simulation — https://arxiv.org/pdf/2507.16792
- Synthetic Dialogue Data Generation: A Comprehensive Survey — https://www.cfilt.iitb.ac.in/resources/surveys/2025/synthetic-dialog-data-generation-survey-anshul.pdf
- Benchmark Data Contamination of Large Language Models: A Survey — https://arxiv.org/pdf/2406.04244
