# POD-Bench v2 — WACV Reviewer 관점 리뷰 (2026-08-02)

리뷰 대상: `README.md`, `docs/redesign_v2_plan.md`, `docs/PROCESS.md`,
`docs/wacv_scenario_v5_report.md`, `configs/`, `construction/`, `scripts/`,
`fsiglip/`, `annotation/`, `data_wacv_scenario_v5/`.

이 문서는 **코드를 수정하지 않고** 작성한 리뷰 보고서다. 미완성 연구라는 점을
전제로, (a) 현재 상태에서 리뷰어가 실제로 쓸 공격, (b) 그 공격을 막는 데 필요한
구체적 수치와 실험, (c) 관련 문헌에서의 위치를 정리했다.

---

## 0. 총평 — 가상 리뷰 스코어

| 항목 | 평가 |
|---|---|
| Novelty | **Borderline positive.** "선호 × 상황(TPO)"의 2×2 강제 분해는 기존 personalization 벤치마크(개념 인식 중심)에 없는 축이다. 다만 "personalization"의 정의가 약해서(§2) novelty가 절반으로 깎일 위험이 크다. |
| Technical soundness (construction) | **Strong accept 수준.** 여기가 이 연구의 진짜 강점이다. availability를 post-hoc filter가 아니라 generation constraint로 옮긴 v5 결정, `solid`를 baseline level로 재정의한 factorial-design 교정, mutation test 48/48, SHA256 재현성, 실패를 감추지 않는 문서화는 데이터셋 논문에서 보기 드문 수준이다. |
| Experimental validation | **Reject 수준 (현재).** 모델 결과가 0개, 인간 정답률 0개, 이미지 폴더 미조립. 데이터셋 논문에서 이건 치명적이다. |
| Reproducibility / documentation | **Strong.** |
| **현재 제출 시 예상 결과** | **Reject** — 단, 사유는 전부 "아직 안 한 것"이지 "잘못한 것"이 아니다. §7 로드맵을 채우면 Accept 권역. |

한 문장 요약: **구성(construction)은 이미 논문 수준을 넘었고, 검증(validation)은
아직 시작되지 않았다.** 남은 작업의 90%는 새 코드가 아니라 **측정**이다.

---

## 1. 강점 — 논문에서 반드시 전면에 내세워야 할 것

리뷰어가 놓치기 쉬우므로 abstract/intro에서 명시적으로 팔아야 한다.

1. **v4 → v5의 availability constraint 이동은 그 자체가 방법론적 기여다.**
   `docs/wacv_scenario_v5_report.md` §"Defect 1"의 논지 — "2,015개를 잃은 게
   문제가 아니라, **그 손실이 무작위가 아니었다는 게** 문제다 (track split
   38:62 → 31:69)" — 는 retrieval 기반 벤치마크를 만드는 모든 사람에게 해당되는
   교훈이다. 이건 부록이 아니라 **본문 한 절**로 써야 한다. 대부분의 데이터셋
   논문은 이 문제를 인지조차 못 한다.

2. **`solid`를 pattern 축의 baseline으로 재정의한 것 (Defect 2).**
   "통제 조건을 처치 수준으로 넣은 factorial design 오류"라는 프레이밍은
   정확하고, 그 결과 "장례식에 레오파드는 틀렸다"라는 이 벤치마크의 가장
   대표적인 대비가 v4에서는 **구조적으로 0건** 생성 불가능했다는 측정
   (0/3,027)은 강력한 증거다.

3. **Physical / Dress-code 트랙 분리와 "pooling 코드 경로가 존재하지 않음".**
   문화 의존적 라벨과 물리 법칙 라벨을 한 숫자로 섞지 않겠다는 결정을
   *코드로 강제*한 것(`split_by_track`이 track 없는 plan에서 abort)은
   리뷰어가 좋아한다.

4. **Mutation test (`tests/test_option_validator_mutations.py`, 48/48).**
   "실패할 수 없는 validator는 증거가 아니다"는 논지 그대로 논문에 써라.

5. **한계를 숨기지 않는 서술.** `wacv_scenario_v5_report.md`의
   "Limitations that cannot be fixed by balancing" 4항목, 70개 미실현 plan의
   원인별 분해, "마지막 행은 목표를 못 맞췄고 문제는 목표에 있다"는 문장.
   이런 서술은 리뷰어의 신뢰를 산다. **절대 다듬어서 빼지 마라.**

---

## 2. Major concern #1 — "Personalization"의 정의가 약하다 (가장 큰 novelty 리스크)

> **[2026-08-02 갱신]** 저자 확인 결과 최종 주제는 `multimodal dialog × ABCD`이며,
> 명시적 프로필은 실험 조건 축 중 하나(oracle ceiling)다. **이 절은
> `docs/WACV_REVIEW_2026_DIALOG.md`로 대체된다.** 단, P1/P2(oracle profile) 조건이
> 논문에 남는 한 아래 지적은 "그 조건에 한해" 여전히 유효하다.
> 나머지 절(§3–§15)은 방향 전환 후에도 그대로 유효하다.

### 지적

`data_wacv_scenario_v5/profiles/profiles.jsonl`의 `narrative_profile`은 이렇다:

> "This user likes garment categories such as formal shirt, long coat, sweater,
> and shorts; colors such as black, blue, and brown; patterns such as checkered
> and leopard print. This user dislikes ... orange, green, and purple ..."

그리고 옵션의 정답 속성은 **동일한 어휘**(`configs/config.py`의 13색/6패턴/23의류)로
정의된다. 즉 모델이 하는 일은:

1. 이미지에서 색/패턴/의류를 인식하고,
2. 프롬프트에 **문자열로 이미 주어진** 리스트와 대조하고,
3. 프롬프트에 **절차까지 주어진** 규칙(`EVAL_PRIORITY_CLAUSE_SITUATION` →
   `..._PREFERENCE`)을 적용하는 것.

이건 personalization이 아니라 **fine-grained attribute recognition +
explicit constraint satisfaction**이다. 리뷰어는 이렇게 쓴다:

> "The user profile is provided verbatim as a list of attribute values drawn from
> the same closed vocabulary used to define the ground-truth options. The task
> therefore reduces to attribute recognition plus string matching against an
> explicitly stated rule. It is unclear what this measures beyond fine-grained
> perception, and the paper's framing as 'personalization' is not supported."

### 문헌상의 위치

- **MMPB** (NeurIPS 2025 D&B): 111개 personalizable concept, 10k image-query
  pair, **concept injection → multi-turn dialogue → personalized query**의 3단계
  프로토콜. 즉 선호/개념을 *주입한 뒤 잊지 않는가*를 본다.
- **Yo'LLaVA / MyVLM**: 개념당 4–17장의 **예시 이미지**로 개인화 개념을 학습.
- **PrefEval** (ICLR 2025 Oral): 3,000개 선호–질의 쌍, **explicit vs implicit**
  선호 표현, 선호와 질의 사이의 context gap을 3k~100k 토큰으로 변화. 핵심 발견:
  **10턴(~3k 토큰)만 지나도 선호 준수 정확도가 10% 아래로 떨어진다.**
- **PerVL-Bench** (WACV 2026): 같은 학회의 직전 연도 personalization 벤치마크.
  **반드시 related work에 넣고 차별점을 한 문단으로 써야 한다.**
- **RealPref / PersonaMem**: 100 프로필 × 1,300 선호, explicit→implicit 4단계
  표현, long-horizon 히스토리.

이 흐름에서 POD-Bench의 프로필은 **가장 explicit한 극단**에 있다. 그 자체는
문제가 아니지만, 그 사실을 논문이 스스로 말하지 않으면 리뷰어가 대신 말한다.

### 보완 (비용 순)

**(a) [무료, 최우선] Profile-swap counterfactual — 이 논문의 킬러 지표가 될 수 있다.**

동일한 4개 이미지(A/B/C/D)를 고정하고, 프로필만 **A값과 B값의 선호가 뒤집힌**
다른 유저의 것으로 교체한다. 진짜로 프로필을 읽는 모델이라면 답이 A → B로
뒤집혀야 한다.

- 지표: `flip rate` = P(answer changes in the predicted direction | profile swapped)
- 이건 데이터 재생성이 필요 없다. `option_plans.jsonl`의 active 값 (a, b)에 대해
  `b`를 좋아하고 `a`를 싫어하는 유저를 프로필 집합에서 찾기만 하면 된다
  (13색 3+3 배정이라 그런 쌍은 충분히 존재한다).
- **왜 강한가**: 현재의 "profile-mode = no / narrative / all" 3조건 비교는
  *정확도 차이*만 보여주지만, flip rate는 **인과적 민감도**를 보여준다.
  높은 strict accuracy + 낮은 flip rate = "프로필을 읽는 척만 한다"의 직접 증거.
  MMPB/PrefEval 어느 쪽에도 이 형태의 통제가 없다 — **novelty를 여기서 벌 수 있다.**

**(b) [중간] Implicit-profile 조건 추가.** 속성 리스트 대신
"이 유저가 최근 구매한 아이템 5개"를 **이미지 또는 상품 제목**으로 제시하고
선호를 추론하게 한다. `annotation/attribute_library.json`에 이미 셀별 이미지가
있으므로, 유저의 liked 셀에서 k장을 뽑으면 끝이다. PrefEval의 explicit/implicit
축을 시각 도메인으로 옮긴 것이 되고, 이게 "personalization 벤치마크"라는
주장을 실질적으로 뒷받침한다.

**(c) [중간] Preference persistence.** 프로필을 프롬프트 맨 앞에 두고 무관한
대화 N턴을 삽입한 뒤 질의. PrefEval의 핵심 발견(10턴 만에 붕괴)이 **시각
선택지 상황에서도 재현되는가**는 그 자체로 논문 한 절짜리 결과다.

**(d) [서술만] 현재 조건을 "explicit-preference upper bound"로 정직하게 명명.**
"프로필이 명시적으로 주어진 조건에서도 모델이 X% 밖에 못 한다"는 프레이밍은
방어 가능하다. 숨기면 공격당하고, 먼저 말하면 강점이 된다.

---

## 3. Major concern #2 — 결과가 없다

### 현황 (코드 조사 결과)

- `data_wacv_scenario_v5/`에는 `profiles/`, `queries/`, `options/`만 있다.
  **이미지 폴더도, eval 결과도 없다.**
- `docs/PROCESS.md` §7 "Remaining" 3–8번이 전부 미완: 139개 coat 셀 재수집,
  27개 `solid` 셀 미주석, plan_id manifest 미동결, 이미지 미조립,
  preflight 미실행, eval 미실행.
- README가 스스로 명시: "These are still **pre-retrieval** numbers... Recompute
  the tables on the frozen `plan_id` manifest before quoting them as final."

### 리뷰어의 말

> "The paper reports no model results. A benchmark without baselines cannot be
> assessed for difficulty, headroom, or discriminative power."

### 최소 요구 세트 (이게 없으면 어떤 리뷰어도 accept 안 한다)

| # | 실험 | 왜 필요한가 |
|---|---|---|
| 1 | **≥8개 VLM** (closed 2–3: GPT-4o/5급, Gemini, Claude / open 5–6: Qwen3-VL, InternVL3.5, LLaVA-OV, Molmo, Llama-3.2-V) × 2 트랙 × 3 profile-mode | 벤치마크가 모델을 *구분*하는지 |
| 2 | **인간 정답률** (층화 200–300문항) | "인간도 못 푸는 벤치" 방어. `redesign_v2_plan.md` §8-3에 이미 계획됨 |
| 3 | **Random / blind / text-only 기준선 4종** (§4) | 시각 필수성 |
| 4 | **모델 크기 스케일링** (같은 계열 3–4 사이즈) | headroom이 스케일로 닫히는지 |
| 5 | **에러 4-way 분포 (A/B/C/D)** | §5 참조 |

**우선순위 조언**: 모든 셀을 다 모을 때까지 기다리지 말고,
**동결 가능한 부분집합(예: image-complete 1,933 plan)으로 먼저 전 실험을 돌려라.**
139개 coat 셀 재수집은 병렬로 진행하되 blocking 사유로 두지 마라. 지금 구조상
"완벽한 manifest를 기다리다 마감을 놓치는" 리스크가 실제로 가장 크다.

---

## 4. Major concern #3 — 시각 필수성(vision-essentiality)이 측정되지 않았다

### 지적

README의 파이프라인에 "Stage 6: Quality Audit (assembly + **vision-essentiality
gate**)"가 있지만, 코드에 구현이 없다 (`grep -rn "essential" scripts/` → 해당
로직 없음). `text_only_eval.py`가 있지만 이건 **속성을 텍스트로 주는** 조건이므로
"시각이 필요한가"의 반대 방향 통제다 — 오히려 **상한(oracle perception)** 이다.

이 벤치마크는 특히 위험하다. 이유:
- `search_query`가 `"blue hoodie"` 같은 형태로 plan에 그대로 들어있다.
- 물리 트랙은 violation axis가 **garment 100%** → "폭설에 드레스 vs 후디"는
  상식 질문에 가깝고, 이미지 없이도 상당 부분 풀린다.

### 문헌

- **NaturalBench** (NeurIPS 2024): MME 등 기존 벤치가 blind QA 모델로 풀린다는
  점을 지적하고, 한 질문에 서로 다른 답을 갖는 **이미지 쌍**을 붙여 shortcut을
  차단. (10,000 human-verified samples)
- **MMEvalPro**: MMMU에서 최고 LMM이 최고 LLM 대비 1.1배에 불과 — 즉 시각
  입력이 거의 불필요한 문항이 다수.
- Composed image retrieval 벤치 감사 연구: CIRR 질의의 **83.6%가 unimodal로
  풀린다**.

### 필요한 기준선 4종 (전부 저렴)

| 기준선 | 입력 | 측정 대상 |
|---|---|---|
| B0 Random | — | 25% |
| B1 **Blind LLM** | query + profile, 옵션은 **레이블만** (A/B/C/D) | 순수 사전확률 |
| B2 **Text-attribute** (= 현 `text_only_eval`) | GT 속성 텍스트 | **지각 없는 추론 상한** |
| B3 **Caption-only** | 옵션 이미지의 VLM 캡션 텍스트 | 지각 손실의 크기 |
| B4 Multimodal | 실제 이미지 | 본 실험 |

**보고할 핵심 수치: `B2 − B4` (지각 격차)와 `B4 − B1` (시각 기여).**
`B2 − B4`가 크면 "이 벤치는 지각 문제다", 작으면 "추론 문제다" — 어느 쪽이든
논문의 claim이 명확해진다. 현재는 이 문장을 쓸 근거가 없다.

추가로 **문항 단위 vision-essentiality gate**: B2에서 맞고 B4에서 틀린 문항,
B1에서 이미 맞는 문항을 태깅해 `plan_id` 단위 메타로 저장하면, "시각이 실제로
필요한 부분집합"에서의 정확도를 별도 보고할 수 있다. NaturalBench식 방어가 된다.

---

## 5. Major concern #4 — 세 지표(strict/TPO/preference)는 수학적으로 종속이다

### 지적

`scripts/multimodal_eval.py`:
```
TPO_SCORE     = {"A": 1, "B": 1, "C": 0, "D": 0}
PROFILE_SCORE = {"A": 1, "B": 0, "C": 1, "D": 0}
```

4지선다 강제 선택이므로 P(A)+P(B)+P(C)+P(D)=1이고,

- strict = P(A)
- TPO = P(A)+P(B)
- pref = P(A)+P(C)
- 따라서 **TPO + pref = 1 + strict − P(D)**

세 지표는 4-way 분포(자유도 3)와 **정보량이 완전히 동일**하다. 즉 "TPO 능력"과
"선호 능력"이 독립적으로 측정되는 것처럼 제시하면 리뷰어가 즉시 지적한다.
("TPO accuracy 90%, preference accuracy 60%"는 두 능력이 아니라 하나의
선택 분포를 두 번 말한 것.)

### 보완 (전부 무료, 데이터 재생성 불필요)

1. **주 표는 4-way 분포 (A/B/C/D %)로 보고하라.** 해석력이 훨씬 높다.
2. **Priority index**: 오답 중 B:C 비율. `B/(B+C)` → 모델이 상황을 우선하는가
   취향을 우선하는가. `redesign_v2_plan.md` §8-2가 이미 "공짜 지표"라고 지목한
   바로 그것인데 아직 구현되지 않았다. **이게 이 벤치마크의 가장 차별적인
   결과가 될 가능성이 높다** — 기존 personalization 벤치는 "선호를 따르는가"만
   묻지, "따르지 말아야 할 때 안 따르는가"를 묻지 않는다.
3. **D-rate를 별도 보고**: D는 "둘 다 틀림" = 지각 실패의 대리 지표.
4. **Macro 평균**: 현재 리포트는 `axis:` / `qtype:` 브레이크다운만 있다.
   archetype·scenario·user 단위 macro도 추가해야 "garment만 잘해도 고득점"
   방어가 된다 (§8-1의 계획, 미구현).

---

## 6. Major concern #5 — 평가 프롬프트가 정답 절차를 알려준다

`configs/scenarios.py`:
```python
EVAL_PRIORITY_CLAUSE_SITUATION = 'First eliminate any option that is inappropriate for the stated situation.'
EVAL_PRIORITY_CLAUSE_PREFERENCE = 'Among the remaining situation-appropriate options, choose the one that best matches the user's stated preferences.'
```

이건 "상황과 취향을 **동시에 고려**할 수 있는가"라는 연구 질문의 답을
프롬프트가 미리 알려주는 것이다. 리뷰어:

> "The system prompt states the exact decision procedure the benchmark claims to
> measure. Reported accuracy therefore conflates instruction following with the
> ability to arrive at that priority ordering."

**보완 (무료)**: priority clause on/off ablation을 필수 실험으로 추가.
- clause **off**가 본 실험(더 어렵고 더 흥미롭다), clause **on**은
  "절차를 알려줘도 못 한다"를 보이는 상한 조건.
- 동시에 `EVAL_FRAME_CLAUSE` (US convention) on/off/다른 문화 프레임 ablation도
  같이 돌려라 → §9의 문화 타당성 방어에 그대로 재사용된다.
- 부수 효과: prompt sensitivity 분석은 요즘 데이터셋 논문의 사실상 필수 항목이다.

---

## 7. Major concern #6 — 통계적 처리가 전혀 없다

### 지적 (코드 확인)

`grep -rn "bootstrap|confidence interval|significan|mcnemar" scripts/ construction/ tests/`
→ **0건.** 리포트는 전부 점추정 퍼센트다.

이 데이터는 특히 비독립적이다:
- 2,641 plan이 2,270 query context에서 나온다 (한 query가 여러 violation-axis
  variant를 낳음).
- 24명 유저 × 59 시나리오 = **1,416 셀**이 실질적 설계 단위인데 item은 2,641개.
- `redesign_v2_plan.md` §8-5가 "쌍둥이 문제는 클러스터 부트스트랩으로 처리"라고
  이미 적어두었으나 미구현.

### 보완

1. **Cluster bootstrap** (클러스터 = `query_id`, 추가로 `user_id`, `scenario_id`
   두 수준 각각) → 95% CI. 모델 간 비교는 **paired** bootstrap 또는 McNemar.
2. **유효 표본 크기(ESS)를 명시**: 2,641이라는 숫자를 그대로 쓰면
   "24명 유저에서 나온 2,641 문항"이라는 반박을 받는다. 먼저 말하라.
3. **시드 안정성**: `redesign_v2_plan.md` §8-7 "시드 3개로 재생성해 모델 순위가
   안정적인가". 지금은 seed 42 하나뿐. **최소한 seed 2개 더** 생성해
   *모델 랭킹의 Kendall τ*를 보고하면, "이 벤치는 특정 난수 배정의 산물"
   공격을 완전히 막는다. 이건 GPU 없이 되는 작업이라 비용 대비 효과가 가장 크다.

---

## 8. Major concern #7 — MCQ 편향 통제가 shuffle 하나뿐

### 현황

`multimodal_eval.py`는 per-item random shuffle + `--fix-correct` 실험 +
position histogram을 제공한다. 이건 평균은 잡지만 **문항 단위 일관성**은
측정하지 않는다.

### 문헌

- **MMBench**의 **CircularEval**: 선택지를 N번 순환 이동시켜 **모든 회전에서
  맞아야** 정답 처리. label bias와 요행을 동시에 제거.
- **Benchmarking and Mitigating MCQA Selection Bias of LVLMs** (arXiv 2509.16805):
  LVLM이 특정 옵션 토큰/위치를 선호한다는 체계적 증거와 logit-level debiasing.

### 보완

1. **CircularEval을 주 지표로 추가**(4 rotation × 전체 = 4배 비용이므로,
   전체가 부담되면 층화 부분집합 500문항으로).
   4지선다에서 CircularEval strict의 우연 수준은 사실상 0에 가까워
   **난이도 headroom을 크게 벌어준다** — 지금처럼 물리 트랙이 saturate할
   위험이 있는 상황에서 특히 유용하다.
2. **Consistency 지표** 자체를 보고: "4회전 중 몇 회전에서 A를 골랐나"의 분포.
   이건 personalization 안정성과도 직결된다.
3. `--guided` 기본 True(A–D 강제)인데, guided/unguided 차이도 한 줄 보고하면
   파싱 아티팩트 논란을 차단한다.

---

## 9. Major concern #8 — Dress-code 라벨의 인간 검증이 아직 0건

### 지적

README가 스스로 명시: "The convention-based half is **not yet human-validated**
— annotation is planned (§4); until it lands, treat those labels as authored,
not verified."

리뷰어는 이 문장을 **그대로 인용해서** 공격한다. 35개 dress-code 시나리오
(전체의 59%)의 정답이 저자 판단이라는 뜻이기 때문이다. `EVAL_FRAME_CLAUSE`로
"mainstream contemporary US"라고 범위를 좁힌 것은 **옳은 조치이고 반드시
강조해야 하지만**, 범위를 좁힌 것과 라벨이 맞다는 것은 다른 명제다.

### 문헌

- **CulturalVQA** (EMNLP 2024): 11개국 2,378 image-question pair. VLM의 문화
  이해도가 지역별로 크게 갈리며(북미 강세, 아프리카 약세), 특히 **clothing**이
  주요 facet 중 하나다. → "US 규범만 다룬다"는 스코프 선언은 이 문헌을 인용해
  **의도적 설계 결정**으로 방어하라.
- "No Filter: Cultural and Socioeconomic Diversity in Contrastive VLMs":
  retrieval corpus 자체의 문화·소득 편향. Amazon 651k corpus를 쓰는 이상
  이 인용은 limitations에 있어야 한다.

### 보완 (비용 대비 효과 최상)

1. **라벨 검증 인간 스터디** — 층화 표본 150–200개 (시나리오 × archetype 균형),
   **응답자 5–7명**, 질문은 단순하게:
   "이 상황에서 X를 입는 것이 부적절한가? (5점 척도)"
   - 보고: **Krippendorff's α 또는 Fleiss' κ**, archetype별 α,
     "저자 라벨과 다수 의견의 일치율".
   - α가 낮게 나오는 archetype이 있으면 **그것도 결과다** — 제거하지 말고
     "convention strength" 라벨로 계층화해서 보고하면 오히려 기여가 된다.
     (예: mourning/ultra_formal은 α 높음, semi_formal_social은 낮음)
2. **US vs non-US 응답자 분리** (각 최소 20명). "문화 프레임을 명시하면 격차가
   줄어드는가"는 §6의 frame-clause ablation과 짝을 이루어
   **논문 한 절짜리 독립 기여**가 된다.
3. **인간 정답률(ceiling)** 은 위 스터디와 **같은 세션에서 동시에** 받아라.
   문항을 그대로 4지선다로 풀게 하고, 프로필도 같이 제시. 200–300문항이면 충분.

---

## 10. Major concern #9 — 이미지 레벨 교란이 미통제 (README가 스스로 자백)

README §"Image collection" 중:

> "Gates are fail-open. *Planned, not implemented in the current collector:*
> **gender consistency** and **intra-set URL dedup** within each 4-option set."
> "*Planned, not implemented:* a decision-aware pattern-FAMILY hit rule"

리뷰어는 이 문장을 인용하고 끝낸다. 특히 **gender consistency 미구현**은
치명적이다: A/B는 여성 모델, C/D는 남성 모델이면 정답이 **의류 판단 없이도**
시각적으로 분리된다.

### 반드시 측정해야 할 "옵션 세트 내 저수준 교란" 감사

각 plan의 4장에 대해 자동 측정 가능한 것들 (전부 저렴, GPU 조금):

| 교란 | 측정 방법 | 위험 |
|---|---|---|
| 착용 성별 불일치 | VLM 이진 분류 or 사람 검출 + 분류 | **최상** |
| 사람 착용 vs flat-lay 혼재 | 사람 검출기 | **최상** (스타일 단서) |
| 배경 유형(흰 배경/야외) 불일치 | 배경 색 분산, 클러스터링 | 상 |
| 이미지 해상도/화질 차이 | 해상도, BRISQUE | 중 |
| 워터마크/로고/텍스트 | OCR 검출 | 중 |
| 동일 URL/근접 중복 | perceptual hash | 상 |
| 모델 포즈/크롭 스케일 | 사람 bbox 면적 비 | 중 |

보고 형태: **"4옵션이 모두 동일 카테고리인 plan의 비율"** 표 + 불일치 plan에서의
정확도 vs 일치 plan에서의 정확도 (차이가 없으면 그 자체가 방어 증거).
이건 "우리는 봤고, 영향이 없음을 보였다"라는 가장 강한 형태의 답변이다.

추가로: **CLIP/SigLIP shortcut probe** — 프로필 없이 query 텍스트와 4장의
이미지 유사도만으로 정답을 고르는 non-VLM 기준선. 이게 우연보다 유의하게 높으면
retrieval-side leakage가 있다는 뜻이다 (`search_query`가 plan에 그대로 있으므로
실제 위험이 있다).

---

## 11. Major concern #10 — 지각 상한(perception ceiling)이 측정되지 않았다

이건 §4(B2−B4)의 문항 단위 버전이며, **가장 저평가된 기회**다.

### 제안: per-option attribute probe

각 옵션 이미지를 **단독으로** VLM에 주고 폐쇄 어휘에서 색/패턴/의류를 맞히게
한다 (`13-way`, `6-way`, `23-way`).

얻는 것:

1. **셀 단위 지각 난이도 맵** — 어떤 `color|garment|pattern` 셀이 모델에게
   실제로 안 보이는가.
2. **`CONFUSABLE_ACTIVE_PAIRS` / `CONFUSABLE_GARMENT_PAIRS`의 경험적 검증.**
   현재 이 표는 `construction/option_planner.py`에 **손으로 적힌 저자 판단**이다
   (black↔navy, jeans↔slacks 등). 리뷰어는 "누가 정했나"를 묻는다.
   confusion matrix로 대체하면 **저자 판단 → 측정값**이 되어 반박 불가능해진다.
   현재 `"pattern": set()` (패턴은 혼동쌍 없음)으로 되어 있는데, 실제로는
   striped↔checkered, floral↔polka_dot이 혼동될 가능성이 높다 — 측정하면
   바로 드러난다.
3. **오류 분해**: 문항 오답 = 지각 실패인가 추론 실패인가.
   "A의 색을 못 맞힌 문항에서만 오답률이 높다"면 그건 지각 벤치마크다.
4. **Fashionpedia (ECCV 2020, 27 category / 294 attribute) 어휘와의 매핑표**를
   부록에 넣으면 "왜 23개인가"에 대한 외부 근거가 생긴다. 현재 어휘 선정 근거는
   `configs/config.py` 주석의 내부 논리뿐이다.

비용: 이미지 수 × 3 질문. 2,641 plan × 4 옵션 = 약 10.5k 이미지지만
**셀 단위로 중복 제거하면 1,237개 셀 × 3 = 3.7k 호출**이면 끝난다. 매우 싸다.

---

## 12. Minor / 사실관계 오류 — 지금 바로 고칠 것

### 12-1. 스케일 비교표가 틀렸다 (README 마지막 bullet)

> "Comparable in scale to: MMPB (~500), NaturalBench (~900), BLINK (~3.8k)"

- **MMPB는 10k image-query pair / 111 concept**이다. ~500이 아니다.
- **NaturalBench는 10,000 human-verified VQA samples**다. ~900이 아니다.
- BLINK ~3.8k(3,807 MCQ)는 맞다.

리뷰어가 이걸 잡으면 **다른 모든 숫자의 신뢰도까지 떨어진다** (이 논문의
최대 강점이 "숫자를 정직하게 센다"인데 그걸 스스로 무너뜨림). 수정하거나,
정말 다른 축(예: "human-verified concept 수", "고유 이미지 수")을 비교한
것이라면 축을 명시하라. 어느 쪽이든 **2,641은 절대 규모로는 작다**는 사실을
인정하고, 대신 "설계된 factorial 구조 + 셀 단위 인간 검증"으로 프레이밍하는
것이 훨씬 안전하다.

### 12-2. "Outfit"이라는 명칭

각 옵션은 **단일 의류 아이템 이미지**(상의 하나 또는 하의 하나)다. "Outfit
Decision"이라는 이름은 코디네이션(상하의 조합, 레이어링)을 기대하게 한다.
리뷰어는 "폭설에 dress가 오답이라지만 코트를 겹쳐 입으면 되지 않나"라고 묻는다.

**보완**:
- 논문에서 task를 "single-garment selection under situational and preference
  constraints"로 정확히 정의하고, 프롬프트에도 "이 아이템 하나만 고려한다"는
  문장을 넣어라 (현재 시스템 프롬프트에 없다).
- 또는 이름을 바꿔라 (`Personalized Garment Decision`). 이름 하나로 리뷰어
  질문 하나가 사라진다.

### 12-3. 유저에게 성별 개념이 없는데 어휘에는 `mini_skirt`/`dress`/`suit_vest`가 섞여 있다

같은 U001이 어떤 문항에서는 dress를, 다른 문항에서는 suit_vest를 제시받는다.
이건 설계상 의도(성별 중립)일 수 있지만, **검색된 이미지에는 성별이 있다.**
§10의 gender consistency와 합쳐서 limitations에 한 문단으로 명시하라.

### 12-4. `--solid-baseline` 순서 버그 (58 plan)

`wacv_scenario_v5_report.md`가 이미 정확히 진단했다 (availability check가
override 전 배경 패턴으로 실행됨). 문서화가 훌륭하므로 그대로 두되,
**논문 본문에는 넣지 마라** — 부록/릴리즈 노트 사항이다. 본문에 있으면
"버그가 있는 데이터"로 읽힌다.

### 12-5. 물리 트랙의 saturate 위험

물리 트랙은 violation axis가 garment 100%이고 대비가 "폭설 vs 탱크톱" 수준이라
최신 모델에서 95%+ 로 포화될 가능성이 높다. 그러면 "physical track은 쉽다"는
한 줄로 끝나고 dress-code만 남는다.

**대비책**: 물리 트랙 안에서 **난이도 계층**을 만들어라.
- easy: 극단 대비 (blizzard × tank_top)
- hard: 근접 대비 (blizzard × windbreaker vs puffer_jacket — 둘 다 아우터지만
  단열성이 다름)
현재 `compatible`/`incompatible` 이분법은 hard 층을 만들 수 없다.
**"marginal" 3번째 등급**을 도입하면 물리 트랙이 살아난다. 데이터 재생성이
필요하지만, 이게 physical 트랙을 논문에 남길 유일한 길일 수 있다.

### 12-6. 데이터셋 문서화 (WACV 요구사항)

- Amazon fashion 651k corpus의 **출처·라이선스·재배포 가능 여부**가 어디에도
  없다. 이미지를 배포하지 못하면 벤치마크로서의 가치가 급락하므로,
  URL + 속성 라벨만 배포하는 형태라도 **지금 결정하고 명시**해야 한다.
- **Datasheet for Datasets** 스타일 부록 (수집 방법, 주석자 수/보상/지시문,
  의도된 용도, 알려진 편향) — 요즘 D&B 트랙에서 사실상 필수.
- 주석자 정보가 `docs/PROCESS.md`에 없다: 몇 명이 1,661셀을 봤는가?
  **1명이라면 반드시 명시하고**, 부분 재주석으로 intra/inter-annotator
  agreement를 내야 한다 (예: 무작위 150셀을 2주 뒤 재주석 → intra-rater κ).
  현재 §4는 "human annotation"이라고만 쓰여 있어 리뷰어가 반드시 묻는다.

---

## 13. 추가하면 "강점"이 되는 수치 (우선순위 정렬)

비용 대비 논문 임팩트 기준. **★ = 지금 당장, GPU 거의 없이 가능**.

| # | 수치 | 비용 | 임팩트 | 방어하는 공격 |
|---|---|---|---|---|
| 1 ★ | **Profile-swap flip rate** (§2a) | 낮음 | **최상** | "personalization이 아니다" |
| 2 ★ | **오답의 B:C 비율 = priority index** (§5) | 무료 | **최상** | "지표가 중복이다" / novelty |
| 3 ★ | **Cluster bootstrap 95% CI** (§7) | 무료 | 상 | "통계가 없다" |
| 4 ★ | **Seed 3개 재생성 → 모델 랭킹 Kendall τ** (§7-3) | 낮음 (CPU) | 상 | "난수 배정의 산물이다" |
| 5 | **기준선 4종 B0–B4** (§4) | 중 | **최상** | "시각이 필요 없다" |
| 6 | **인간 정답률 + IAA (α)** (§9) | 중 (사람) | **최상** | "라벨이 저자 취향이다" |
| 7 | **Per-option attribute probe → confusion matrix** (§11) | 낮음 | 상 | "혼동쌍 표가 임의적이다" |
| 8 | **4옵션 세트 내 저수준 교란 감사** (§10) | 낮음 | 상 | "이미지 단서로 풀린다" |
| 9 | **Priority clause / frame clause ablation** (§6) | 낮음 | 상 | "프롬프트가 답을 준다" |
| 10 | **CircularEval (부분집합)** (§8) | 중 (4×) | 중상 | "위치 편향" + headroom |
| 11 | **Implicit-profile 조건** (§2b) | 중 | **최상** | novelty 확보 |
| 12 | **US vs non-US 응답자 격차** (§9-2) | 중 (사람) | 상 | 문화 타당성 → 독립 기여 |
| 13 | **모델 크기 스케일링 곡선** | 중 | 중 | "곧 포화된다" |
| 14 | **Preference persistence (N턴 삽입)** (§2c) | 중 | 상 | PrefEval과의 연결 |
| 15 | **물리 트랙 marginal 등급 도입** (§12-5) | 높음 (재생성) | 중상 | "physical은 쉽다" |

**마감이 촉박하다면 1–4번(전부 ★)만 해도 리뷰 톤이 크게 바뀐다.**
GPU 없이, 데이터 재생성 없이, 기존 아티팩트만으로 가능하다.

---

## 14. 논문 서술 조언

### 14-1. Contribution 문장을 이렇게 재배치하라

현재 구성 문서의 강점 순서와 논문에서 팔아야 할 순서가 다르다.

1. **선호와 상황이 *충돌*하도록 강제 설계된 첫 VLM 벤치마크.**
   (기존 personalization 벤치는 "선호를 따르는가"만 묻는다. 이 벤치는
   "따르면 안 될 때 안 따르는가"까지 묻는다 — 이게 진짜 novelty다.)
2. **2×2 factorial 설계로 오답이 원인별로 해석된다** (B=취향 무시,
   C=상황 무시, D=지각 실패).
3. **retrieval 가용성을 생성 제약으로 올린 구성 방법론** (v4→v5, §1-1).
4. 트랙 분리 + 문화 프레임 명시.
5. 재현성 (SHA256, mutation test, verify_release.sh).

### 14-2. Related work에 반드시 들어가야 할 표

| 벤치마크 | 개인화 표현 | 상황 제약 | 충돌 유도 | 모달리티 |
|---|---|---|---|---|
| MyVLM / Yo'LLaVA | 개념 예시 이미지 | ✗ | ✗ | V+L |
| MMPB | 개념 주입 + 다중턴 | ✗ | ✗ | V+L |
| PerVL-Bench (WACV'26) | (확인 필요) | ✗ | ✗ | V+L |
| PrefEval (ICLR'25) | 명시/암시 선호, long context | ✗ | ✗ | L |
| RealPref / PersonaMem | 장기 상호작용 이력 | ✗ | ✗ | L |
| CulturalVQA | ✗ | 문화 지식 | ✗ | V+L |
| **POD-Bench (ours)** | **명시 속성 프로필** | **TPO 규범** | **✓ (2×2)** | **V+L** |

이 표의 "충돌 유도 ✓" 열이 비어 있는 것이 이 논문의 자리다.
**단, "개인화 표현" 열이 가장 약한 형태라는 것도 같은 표에 드러난다** —
그래서 §2의 implicit-profile 조건이 novelty 방어에 필요하다.

### 14-3. Limitations 절은 이미 쓰여 있다

`wacv_scenario_v5_report.md`의 "Limitations that cannot be fixed by balancing"
4항목 + 70 plan 분해를 거의 그대로 논문 limitations로 옮겨라.
다만 **"왜 이게 도메인의 속성이지 데이터의 결함이 아닌가"** 문장을 각 항목에
한 줄씩 붙여라 (이미 2번 항목에는 있다 — "formal dress codes admit essentially
one pattern... a property of the domain rather than an artifact of the data").
그 문장이 있는 항목과 없는 항목의 인상 차이가 크다.

---

## 15. 제출 전 체크리스트

**Blocking (없으면 reject)**
- [ ] 이미지 폴더 조립 + `plan_id` manifest 동결
- [ ] manifest 위에서 모든 balance/confound 리포트 **재계산** (2,641-plan
      construction 리포트를 그대로 옮기지 말 것 — PROCESS.md §7-5가 이미 경고)
- [ ] ≥8 모델 × 2 트랙 × 3 profile-mode
- [ ] 기준선 B0–B4
- [ ] 인간 정답률 (층화 200–300)
- [ ] 라벨 검증 IAA (dress-code 150–200문항, 5–7명)
- [ ] 통계: cluster bootstrap CI

**High value (있으면 accept 쪽으로 이동)**
- [ ] Profile-swap flip rate
- [ ] Priority index (B:C)
- [ ] Seed 3개 랭킹 안정성
- [ ] 4옵션 세트 저수준 교란 감사 (특히 gender)
- [ ] Per-option attribute probe / confusion matrix
- [ ] Priority + frame clause ablation

**Fix now (문서 수정만)**
- [ ] MMPB / NaturalBench 규모 수치 정정 (§12-1)
- [ ] "outfit" → 단일 아이템 task 정의 명확화 + 프롬프트 문장 추가 (§12-2)
- [ ] 주석자 수/절차/보상 명시, intra-rater 재현성 (§12-6)
- [ ] corpus 라이선스·배포 형태 결정 및 명시 (§12-6)
- [ ] Datasheet 부록

---

## 16. 참고 문헌

**Personalization 벤치마크**
- MMPB: It's Time for Multi-Modal Personalization (NeurIPS 2025 D&B) — https://arxiv.org/abs/2509.22820
- Yo'LLaVA: Your Personalized Language and Vision Assistant — https://arxiv.org/html/2406.09400
- MC-LLaVA: Multi-Concept Personalized Vision-Language Model — https://arxiv.org/pdf/2411.11706
- PerVL-Bench: Benchmarking Multimodal Personalization for Large Vision-Language Models (WACV 2026) — https://openaccess.thecvf.com/content/WACV2026/papers/Kim_PerVL-Bench_Benchmarking_Multimodal_Personalization_for_Large_Vision-Language_Models_WACV_2026_paper.pdf
- PrefEval: Do LLMs Recognize Your Preferences? (ICLR 2025 Oral) — https://arxiv.org/abs/2502.09597
- Towards Realistic Personalization: Long-Horizon Preference Following (RealPref) — https://arxiv.org/html/2603.04191v1
- Know Me, Respond to Me (PersonaMem) — https://arxiv.org/html/2504.14225v2
- Can LLMs Understand Preferences in Personalized Recommendation? — https://arxiv.org/pdf/2501.13391
- RAP: Retrieval-Augmented Personalization for MLLMs — https://arxiv.org/pdf/2410.13360

**벤치마크 설계 / shortcut / 편향**
- NaturalBench: Evaluating VLMs on Natural Adversarial Samples (NeurIPS 2024) — https://arxiv.org/abs/2410.14669
- MMBench: Is Your Multi-modal Model an All-around Player? (ECCV 2024, CircularEval) — https://arxiv.org/pdf/2307.06281
- Benchmarking and Mitigating MCQA Selection Bias of LVLMs — https://arxiv.org/abs/2509.16805
- MMEvalPro: Calibrating Multimodal Benchmarks Towards Trustworthy Evaluation — https://arxiv.org/pdf/2407.00468
- Do Composed Image Retrieval Benchmarks Require Multimodal Composition? — https://arxiv.org/pdf/2605.14787
- A Survey on Benchmarks of Multimodal Large Language Models — https://arxiv.org/pdf/2408.08632

**문화 타당성**
- Benchmarking Vision Language Models for Cultural Understanding / CulturalVQA (EMNLP 2024) — https://aclanthology.org/2024.emnlp-main.329/
- No Filter: Cultural and Socioeconomic Diversity in Contrastive Vision-Language Models — https://arxiv.org/html/2405.13777v3

**패션 도메인**
- Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset (ECCV 2020) — https://arxiv.org/pdf/2004.12276
- DeepFashion2 — https://arxiv.org/pdf/1901.07973
- Occasion and Color-Aware Personalized Outfit Recommendation System — https://link.springer.com/chapter/10.1007/978-3-032-07837-7_21
- TATTOO: Training-free AesTheTic-aware Outfit RecOmmendation — https://arxiv.org/html/2509.23242
- Personalised Outfit Recommendation via History-aware Transformers — https://arxiv.org/pdf/2407.00289
- Preliminary Study of an Evaluation Benchmark for VLMs in Fashion E-Commerce (SIGIR 2026) — https://doi.org/10.1145/3805712.3808442
- Can GPT-4o mini and Gemini 2.0 Flash Predict Fine-Grained Fashion Product Attributes? — https://arxiv.org/pdf/2507.09950

---

## 17. 다음 리뷰 라운드에서 볼 것

이번 라운드는 구성(construction)과 문서에 집중했다. 다음에 볼 것:

1. `construction/option_planner.py`의 목적함수 가중치
   (`CONFUSABLE_PAIR_PENALTY = 10_000`)가 counterbalance를 얼마나 밀어내는지 —
   가중치 sensitivity가 데이터 분포를 바꾸는지.
2. `construction/compatibility.py`의 relaxed compatibility check가
   physical 트랙에서 "색은 자유"로 두는 것의 부작용
   (예: 야간 러닝에서 색이 preference axis일 때 안전 규범과 충돌하지 않는지).
3. `fsiglip/collector_sam3.py`의 pattern patch 분류 임계값
   (`attr_min_score=0.2`)이 패턴별로 다르게 작동하는지 — leopard 57% /
   polka_dot 60% 가용성이 실제 코퍼스 희소성인지 임계값 아티팩트인지.
4. `annotation/serve_annotator.py`의 `show_n=10` 상위 후보 제시가
   selection bias를 만드는지 (screen-rank top-1 일치율 32%는 주석자가
   스코어러와 독립적으로 판단했다는 좋은 신호지만, top-10 밖은 아예 못 봤다).
