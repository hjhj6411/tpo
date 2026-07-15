# POD-Bench 재설계 v2 통합 계획서

> 2026-07-15 세션 통합 정리. 목적: WACV 제출 전 벤치마크 구조 개편의 전체 논리를
> 한눈에 검토할 수 있게 문제 → 원리 → 규칙 → 검증 → 실행 순으로 배열한다.
> 각 설계 요소가 어떤 공격(W#)을 막는지 추적 가능하도록 표기한다.

---

## 0. 요약 (TL;DR)

- **문제 정의를 이원화**한다: Physical(TPO=garment 고정) vs Dress-code(제약 축 회전).
- **프로필을 티어 쿼터 규칙(R1–R4)으로 재생성**한다: 어휘를 ANCHOR/RESERVE/FREE로
  분할하고, 선호 배정을 쿼터로 못 박아 feasibility를 시뮬레이션 결과가 아닌
  **구성적 보장(by construction)**으로 만든다.
- **시나리오 카탈로그를 2×2 격자(제약근원×격식)로 완비**한다: 신규 4 아키타입
  12 시나리오 추가(구현 완료), 보수적 검수 패치 반영(구현 완료).
- 결과(시뮬 v4): 총 문제 상한 2,760 → 4,736, 위반 축 garment 89% → 67%,
  color 9% → 23%, pattern 2% → 10%, per-user 완전 균등.
- 남은 구현: profile_generator R1–R4 이식 → 파이프라인 확장(garment-active·병렬
  variant·violation_garment_scope) → 전체 재생성 → 지표 재검증 → 인간 검증.

---

## 1. 진단: 현재 벤치(2,760문제)의 문제

### 1.1 세션에서 확정된 사실 (실데이터 검증)

| # | 문제 | 수치 | 리뷰어 공격 시나리오 |
|---|---|---|---|
| D1 | 위반(TPO) 축이 garment에 쏠림 | garment 2,450(89%) / color 248 / pattern 62 | "garment 적합성만 봐도 TPO 만점" 지름길 |
| D2 | garment가 선호(active) 축이 된 적 없음 | `ALLOWED_ACTIVE_AXES={color,pattern}` | "3축 프로필을 만들고 2축만 측정" |
| D3 | 제약 여부와 격식이 100% 교락 | 3축 제약 ⟺ formal (28/28), physical ⟺ very_casual | "장면이 캐주얼이면 색·무늬 무시" 휴리스틱 |
| D4 | 밝은 색이 위반 선지에만 등장 | orange/yellow/white는 C/D에서만 출현 | "쨍한 색 = 오답" 통계 지름길 |
| D5 | 회전 상한이 프로필에 막힘 | 회전 가능 328/2,760, 실제 310(상한의 94.5%) | 선택 로직 개선으로는 해결 불가 — 상류 병목 |
| D6 | 유저별 속성 쏠림 재발 위험 | (기존 수정: U001 striped 59/60 등) | per-user 균형이 사후 배분에 의존 |

### 1.2 병목의 3중 게이트 (D5의 해부)

위반 축을 color/pattern으로 회전시키려면:

1. **시나리오 게이트** — 그 축을 시나리오가 제약해야 함
2. **프로필 게이트** — 프로필 선호를 빼고도 compatible/incompatible **양쪽에**
   중립값이 남아야 함 (pattern 어휘 6개 중 프로필이 4개 소진 → 산술적으로 잔인)
3. **active 충돌 게이트** — active 축은 위반 축이 될 수 없음

결론: **선택기(planner)는 이미 상한에 붙어 있으므로, 게이트 자체(카탈로그·프로필)를
여는 재설계가 필요하다.** 이것이 이하 모든 설계의 출발점이다.

---

## 2. 문제 정의 이원화 (논문 프레임)

> **원리: 제약의 본성이 축 구조를 결정한다.**

| | Physical | Dress-code (coded) |
|---|---|---|
| 제약의 본성 | 물리·기능 (방수, 보온, 운동성) | 규범 또는 기능 코드 (격식, 안전 시인성, 촬영 규칙…) |
| 제약 수준 | garment 수준에만 존재 | garment ∪ color ∪ pattern |
| TPO(위반) 축 | **garment 고정** | **제약된 축에서 회전** |
| 선호(active) 축 | color, pattern | **color, pattern, garment (3축)** |
| 시나리오 수 | 32 | 40 (기존 28 + 신규 12) |

- Physical의 garment 100% 위반은 결함이 아니라 **정의의 귀결**로 서술된다.
- Dress-code에서 garment 선호 측정(D2 해소)이 가능해지는 근거: 격식 코드는
  garment 취향과 TPO 판정을 분리할 수 있는 유일한 맥락(ANCHOR 존재) — §3.

---

## 3. 프로필 구축 규칙: 티어 쿼터 (R1–R4)

### 3.1 원리

> **각 축의 어휘를 카탈로그의 제약 구조로부터 자동 분할한다:**
> - **ANCHOR** — 코드형 시나리오에서 (준)보편적으로 compatible한 값 → 선호 측정용
> - **RESERVE(hard)** — 준보편적으로 incompatible로 등장하는 값 → **선호 배정 금지**,
>   TPO 위반 측정 도구로 예약 (신호 채널 분리)
> - **FREE** — TPO 역할이 없는 값 → 페르소나 다양성용
>
> "격식"이 아니라 "제약 구조"가 기준이므로 카탈로그가 바뀌면 티어도 자동 재계산
> — 도메인 중립적 규칙 (general함의 근거).

### 3.2 규칙

| 규칙 | 내용 | 보장하는 것 |
|---|---|---|
| **R1 garment** | likes 3 = ANCHOR {blazer, formal_shirt, dress} 1 + FREE 2; dislikes 동일 구조 (서로소) | 모든 코드형-격식 시나리오에서 garment-active 성립(A·B 존재) + 중립 TPO 쌍 보존 |
| **R2 color** | likes 3 = ANCHOR {black, navy, gray} 1 + FREE {blue, red, green, brown, beige, purple} 2; dislikes 동일. RESERVE {orange, yellow, pink, white} 선호 금지 | 모든 색 제약 시나리오에서 color-active A/B 존재 + **color 위반 회전 유저 무관 성립** |
| **R3 pattern** | {solid, striped} 중 1 like·1 dislike(유저 절반 교차); floral은 like/dislike/중립 ⅓ 순환; {checkered, leopard, polka_dot} 항상 중립 | checkered=compatible-중립 도구, leopard·polka_dot=위반값 공급 → pattern 회전 보장 |
| **R4 rejection sampling** | FREE 배정이 어떤 시나리오의 중립 garment 쌍(compatible/incompatible 각 ≥1)을 죽이면 재추첨 | 잔여 커버리지 홀(v3의 severe_weather 6건) 제거 |

### 3.3 시나리오 공동 규칙 (프로필 규칙의 성립 조건)

| 규칙 | 내용 | 상태 |
|---|---|---|
| **S1** | 규범-격식 7 아키타입의 garment compatible ⊇ ANCHOR {blazer, formal_shirt, dress} | dress 9건 추가 필요 — **시뮬 패치로만 존재, 카탈로그 미반영** |
| **S2** | 모든 color-incompatible 목록은 hard-RESERVE 색 ≥1 포함 (위반값 공급 보장) | 예외 4건 문서화: safety_visibility 3종(ANCHOR 산수로 대체 보장), greenscreen |

### 3.4 알려진 트레이드오프 (정직하게 논문에 쓸 것)

1. **pattern 선호 다양성 축소**: A/B가 사실상 solid↔striped(+물리에서 floral).
   어휘 6개 중 3개를 측정 도구로 예약한 대가. 근본 해결은 pattern 어휘 확장
   (이미지 코퍼스 캐스케이드 → future work).
2. **RESERVE 색 4종은 벤치 전체에서 선호 대상이 아님**: 선호 측정이 13색이 아닌
   9색 위에서 수행 — "TPO 의미를 가진 값은 선호 채널에서 배제"라는 통제 변인으로
   정면 서술 (이 분리가 없으면 "노랑 싫어해서 거른 것 vs TPO 이해" 교란 공격이 성립).
3. **strict 시나리오의 pattern 회전 불가는 존치**: religious_modest 등
   compatible={solid, striped}는 R3가 필연적으로 소진 → 의도적 설계로 문서화.

---

## 4. 시나리오 카탈로그 v2 (구현 완료)

### 4.1 분류 격자 — D3(교락) 해소

| 제약 근원 \ 격식 | Formal | Casual |
|---|---|---|
| **규범적** | A1–A7 (기존 7 아키타입, 28) | **A11 club_code (3)** |
| **기능적** | **A10 stage_tv_interview (1)** | **A8 safety_visibility (3), A9 field_stealth (3), A10 나머지 (2)** |

4칸이 전부 채워져 "formality → 제약 유무" 추론이 구조적으로 무효화된다.

### 4.2 신규 아키타입 요지

| 아키타입 | 색 방향 | 핵심 설계 포인트 | 주 호스팅 |
|---|---|---|---|
| A8 safety_visibility | **역방향** (bright ✓ dark ✗) | D4 해소 담당 — RESERVE 색이 정답 선지에 등장, ANCHOR 색이 위반값 | color 위반 전원 보장 |
| A9 field_stealth | earth tone | **checkered를 compatible에 포함**(플란넬) → RESERVE-중립으로 pattern 위반 성립 | color·pattern 위반 |
| A10 stage_media | 시나리오별 | tv_interview가 기능×포멀 유일 셀 — garment-active + 기능 제약 결합 | garment-active, color 위반 |
| A11 club_code | white 단일 등 | 규범이지만 비격식 — 교락을 안쪽에서도 차단. tennis whites는 성문 규칙 | color 위반 전원 보장 |

### 4.3 상식성 정책 — "인간도 모르는 문제" 방지

- 측정 대상 분리: **상식 규범은 implicit 허용, 틈새 기능 규칙은 explicit(지시 명시) 전용**
  → 후자는 "지식 문제"가 아닌 "제약 적용 + 개인화 결합 문제"로 변환.
- weak-implicit 2종(wildlife_hide, greenscreen)은 implicit 시드에도 규칙 출처를 내장.
- **전 시나리오 인간 정답률 검증이 배포 전 필수** (기준 미달 시 implicit 금지/제외).
- 부수 효과: explicit vs implicit 성능 차 = "지식 vs 적용" 분석 축.

### 4.4 보수적 검수 패치 (구현 완료) — 반례 봉쇄

원칙: **incompatible 목록이 입증 부담** — 합리적 다수가 동의 못하는 항목은 라벨 노이즈.

| 패치 | 반례 | 조치 |
|---|---|---|
| ultra_formal ×4 | 핑크/옐로 이브닝가운은 갈라에서 정상 (젠더 비대칭) | `violation_garment_scope: [blazer, formal_shirt, slacks]` — color 위반을 수트 계열 garment로만 실현 |
| business ×4 | 핑크 드레스셔츠는 주류 비즈니스 복장 | C− pink 제거 → {orange, yellow} |
| yacht_regatta | Nantucket red는 요트클럽 아이콘 | C− red·pink 제거 → {orange, yellow} |
| mourning ×4 | 다크그린 정장은 장례 통용 가능 | C− green 제거 |
| gallery·first_date | 도트 원피스는 클래식 데이트 복장 | P− polka_dot 제거 → {leopard} |
| greenscreen | 그린스크린에서 파란 옷은 무해 | "green & blue 스테이지 겸용"으로 재프레임 |

패치 후 v4 재실행: **분포 완전 동일(무손실)** — 제거된 값은 회전 보장 담당이 아니었음.

잔여 리스크(파이프라인 처리): religious 계열 'dress' 토큰은 민소매 포함 →
이미지 수집 시 covered-style 필터 필요. biz 4종 제약 동일 → pseudo-replicate
공격에는 "TPO 맥락·시드 상이 + 아키타입 수준 클러스터 통계"로 대응.

---

## 5. 문제 생성 구조

### 5.1 선지 불변식 (기존 유지, 전 카테고리 공통)

> **선호는 active 축 한 곳, TPO는 violation 축 한 곳, 나머지는 무해한 배경**
> (TPO-compatible ∩ 선호중립, 없으면 unfixed)

| | active 축 (선호) | violation 축 (TPO) | 배경 축 |
|---|---|---|---|
| A tpo_and_preference | liked ∩ compatible | compatible-중립 | comp ∩ 중립 (공유) |
| B tpo_only | disliked ∩ compatible | compatible-중립 | 〃 |
| C preference_only | liked ∩ compatible | **incompatible-중립** | 〃 |
| D neither | disliked ∩ compatible | 〃 | 〃 |

### 5.2 병렬 variant 정책 (문제 수 확장)

- 쿼리당 위반 축을 "선택"하지 않고, **feasible한 모든 위반 축으로 각각 문제 생성**
  (garment 판은 기본, color/pattern 판은 조건부 추가).
- 같은 유저×시나리오의 (active=color, active=pattern, active=garment) 쿼리를 합치면
  3축 위반이 모두 커버됨.
- **paired probe 이점**: 같은 쿼리에서 위반 축만 다른 쌍 → 축 이해도의 within-query
  통제 비교 (논문 분석 축).
- **주의(통계)**: 쌍둥이 문제는 비독립 → 쿼리 단위 클러스터 부트스트랩 필수.
- **주의(균형)**: A/B 값·garment 쌍 배분 카운터를 variant 단위 슬롯으로 재실행
  (D6 재발 방지).

### 5.3 garment-active 템플릿 (신규)

- Dress-code 한정: A/C = liked ∩ compatible garment, B/D = disliked ∩ compatible,
  위반 축은 color/pattern으로 **강제 회전** (garment는 active라 위반 불가).
- 성립 조건 = R1 + S1. field/stage/club 계열은 부분 호스팅(호스팅 클래스로 선언).

### 5.4 호스팅 클래스 (프레임워크 일반화)

역방향(A8)·단일값 compatible(백스테이지·테니스) 시나리오는 전역 ANCHOR 가정을
깨므로, 보장 명제를 **"모든 시나리오는 자신이 선언한 템플릿 집합을 모든 유저에
대해 호스팅한다"**로 일반화한다. 예: tennis_whites는 color-active 불가(white는
RESERVE) 대신 color 위반 전원 보장 — 표는 §6.2.

---

## 6. 검증 결과 (시뮬레이션 이력)

### 6.1 반복 이력 — 왜 규칙 기반인가

| 버전 | 접근 | 결과 | 판정 |
|---|---|---|---|
| v1 | 프로필 임시 스왑 (무검증) | garment-active 69→249, **기존 커버리지 −156** | 기각 (커버리지 손실) |
| v2 | 커버리지-보존 검증 스왑 | 손실 0, garment-active 169, **3명 0건** | 기각 (per-user 홀) |
| v3 | **티어 쿼터 규칙 생성** | 커버리지 99.8%, garment-active 576 **전원 24개 균등** | 채택 |
| v4 | v3 + 카탈로그 v2 (72 시나리오) | 아래 표 | 채택 (검수 패치 후 재확인) |

→ "임시 수정 → 검증 → 실패 → 규칙화"의 수렴 과정 자체가 규칙 기반 설계의 정당화.

### 6.2 v4 최종 수치 (문제 수 상한 기준)

**위반 축 분포:**

| 카테고리 | 총 | Garment | Color | Pattern |
|---|---|---|---|---|
| Physical | 1,536 | 1,536 (100%, 정의) | 0 | 0 |
| Dress-code | 3,200 | 1,627 (50.8%) | 1,107 (34.6%) | 466 (14.6%) |
| 　└ 규범-격식 | 2,616 | 51.4% | 33.9% | 14.7% |
| 　└ 신규 코드형 | 584 | 48.5% | 37.5% | 14.0% |
| **전체** | **4,736** | 66.8% | 23.4% | 9.8% |

(기존 실현치 89/9/2 대비 color ×4.5, pattern ×7.5. 신규 아키타입이 내부 비율을
왜곡하지 않음 — 규범 51/34/15 vs 신규 49/38/14.)

**기타:** per-user 129~134 (균등) · "bright=오답" 차단 문제 84건 ·
S2 예외 4건(설계) · religious garment-active 0(의도).

### 6.3 garment이 여전히 많은 이유 (선제 방어 문단)

① Physical 1,536은 정의상 garment ② dc의 garment 판은 "모든 쿼리가 갖는 기본
판"(color/pattern 판은 조건부 보너스) ③ color/pattern은 선호 축 겸직으로 위반
후보에서 수시 제외 ④ 도메인 실제 구조가 garment-first("장례식에 반바지 금지"가
1차 규범). → 대응: **축별 macro-average를 헤드라인 지표로**, 필요시 dc garment
기본 판 서브샘플(paired-probe 서브셋은 보존).

---

## 7. 평가 프로토콜 보강 (데이터 재생성과 독립)

| 항목 | 내용 | 막는 공격 |
|---|---|---|
| E1 | TPO 점수를 위반 축별 분리 보고 + macro-average 헤드라인 | D1 잔여("garment만 잘하면 총점") |
| E2 | **B vs C 상대 순위** 부가 지표 (TPO-only vs preference-only) | "정답 A가 항상 지배적 → conjunction 탐지로 환원" (W1) |
| E3 | 인간 상한선 측정 (층화 200–300문항, 정답률+일치도) | 합성 데이터 타당성 (W4) + 신규 시나리오 상식성 |
| E4 | explicit vs implicit 분리 분석 | 지식 vs 적용 구분 |
| E5 | 쿼리 단위 클러스터 부트스트랩 CI | 병렬 variant 비독립 |
| E6 | 선지 위치 셔플 확인 + fixed-attr 통계 규칙성 vs 정답 무상관 검산 | position bias, 배경 속성 지름길 |
| E7 | 다중 시드(3개) 재생성으로 모델 순위 안정성(Kendall τ) | 단일 인스턴스 결론 |

---

## 8. 실행 계획 (의존성 순서)

```
[완료] 카탈로그 v2 (신규 12 + 검수 패치)          — configs/scenarios.py
[완료] 규칙·수치 검증 (시뮬 v3/v4)                 — scratchpad/sim_v3_rules.py, sim_v4.py

1. profile_generator.py에 R1–R4 이식               (v4 시뮬 생성기 → 실코드)
2. 카탈로그 S1 반영 (규범-격식 7종에 dress 9건)     ← 재생성 직전에 (기존 데이터 재현성 보존)
3. query_generator: garment-active 축 지원,
   회전-인지 active 축 배분, 틈새 시나리오 explicit 정책
4. option_planner: 병렬 variant, garment-active 템플릿,
   violation_garment_scope 소비(assign_violation_values 교집합 한 줄),
   배분 카운터 variant-슬롯화
5. 전체 재생성 (seed 42 + 시드 2개 추가)
6. 재검증: 커버리지 / per-user 균형 / 혼동쌍 비율 / 위반 축 분포
   / fixed-attr–정답 상관 검산 (§7 E6)
7. 이미지 수집: modest 시나리오 covered-style 필터, 신규 시나리오 코퍼스 확인
8. 인간 검증 (E3) → 미달 시나리오 implicit 금지/제외 → 벤치 확정
```

**게이트 조건**: 6에서 하나라도 기존 지표가 악화되면 5로 롤백하지 말고 원인
규칙(R/S)을 수정 후 재생성 — 임시 패치 금지 (v1/v2의 교훈).

---

## 9. 공격 → 방어 매핑 (자기 점검표)

| 예상 공격 | 방어 위치 |
|---|---|
| "garment만 봐도 TPO 만점" | §2 이원화 + §5.2 회전 + §7 E1 |
| "garment 선호는 왜 안 재나" | §5.3 garment-active (규범-격식 전수 보장) |
| "정답이 항상 지배적" | §7 E2 (B vs C) — **미구현, 평가 단계** |
| "formality 휴리스틱 지름길" | §4.1 격자 완비 |
| "밝은 색 = 오답 상관" | §4.2 A8·A11 (84건) |
| "이 조합이 왜 부적절? (반례)" | §4.4 검수 패치 + incompatible 입증부담 원칙 |
| "인간도 못 푸는 문제" | §4.3 explicit 정책 + E3 인간 검증 |
| "합성 프로필 타당성" | §3.4 통제변인 서술 + E3 |
| "쌍둥이 문제 n 부풀리기" | §7 E5 클러스터 통계 |
| "왜 이 유저는 이 문제가 없나" | §3 by-construction 보장 + §5.4 호스팅 클래스 선언 |
| "문화 편향" | CULTURAL_FRAME 명시 + 기능 시나리오는 무관 + wedding-white 등 보수 유지 |

---

## 10. 미결 사항 (검토 시 결정 필요)

1. **dc garment 기본 판 서브샘플 비율** — 예산(이미지 수집량)과 함께 결정.
   전량 유지 시 4,736, paired-probe 보존 하에 감축 가능.
2. **pattern 어휘 확장 여부** — R3 트레이드오프(§3.4-1)의 근본 해결이나
   코퍼스 캐스케이드 비용. 이번 제출은 한계+future work 권고.
3. **신규 시나리오 이미지 코퍼스 커버리지** — hi-vis 러닝복, 사파리 셔츠 등
   검색 가능성 사전 확인 필요 (7번 단계 전).
4. **기존 데이터와의 버전 관리** — 재생성본은 v2로 분리 보관, 기존 2,760은
   ablation/비교용 유지 권장.
