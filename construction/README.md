# Benchmark Construction (이미지 수집 직전 단계)

POD-Bench v2의 **텍스트 스펙 구축 단계**(STAGE 1–3)입니다. 사용자 프로필 → 쿼리 →
4지선다(A/B/C/D) 옵션 *명세*까지 만듭니다. 실제 이미지 수집·라벨링·평가는 이 폴더
범위 밖(다음 단계)입니다. 이 단계의 모든 산출물은 **LLM 없이 결정론적**이며 동일
seed에서 동일 출력입니다.

## 데이터 흐름

```
configs/profiles.py ─┐
                     ├─► STAGE 1  profile_generator ─► profiles.jsonl
configs/scenarios.py ┘                                      │
                                                            ▼
                       STAGE 2  query_generator     ─► queries.jsonl
                       (compatibility.py 사용)              │
                                                            ▼
                       STAGE 3  option_planner      ─► option_plans.jsonl
                                                            │
                                                            ▼
                                            [ 이미지 수집 (다음 단계) ]
```

## 폴더 구성 / 의존성

```
construction/
├── README.md
├── __init__.py
├── profile_generator.py     # STAGE 1
├── query_generator.py       # STAGE 2
└── option_planner.py        # STAGE 3
```

- 세 파일은 sibling 공유 모듈 **`utils.py`**(`save_jsonl/load_jsonl/log_step`)와
  **`compatibility.py`**(`get_compatible_instances/print_compatibility_report`)를
  `from .utils` / `from .compatibility`로 import합니다. 두 모듈을 **이 폴더에 함께
  두거나**, 다른 위치(예: `common/`)에 둔다면 그 두 import 줄만 고치세요.
- **`configs/`**(`config.py`, `profiles.py`, `scenarios.py`)는 repo 루트의 top-level
  패키지로 가정합니다 (`sys.path.insert(parent.parent)` + `from configs....`).

## 실행 (repo 루트에서, **모듈 모드 필수**)

```bash
python -m construction.profile_generator --n_users 24 --force
python -m construction.query_generator   --force
python -m construction.option_planner    --force
```

> `python construction/profile_generator.py`처럼 직접 실행하면 relative import
> (`from .utils`)로 실패합니다. 반드시 `python -m construction.<stage>` 형태로.

주요 옵션: `--seed`(기본 42), `--per_instance`(쿼리당 인스턴스 수, 기본 1),
`--explicit_ratio`(explicit vs implicit 쿼리 비율, 기본 0.5), `--limit`(디버그용 상한).

## 2×2 설계 (요약)

| 옵션 | label | active 값(color/pattern) | garment(TPO) |
|---|---|---|---|
| A | tpo_and_preference | liked | compatible |
| B | tpo_only | non-preferred | compatible |
| C | preference_only | liked | incompatible |
| D | neither | non-preferred | incompatible |

- `active_axis ∈ {color, pattern}`. **garment이 TPO 신호를 운반**, color/pattern은
  preference 축.
- A·C는 liked 값 공유, B·D는 non-preferred 값 공유 / A·B는 compatible garment 공유,
  C·D는 incompatible garment 공유.
- 비활성 축(non-active)은 항상 **preference-neutral & TPO-compatible** 값으로 고정
  (없으면 미고정 — 절대 선호/비선호·TPO-위반 값을 넣지 않음).

## 레코드 스키마 (핵심 필드)

**profiles.jsonl**
`user_id`, `preference_archetype`, `variant_index`,
`structured_attributes`{`garment_category`,`color`,`pattern` × `likes`/`dislikes`},
`likes_keywords`, `dislikes_keywords`, `narrative_profile`

**queries.jsonl**
`query_id`, `user_id`, `scenario_id`, `scenario_archetype`, `scenario_name`,
`active_axis`, `liked_compatible`, `disliked_compatible`, `neutral_compatible`,
`compatible_garments`, `incompatible_garments`, `query_type`, `query_text`,
`fixed_attrs`

**option_plans.jsonl**
`query_id`, `user_id`, `scenario_id`, `scenario_archetype`, `scenario_name`,
`active_axis`, `fixed_attrs`, `main_category`(=compatible garment),
`violation_value`(=incompatible garment), `options`{A..D:{`label`,`attributes`,
`search_query`,`rationale`}}

## 재현성 & 불변식 (이번 정리에서 보장)

1. 고정 seed → **동일 출력** (LLM·process-salted hash 없음).
2. 모든 color/pattern 값이 A·B 양쪽에 등장하도록 전역 counterbalancing →
   preference-blind 모델의 value-prior ≈ 0 (≈0.50). **24명 전제**.
3. 비활성 축은 항상 preference-neutral & TPO-compatible (못 채우면 생략).
4. `query_id = q{idx:05d}__{user}__{scenario}__{axis}` → 전역 번호 우선 정렬
   (이미지 폴더도 번호순 정렬됨).
5. normative 시나리오는 `CULTURAL_FRAME = "contemporary_western"` 하에서 정의
   (`configs/scenarios.py`). physical 시나리오는 color/pattern 무제약(순수 preference).

## 변경 이력 (기존 `src/` 대비)

- **profile_generator** — narrative를 결정론 템플릿으로 교체(LLM 제거 → She/This
  혼용 해소, provider 불필요, 재현성 확보); `n_users < 24` 경고 추가; docstring
  21→24 정정.
- **query_generator** — `query_id` 번호 우선 정렬; 비활성 축 fallback을
  scenario-aware + neutral 보장(비선호·TPO-위반 값 누설 차단); profile 조회 O(N²)→O(1);
  `--force` 미지정 시 build 전에 조기 종료.
- **option_planner** — garment 선택 seed를 `hash(str)`(process-salted) → **MD5 기반
  안정 seed**로 교체(재현성); 실패 사유별 카운트(filter %) 리포트 추가.

## 다음 단계 (범위 밖)

`option_plans.jsonl`의 각 옵션 `search_query`/`attributes`로 이미지 수집
(FashionSigLIP + FAISS retrieve-then-verify, patch-coverage / color / segmentation
expert gate) → labeling → audit(`validate_options`) → eval.
