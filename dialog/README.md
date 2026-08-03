# Dialog Construction (대화 기반 프로필 전달)

프로필을 서술문(narrative)이나 키-값(all)이 아니라 **다중 턴 대화**로 전달하는
입력 형식을 만듭니다. 대화는 새로운 과제가 아니라 기존 ablation 의 새 조건
(`dialog`, `dialog+query`)이며, 비교 대상은 narrative 입니다.

핵심 원칙: **대화는 취향만 전달하고 상황 판단은 절대 포함하지 않습니다.**
대화 속 어시스턴트가 "결혼식엔 흰색을 피하세요" 같은 말을 하는 순간 평가하려는
능력이 대화에 새어 나갑니다.

## 데이터 흐름

```
configs/dialog_templates.py ─┐   (에피소드·문장 템플릿)
profiles.jsonl ──────────────┤
attribute_library.json ──────┘
                             │
        D1 expression.py          방식 정의(single/differ/share2/share3)
                             │
        D2 image_picker.py        available 셀에서 방식별 조합 선택
                             │
        D3 dialog_planner.py      취향 18개 배정 → 에피소드/위치 계획
                             │
        D4 renderer.py            계획 → 턴 (image 변형 + text 쌍둥이)
                             │
                             ▼
              data_<variant>/dialogs/dlg__U001__image.json
                                     dlg__U001__text.json
                                     dialogs_manifest.json
                             │
        validate_dialogs.py       R1~R5 + 커버리지 (독립 실행)
        report_dialog_balance.py  방식·축·극성·downgrade 분포
```

## 재생성

```bash
POD_VARIANT=wacv_scenario_v5 python -m dialog.build_dialogs
POD_VARIANT=wacv_scenario_v5 python -m dialog.validate_dialogs --self-test
POD_VARIANT=wacv_scenario_v5 python -m dialog.report_dialog_balance
```

시드 고정 결정론이므로 같은 입력에서 같은 대화가 나옵니다. `--users U001 U002`
로 일부만 생성할 수 있습니다.

## 표현 방식

취향 사실 하나는 (축, 값, 극성) 셋입니다. image 변형에서는 **값이 항상
이미지에만** 있고, 축과 극성은 텍스트가 나릅니다. 축은 값 이름 대신 힌트
단어로 지목합니다: 색→`color`, 무늬→`print`, 의류→`cut`.

| 방식 | 이미지 | 모델이 해야 할 일 |
|---|---:|---|
| `single` | 1 | 그 축의 값을 읽기 |
| `differ` | 2 | 값 읽기 + 지시 대상 정렬(서수) |
| `share2` | 2 | 두 이미지 비교 → 공통값 찾기 |
| `share3` | 3 | 세 이미지 교집합 계산 |

`share` 의 텍스트가 "정확히 하나를 공유하고, 그것이 내 취향"이라고 **명시적으로
단언**하므로 과제는 추측이 아니라 집합 계산입니다. 검증기가 실제로 한 축만
공유하는지 확인합니다.

### 축 종류에 따른 문장 분기

문장은 축이 **속성(색·무늬)** 인지 **의류** 인지에 따라 갈립니다. 사실 정합의
핵심입니다:

| | differ 의 두 이미지 | 쓸 수 있는 문장 |
|---|---|---|
| 색·무늬 축 | 같은 옷, 그 속성만 다름 | "같은 물건 두 개, 하나만 다름" ✓ |
| 의류 축 | **다른 옷**, 색·무늬는 같음 | "다른 두 벌, 나머지는 동일" |

의류 축에서 "same piece" 라고 쓰면 스웨터와 청바지를 같은 물건이라 부르는
셈이라 문장이 이미지와 어긋납니다. 그러면 모델은 텍스트와 이미지 중 하나를
버려야 하고, 측정값은 모델 능력이 아니라 데이터 결함을 반영하게 됩니다.
`share` 도 같은 이유로 갈립니다 — 의류를 공유하면 서로 닮아 보이므로
"look nothing alike" 가 어색합니다.

`differ` 의 파트너 이미지 값은 사용자-중립이어야 합니다. 비선호 값을 쓰면
말하지 않은 극성이 우연히 맞아버려 통제가 흐려집니다.

`share` 에서 다른 축은 "쌍마다 달라야" 하는 것이 아니라 "**전부가 공유하지는
않아야**" 합니다. 색이 evidence 일 때 사용자-중립 무늬가 `solid` + 1종뿐이라
(2+2가 취향으로 소진) 3장에 서로 다른 무늬를 줄 수 없기 때문입니다.
`solid/solid/striped` 면 무늬는 공통 원소가 아니므로 연역은 유지됩니다.

## 커버리지와 쿼터

프로필의 취향 **18개(like 9 + dislike 9)를 전부** 노출합니다. narrative/KV
조건이 18개를 모두 제공하므로, 대화도 전부 실어야 정보량이 같아지고 조건
비교가 공정해집니다.

방식 쿼터는 `single 6 / differ 6 / share2 3 / share3 3` 입니다. **커버리지는
경성 제약, 쿼터는 연성 목표**입니다 — 셀이 없어 요청 방식이 불가능하면
`share3 → share2 → differ → single` 로 완화하고 `downgrades` 에 기록합니다.
조용한 폴백을 만들지 않습니다.

## 검증 규칙

| | 내용 |
|---|---|
| R1 | 시나리오·TPO 어휘 금지 (단어 경계로 검사) |
| R2 | image 변형 턴에 벤치마크 어휘(13색·23의류·6무늬)와 렌더링 별칭 금지 |
| R3 | 비-evidence 축 값은 사용자-중립 (`solid` 는 전원 중립 기준선) |
| R4 | 이미지는 검수 완료 available 셀만 |
| R5 | 방식별 연역 조건 |
| C | 취향 18개 전부 노출 |
| P | like/dislike 둘 다 존재 |

`validate_dialogs.py` 는 생성기와 **독립 실행**입니다. 생성기 안에서만 검증하면
자기 산출물을 자기 기준으로 통과시키는 순환이 생깁니다. `--self-test` 는 일부러
위반을 주입해 검증기가 잡는지 확인합니다(돌연변이 테스트).

## text 쌍둥이

`dlg__U001__text.json` 은 같은 계획에서 렌더링되며 이미지 자리에
`{blue solid hoodie}` 자리표시자가 들어갑니다. 턴 수·순서·시간·evidence 배치가
완전히 동일하므로 **두 조건의 유일한 차이가 값의 모달리티**입니다. 두 판의
성능 차이가 곧 "이 대화를 이해하는 데 시각 인지가 얼마나 필요한가"입니다.

## 현재 규모 (24명, seed 42)

```
대화 48개 (image 24 + text 24)   evidence 432 = 24 × 18
평균 턴 48.0   이미지 32.8   (턴당 0.68)
방식 실제/요청  single 144/144  differ 151/144  share2 70/72  share3 67/72
downgrade 7건 (1.6%)  — 전부 의류 축(pea_coat, leather_jacket 등) 셀 부족
differ 위치  first 78 / second 73  (카운터밸런스 ✓)
고유 셀 497개 / 참조 787회
```

## 알려진 제약

* 대화 하나에 이미지 33장 + 선택지 4장 = **프롬프트당 37장**입니다. vLLM 서빙을
  `--limit-mm-per-prompt '{"image":40}'` 와 충분한 `--max-model-len` 으로 다시
  띄워야 합니다. 512px 기준 이미지만 1.5만 토큰입니다.
* 대화는 사용자당 하나이므로 같은 사용자의 문항들이 동일한 접두사를 공유합니다.
  평가 프롬프트에서 **대화를 query·옵션보다 앞에** 두면 vLLM 접두사 캐싱과
  API 캐시 단가가 적용됩니다.
* 문장이 템플릿 기반이라 표면 다양성이 낮습니다. LLM 패러프레이즈를 붙인다면
  `renderer.py` 에만 들어가며, 채택 전 `validate_dialogs` 를 통과해야 합니다.
  템플릿판은 보존해 생성 모델 문체 의존성 확인용으로 씁니다.