# 공학연구인턴십 성과발표 (2026. 8. 5.)

5분 발표용 슬라이드 13장. 모든 수치는 이 저장소에서 직접 계산해 넣었고,
스크립트를 다시 돌리면 데이터가 바뀐 만큼 그림도 같이 바뀝니다.

```
presentation/
├── POD-Bench_성과발표.pptx   ← 제출본
├── make_figs.py              그림 4종 생성 (matplotlib)
├── build_deck.py             슬라이드 조립 (python-pptx) + 저장 시 패키지 무결성 검사
├── sanitize_template.py      템플릿에서 원본 파일의 잔재 제거 (아래 참고)
├── assets/
│   ├── template.pptx         기존 발표자료에서 슬라이드만 비운 껍데기
│   │                         (A4 10.83×7.5in · 맑은 고딕 테마 · 마스터)
│   ├── header_rule.png       상단 파란 점선 룰
│   └── uos_logo.png          서울시립대 로고
└── figs/                     생성된 그림 (make_figs.py 산출물)
```

## 다시 만들기

```bash
pip install python-pptx matplotlib
cd presentation
python make_figs.py     # figs/ 재생성
python build_deck.py    # POD-Bench_성과발표.pptx 재생성
```

## 슬라이드 구성 (발표 5분)

| # | 내용 | 누적 시간 |
|---|---|---|
| 1 | 표지 | 0:00 |
| 2 | 목차 | 0:15 |
| 3 | 연구실 · 인턴십 개요 / 담당 범위 | 0:20 |
| 4 | 무엇을 풀려고 했나 — 한겨울 야외 예시 + A/B/C/D | 0:40 |
| 5 | Method ① 답 하나로 두 가지를 채점 | 1:05 |
| 6 | Method ② 문항 만드는 법 (사람 → 상황 → 문항) | 1:30 |
| 7 | Method ③ 진짜 어려운 건 사진 | 1:55 |
| 8 | 문제해결 ① 검색이 엉뚱한 옷을 가져온다 | 2:20 |
| 9 | 문제해결 ② 거르는 순서를 바꿨다 | 2:50 |
| 10 | 결과 — 따로는 되는데 같이는 안 된다 | 3:20 |
| 11 | 기여 | 3:50 |
| 12 | 향후 계획 · 소감 | 4:15 |
| 13 | Q & A | 5:00 |

## 서술 원칙

심사위원은 VLM 연구자가 아니므로, Method는 **왜 필요한지 → 한 줄 설명 → 예시**
순서로만 씁니다. 전문 용어는 슬라이드에서 빼고 발표 노트로 내렸습니다.

| 원래 표현 | 슬라이드 표현 |
|---|---|
| 2×2 factorial design | 답 하나로 두 가지를 채점 |
| Strict / TPO / Preference | 둘 다 맞춤 / 상황 점수 / 취향 점수 |
| deterministic construction (Stage 1–3) | 사람 손이 들어가지 않는 규칙 생성 |
| FashionSigLIP + SAM3 + patch-wise scoring | 검색 → 옷 부분 찾기 → 칸마다 확인 |
| global average pooling | 사진 한 장을 숫자 하나로 요약 |
| availability를 generation constraint로 이동 | 거르는 순서를 바꿨다 |

한 칸(카드·텍스트박스)에는 **제목 1줄 + 본문 최대 2줄**까지만 넣습니다.

각 슬라이드 발표 노트에 대본과 타임코드가 들어 있습니다 (PowerPoint 슬라이드 노트).

## 심사 기준 대응

| 항목 | 배점 | 대응 |
|---|---:|---|
| 체계 및 구성 | 30 | 목차에 시간 배분 명시 · 문제 → 방법 → 문제해결 → 결과 → 계획 순서 |
| 전공교육과 연관성 | 40 | 슬라이드 8–9 문제해결 2건 (증상 → 시도 → 원인 규명 → 해결), 슬라이드 12 전공 수업이 어디에 쓰였나 |
| 표현 및 발표 | 30 | 전 슬라이드 그림/도식 배치, 기존 발표자료의 색·서체·헤더 룰 유지, 칸당 글자 수 제한 |

## 수치 출처

| 수치 | 출처 |
|---|---|
| 2,621 option plans / 999 · 1,622 track 분리 | `data_wacv_scenario_v5/options/option_plans.jsonl` |
| 2,561 queries · 24 profiles | `data_wacv_scenario_v5/queries/`, `profiles/` |
| 97.3% (2,550/2,621) 4장 확보 | plans × `annotation/attribute_library.json` 교차 계산 |
| 1,661 셀 → 1,223 available | `annotation/attribute_library.json` |
| 166,100 → 127,859 → 38,829 → 19,012 | `docs/PROCESS.md` §3–§4 (퍼널은 슬라이드에서 직접 그림) |
| v4 33.4% (1,012/3,027), track 38:62 → 31:69 | `docs/PROCESS.md`, `docs/wacv_scenario_v5_report.md` |
| 1,964 counterbalanced plans | `data_wacv_scenario_v5/options/validation_report.counterbalanced_ids.json` |
| Strict/TPO/Pref × 5 input format | `data_wacv_scenario_v5/eval/Qwen_Qwen3-VL-4B-Instruct/results.jsonl` (4,189행, 오류 0) |

`make_figs.py` 상단 주석에 파일별 대응이 정리되어 있습니다.

### 그림의 한글 폰트

이 환경의 유일한 한글 지원 폰트인 WenQuanYi Zen Hei는 **볼드 자체(字體)가 없고
U+2212 마이너스 기호도 없습니다.** 그래서 `make_figs.py`는 한글을 Zen Hei로,
숫자를 볼드가 실제로 먹는 DejaVu Sans로 렌더링하고, 마이너스는 ASCII 하이픈을
씁니다. 퍼널(7장)과 이전/현재 막대(9장)는 그림 대신 **PowerPoint 도형으로 직접**
그렸습니다 — 그쪽은 맑은 고딕 볼드가 제대로 나옵니다.

## 템플릿 정리 (PowerPoint 멈춤 수정)

첫 빌드에서 PowerPoint가 파일을 열다 멈췄습니다. 원인은 템플릿이 원본 발표자료의
패키지를 통째로 물려받은 것이었고, `sanitize_template.py`로 세 가지를 제거했습니다.

| 잔재 | 문제 |
|---|---|
| `<p:embeddedFontLst>` + `ppt/fonts/*.fntdata` (2.9 MB) | 원본 76장이 쓰던 글자만 담긴 **맑은 고딕 서브셋**. 새 슬라이드의 한글은 그 서브셋 밖이라 PowerPoint가 커버되지 않는 임베드 폰트를 처리하려다 멈춤. 맑은 고딕은 Windows 기본 폰트라 임베드가 애초에 불필요 |
| `docProps/app.xml` | `<Slides>76</Slides>`, `TitlesOfParts` 79개 — 실제 슬라이드 수와 불일치. python-pptx는 이 파트를 그대로 복사만 함 |
| `docProps/thumbnail.jpeg`, `core.xml` | 이전 파일의 미리보기 이미지와 작성자 정보(revision 122) |

결과: **3.53 MB → 0.34 MB**, 렌더링 결과는 동일합니다.

재발 방지로 `build_deck.py`가 저장 직후 검사합니다 — 모든 relationship과 content type이
실제 파트를 가리키는지, 임베드 폰트가 다시 들어오지 않았는지, `app.xml`의 슬라이드 수가
실제와 맞는지. 하나라도 어긋나면 빌드가 실패합니다.

## 발표 전 확인할 것

- **Qwen2.5-VL-7B 결과는 쓰지 않았습니다.** `eval/Qwen2.5-VL-7B-Instruct/results.jsonl`은
  4,624행 전부 엔드포인트 404 오류라 유효한 결과가 없습니다. 슬라이드의 수치는
  Qwen3-VL-4B-Instruct 단일 모델(사진 4장을 보고 고르는 문제 838개 × 5가지 입력 방식)입니다.
  질문이 나올 수 있으니 "현재 유효한 모델은 1종, 확대가 향후 계획"이라고 답할 준비를 해두세요.
- **`summary.json`(n=905)과 슬라이드 수치가 다릅니다.** 그 파일은 이전 부분 실행 기록이고,
  슬라이드는 완주한 `results.jsonl`을 직접 집계했습니다.
- **available 셀 수는 1,223** 입니다. `README.md`에 적힌 1,237은 `long_coat`/`pea_coat`
  어휘 정정 전 숫자로, 현재 라이브러리 파일과 다릅니다.
- text-only 평가 결과 파일은 저장소에 없어 슬라이드에 넣지 않았습니다
  (`exp/llm_eval/text_eval.py`는 있으나 `options/text_eval/` 산출물 없음).
