# 공학연구인턴십 성과발표 (2026. 8. 5.)

5분 발표용 슬라이드 18장. 모든 수치는 이 저장소에서 직접 계산해 넣었고,
스크립트를 다시 돌리면 데이터가 바뀐 만큼 그림도 같이 바뀝니다.

```
presentation/
├── POD-Bench_성과발표.pptx   ← 제출본
├── make_figs.py              그림 6종 생성 (matplotlib)
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

## 실제 파이프라인 다섯 단계

발표의 뼈대는 코드가 실제로 수행하는 순서입니다.

```
① construction  →  ② dialog  →  ③ image collection  →  ④ annotation  →  ⑤ eval
        ↑                                                      │
        └──────────────────────────────────────────────────────┘
          ④에서 사람이 승인한 셀 목록이 ①의 생성 입력으로 되돌아감
          (annotation/attribute_library.json → construction/option_planner.py)
```

| 단계 | 코드 | 산출물 |
|---|---|---|
| ① construction | `construction/` | `data_*/profiles·queries·options` |
| ② dialog | `dialog/` | `data_*/dialogs/dlg__U0xx__{image,text}.json` |
| ③ image collection | `fsiglip/`, `availability_audit/` | 후보 이미지 · 스크린 점수 |
| ④ annotation | `annotation/` | `annotation/attribute_library.json` |
| ⑤ eval | `exp/vlm_eval/`, `exp/llm_eval/` | `data_*/eval/`, `data_*/eval_text/` |

되먹임 화살표가 발표의 문제해결 사례 ②입니다 — 이미지 가용성 제약을 파이프라인
끝의 사후 필터가 아니라 ①의 생성 입력으로 옮긴 것.

## 슬라이드 구성 (발표 5분)

| # | 내용 | 누적 |
|---|---|---|
| 1 | 표지 | 0:00 |
| 2 | 목차 | 0:15 |
| 3 | 연구실 · 인턴십 개요 | 0:20 |
| 4 | 무엇을 풀려고 했나 — 한겨울 야외 예시 + A/B/C/D | 0:40 |
| 5 | 어떻게 채점하나 — 답 하나로 두 가지 | 1:05 |
| 6 | **만드는 과정 — 다섯 단계 (조망 + 되먹임)** | 1:30 |
| 7 | ① Construction — 문항 만들기 | 1:50 |
| 8 | **② Dialog — 취향을 대화로 (신규)** | 2:15 |
| 9 | **② Dialog — 실제 대화와 검증 (신규)** | 2:40 |
| 10 | ③ Image collection — 사진 모으기 | 3:05 |
| 11 | ③에서 부딪힌 문제 — 국소 패턴 | 3:25 |
| 12 | ④ Annotation — 사람 확인 + ①로 되먹임 | 3:45 |
| 13 | ⑤ Eval — 모델 8종 결과 | 4:05 |
| 14 | ⑤ Eval — 사진 vs 글자 | 4:20 |
| 15 | ⑤ Eval — 어디가 특히 약한가 | 4:35 |
| 16 | 기여 | 4:40 |
| 17 | 향후 계획 · 소감 | 4:50 |
| 18 | Q & A | 5:00 |

**시간이 모자라면 15번(어디가 약한가)을 빼십시오.** 결론에 영향이 없는 유일한
슬라이드입니다. 각 슬라이드 발표 노트에 대본과 타임코드가 들어 있습니다.

## 서술 원칙

심사위원은 VLM 연구자가 아니므로 Method는 **왜 필요한지 → 한 줄 설명 → 예시**
순서로만 씁니다. 한 칸에는 제목 1줄 + 본문 최대 2줄.

| 원래 표현 | 슬라이드 표현 |
|---|---|
| 2×2 factorial design | 답 하나로 두 가지를 채점 |
| Strict / TPO / Preference | 둘 다 맞춤 / 상황 점수 / 취향 점수 |
| deterministic construction | 사람 손이 들어가지 않는 규칙 생성 |
| expression mode (single/differ/share2/share3) | 대화가 취향을 전달하는 네 가지 방식 |
| image variant / text twin | 사진판 / 글자판 (쌍둥이) |
| global average pooling | 사진 한 장을 숫자 하나로 요약 |
| availability를 generation constraint로 | 거르는 순서를 바꿨다 |

'한겨울 야외 시장 + 파란색을 좋아하는 사람' 예시 하나를 4번 슬라이드부터 끝까지
관통시켰습니다.

## 심사 기준 대응

| 항목 | 배점 | 대응 |
|---|---:|---|
| 체계 및 구성 | 30 | 목차 시간 배분 · 문제 → 다섯 단계 → 결과 → 계획 · 6번에서 전체 조망 |
| 전공교육과 연관성 | 40 | 11번·12번 문제해결 2건(증상→시도→원인→해결), 9번 검증 설계, 17번 전공 수업 연결 |
| 표현 및 발표 | 30 | 전 슬라이드 그림/도식, 기존 발표자료의 색·서체·헤더 룰 유지, 칸당 글자 수 제한 |

## 수치 출처 (전부 직접 실행해 확인)

| 수치 | 출처 |
|---|---|
| 문항 2,621개 / physical 999 · dress_code 1,622 | `data_wacv_scenario_v5/options/option_plans.jsonl` |
| 상황 59개 · 사람 24명 · query 2,561개 | `configs/scenarios.py`, `data_*/profiles·queries` |
| 대화 48개 · 턴 평균 48.0 · 이미지 평균 32.8 | `dialog.report_dialog_balance` 실행 |
| evidence 432 = 24 × 18 · downgrade 8건(1.9%) | 같음 |
| 방식 single 144 / differ 152 / share2 69 / share3 67 | 같음 (요청 144/144/72/72) |
| 검증 48/48 통과 · 돌연변이 6/6 검출 | `dialog.validate_dialogs --self-test` 실행 |
| 셀 1,661 → available 1,223 / excluded 438 | `annotation/attribute_library.json` |
| 제외 사유 218 / 210 / 10 | 같음 |
| 퍼널 166,100 → 127,859 → 38,829 → 19,012 | `docs/PROCESS.md` §3–§4 |
| v4 33.4% → v5 97.3% | `docs/PROCESS.md`, plans × attribute_library 교차 계산 |
| 모델별 strict/tpo/pref | `data_*/eval/*/results.jsonl` 8개 디렉터리 직접 집계 |
| 사진 73.1% vs 글자 83.8% | `eval/` vs `eval_text/` (Qwen3-VL-30B, 같은 2,550 문항) |

### 평가 실행 현황 (results.jsonl 직접 집계)

| 모델 | 행 | 오류 | 상태 |
|---|---:|---:|---|
| Qwen3-VL-30B-A3B | 12,750 | 0 | 완주 (5조건 × 2,550) |
| Qwen3-VL-4B | 12,750 | 0 | 완주 |
| Qwen2.5-VL-7B | 12,750 | 0 | 완주 |
| InternVL3.5-8B | 12,750 | 0 | 완주 |
| Gemma-3-27B | 12,752 | 0 | 완주 |
| GPT-5-mini | 12,750 | 0 | 완주 |
| Qwen2.5-VL-72B-AWQ | 2,518 | 0 | **부분 실행** — 슬라이드에서 제외 |
| MiniCPM-V-4.6 | 12,750 | 12,744 | **전량 실패** — 슬라이드에서 제외 |
| (text-only) Qwen3-VL-30B | 12,750 | 0 | 완주 |

슬라이드의 "AI 8종"은 **실행을 시도한 수**이고, 그래프의 6개 선은 **완주한 모델**입니다.
발표 때 이 구분을 물으면 위 표대로 답하십시오.

### 그림의 한글 폰트

이 환경의 유일한 한글 지원 폰트인 WenQuanYi Zen Hei는 **볼드 자체(字體)가 없고
U+2212 마이너스 기호도 없습니다.** 그래서 `make_figs.py`는 한글을 Zen Hei로,
숫자를 볼드가 실제로 먹는 DejaVu Sans로 렌더링하고, 마이너스는 ASCII 하이픈을
씁니다. 퍼널(10번)·이전/현재 막대(12번)·채점표(5번)·파이프라인(6번)은 그림 대신
**PowerPoint 도형으로 직접** 그렸습니다 — 그쪽은 맑은 고딕 볼드가 제대로 나옵니다.

## 템플릿 정리 (PowerPoint 멈춤 수정)

첫 빌드에서 PowerPoint가 파일을 열다 멈췄습니다. 원인은 템플릿이 원본 발표자료의
패키지를 통째로 물려받은 것이었고, `sanitize_template.py`로 세 가지를 제거했습니다.

| 잔재 | 문제 |
|---|---|
| `<p:embeddedFontLst>` + `ppt/fonts/*.fntdata` (2.9 MB) | 원본 76장이 쓰던 글자만 담긴 **맑은 고딕 서브셋**. 새 슬라이드의 한글은 그 서브셋 밖이라 PowerPoint가 커버되지 않는 임베드 폰트를 처리하려다 멈춤 |
| `docProps/app.xml` | `<Slides>76</Slides>`, `TitlesOfParts` 79개 — 실제 슬라이드 수와 불일치. python-pptx는 이 파트를 그대로 복사만 함 |
| `docProps/thumbnail.jpeg`, `core.xml` | 이전 파일의 미리보기 이미지와 작성자 정보(revision 122) |

재발 방지로 `build_deck.py`가 저장 직후 검사합니다 — 모든 relationship과 content type이
실제 파트를 가리키는지, 임베드 폰트가 다시 들어오지 않았는지, `app.xml`의 슬라이드 수가
실제와 맞는지. 하나라도 어긋나면 빌드가 실패합니다.

## 발표 전 반드시 확인할 것

- **대화(dialog) 조건은 아직 평가하지 않았습니다.** `dialog/`는 생성과 검증까지
  끝났지만 `exp/vlm_eval/eval_multimodal.py:85`의 `INPUT_FORMATS`는 여전히
  `["query", "narrative", "narrative+query", "all", "all+query"]` 다섯 가지뿐이고
  대화 조건은 없습니다. `dialog_eval/` 결과 디렉터리도 없습니다.
  슬라이드 16번 '한계'와 17번 '남은 일'에 이 사실을 명시했습니다.
  **"대화로 줬더니 성능이 이랬다"는 말은 절대 하지 마십시오 — 아직 데이터가 없습니다.**
- **`dialog/README.md`의 통계는 한 세대 전 숫자입니다** (differ 151, share2 70,
  downgrade 7, 고유 셀 497). 현재 데이터는 differ 152 / share2 69 / downgrade 8 /
  고유 셀 489입니다. 슬라이드는 현재 데이터를 씁니다.
- **대화가 참조하는 이미지 파일은 이 저장소에 없습니다** (`data_*/images/`는
  gitignore). `report_dialog_balance`가 "파일 없는 이미지 489종"을 경고하는 것이
  이 때문이며, 셀 자체는 검수 완료된 available 셀입니다.
- **available 셀 수는 1,223입니다.** 최상위 `README.md`의 1,237은 `long_coat`/
  `pea_coat` 어휘 정정 전 숫자입니다. 대화 매니페스트의 `library_sha256_16`
  (`828978fa2eab1577`)이 현재 라이브러리 해시와 일치하는 것을 확인했습니다.
- 슬라이드의 그래프는 **Qwen3-VL-30B** 기준입니다(완주 모델 중 최고). 부분 실행된
  Qwen2.5-VL-72B는 80.3%로 더 높지만 문항의 20%만 돌아서 제외했습니다.
