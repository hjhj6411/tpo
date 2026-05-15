# POD-Bench: Preference Origin Disentanglement Benchmark

VLM이 사용자의 **본질적 취향(Intrinsic Preference)**과 **상황적 선호(TPO: Time/Place/Occasion)**를 구분할 수 있는지 진단하는 멀티모달 벤치마크.

## 핵심 구조

각 인스턴스는 4지선다 + 모든 선지가 이미지:

```
Query: "비 오는 날 외출용으로 가져갈 최선의 선택은?"
User Profile: "차분한 어두운 푸른 톤, 미니멀 디자인 선호..."

A) 네이비 우비, 미니멀 라인       (TPO+선호) ← 정답
B) 빨간 우비, 큰 로고             (TPO만)
C) 네이비 후드티, 미니멀 디자인    (선호만)
D) 베이지 우비, 화려한 패턴       (둘 다 부분 X)
```

오답 패턴이 진단적 의미를 가짐:
- B 편향 → TPO 과다 반영 (PCogAlign 계열 실패 양상)
- C 편향 → 선호 과다 반영 (Whose Boat?/SynthesizeMe 계열의 "기계적 FAE")

## One-Cycle Pipeline

```
1. profile_generator.py     → 50명 narrative profile 생성
2. query_generator.py       → 사용자당 20개 query 생성
3. option_planner.py        → 각 query에 대해 4 선지의 속성 조합 계산
4. image_collector.py       → Amazon Reviews 2023 + Google 검색으로 이미지 수집
5. label_verifier.py        → 자동 라벨링 + LLM judge ensemble
6. quality_audit.py         → vision-essentiality 검증
7. evaluator.py             → VLM 평가 실행
```

## 비용 정책

- **GPT-5-mini**: 유료 (profile/query 생성, judge ensemble 일부)
- **나머지 모두 무료**:
  - Local LLM judge (Qwen2.5-7B via vLLM, 사용자의 4× RTX 6000 Ada 활용)
  - CLIP/SigLIP 임베딩 (HuggingFace 무료)
  - 이미지 수집: Amazon Reviews 2023 (이미 다운로드된 데이터) + Google Custom Search (free tier 100 queries/day)
  - VLM 평가: vLLM으로 로컬 서빙 (Qwen2.5-VL, InternVL 등)

## 실행 순서

```bash
# Phase 1: 데이터 구축 (1회만)
python -m src.profile_generator --n_users 50 --domain_split 0.7
python -m src.query_generator --n_per_user 20
python -m src.option_planner
python -m src.image_collector
python -m src.label_verifier
python -m src.quality_audit

# Phase 2: 모델 평가
python -m src.evaluator --model qwen2.5-vl-7b
```

## 디렉토리

```
pod_bench/
├── src/                    # 파이프라인 코드
├── data/
│   ├── profiles/          # 생성된 user profiles
│   ├── queries/           # 생성된 queries
│   ├── options/           # 선지 속성 plan
│   ├── images/            # 수집된 이미지
│   ├── labels/            # 자동 라벨 + judge 결과
│   └── final/             # 최종 벤치마크 JSON
├── configs/                # 설정 파일
└── scripts/                # 실행 스크립트
```
