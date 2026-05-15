# Setup Guide

## 0. 환경 준비

```bash
# Python 3.10+
pip install -r requirements.txt

# vLLM (별도 설치 — GPU 환경 필요)
pip install vllm
```

## 1. 환경 변수

```bash
# 필수 (GPT-5-mini용)
export OPENAI_API_KEY=sk-...

# 선택 (Google Custom Search — Amazon에서 못 찾으면 대체)
export GOOGLE_API_KEY=...
export GOOGLE_CSE_ID=...

# 선택 (Amazon Reviews 2023 메타데이터 경로)
export AMAZON_META_DIR=/path/to/amazon/data
```

Google Custom Search 설정:
1. https://console.cloud.google.com/ 에서 프로젝트 생성
2. Custom Search API 활성화
3. https://programmablesearchengine.google.com/ 에서 검색 엔진 생성
4. "Image search" 활성화, "Search the entire web" 선택
5. CSE ID와 API key 복사

무료 한도: 100 queries/day.

## 2. 로컬 vLLM 서버 (멀티 단계)

POD-Bench는 단계마다 다른 모델을 사용합니다. 단계별로 별도 터미널에서 띄우거나, 한 번에 띄울 수 있는 모델로 통합하세요.

### Option A: 단계별 띄우기 (저메모리)

```bash
# 단계 5 (label_verifier) 실행 전:
# 터미널 1: Qwen2.5-7B (port 8000)
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 --dtype bfloat16 \
    --gpu-memory-utilization 0.4

# 터미널 2: Llama-3.1-8B (port 8001)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8001 --dtype bfloat16 \
    --gpu-memory-utilization 0.4
```

```bash
# 단계 6 (quality_audit) / 평가 단계 전:
# label verifier 서버 종료 후
# Qwen2.5-VL-7B (멀티모달, port 8000)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --port 8000 --dtype bfloat16 \
    --limit-mm-per-prompt image=5 \
    --gpu-memory-utilization 0.8
```

### Option B: 4× RTX 6000 Ada 환경 (텐서 병렬)

```bash
# GPU 0,1: Qwen2.5-VL-7B (멀티모달, port 8000)
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --port 8000 --tensor-parallel-size 2 \
    --limit-mm-per-prompt image=5

# GPU 2: Qwen2.5-7B (텍스트, port 8001)
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8001

# GPU 3: Llama-3.1-8B (텍스트, port 8002)
CUDA_VISIBLE_DEVICES=3 vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8002
```

이 경우 `configs/config.py`의 `api_base`를 수정하세요:
```python
"local_judge":     {"api_base": "http://localhost:8001/v1", ...}
"local_judge_alt": {"api_base": "http://localhost:8002/v1", ...}
"vlm_evaluator":   {"api_base": "http://localhost:8000/v1", ...}
```

## 3. 파이프라인 실행

```bash
# 전체 파이프라인
bash scripts/run_pipeline.sh

# 또는 단계별
python -m src.profile_generator --n_users 50
python -m src.query_generator --n_per_user 20
python -m src.option_planner
python -m src.image_collector
python -m src.label_verifier
python -m src.quality_audit

# 평가
python -m src.evaluator --model vlm_evaluator
```

## 4. 디버그 / 부분 실행

각 단계 모듈은 `--limit N` 옵션을 지원합니다:
```bash
python -m src.option_planner --limit 10  # 10개만 처리
```

각 단계는 idempotent하게 작성되어 있어, 중간에 중단 후 다시 실행하면 이어서 진행합니다.
강제 재실행은 `--force` 옵션.
