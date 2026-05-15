"""
공통 유틸리티: LLM API 래퍼, JSON 파싱, 재시도 로직.

비용 정책:
  - GPT-5-mini: 유료 (call_gpt5_mini)
  - 로컬 vLLM: 무료 (call_local_llm)
"""

import json
import os
import re
import time
from typing import Optional
from pathlib import Path

import requests


# ─────────────────────────────────────────────
# GPT-5-mini (OpenAI API)
# ─────────────────────────────────────────────

def call_gpt5_mini(prompt: str, system: str = "", max_tokens: int = 2048,
                    temperature: float = 0.7, max_retries: int = 3) -> str:
    """GPT-5-mini API 호출.

    환경변수 OPENAI_API_KEY 필요.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable required")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "gpt-5-mini",
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload, headers=headers, timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  GPT-5-mini error (attempt {attempt+1}): {e}, retrying in {wait}s")
            time.sleep(wait)

    raise RuntimeError("Unreachable")


# ─────────────────────────────────────────────
# 로컬 vLLM (무료)
# ─────────────────────────────────────────────

def call_local_llm(prompt: str, system: str = "",
                    api_base: str = "http://localhost:8000/v1",
                    model: str = "Qwen/Qwen2.5-7B-Instruct",
                    max_tokens: int = 2048,
                    temperature: float = 0.3,
                    max_retries: int = 3) -> str:
    """로컬 vLLM 서버 호출.

    vLLM 서빙 명령:
      vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

    멀티모달 평가용:
      vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 \\
                  --limit-mm-per-prompt image=5
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    url = f"{api_base.rstrip('/')}/chat/completions"

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            # Qwen3의 <think> 블록 제거
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            return text
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError("Unreachable")


# ─────────────────────────────────────────────
# 멀티모달 로컬 호출 (이미지 + 텍스트)
# ─────────────────────────────────────────────

def call_local_vlm(prompt: str, image_paths: list[str], system: str = "",
                    api_base: str = "http://localhost:8000/v1",
                    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                    max_tokens: int = 512,
                    temperature: float = 0.0) -> str:
    """로컬 VLM 호출. 이미지를 base64로 인코딩하여 전송."""
    import base64

    content = []
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = f"{api_base.rstrip('/')}/chat/completions"

    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


# ─────────────────────────────────────────────
# JSON 파싱
# ─────────────────────────────────────────────

def parse_json_response(text: str) -> Optional[dict]:
    """LLM 출력에서 JSON 파싱. ```json``` 펜스, 주석 등 제거."""
    text = text.strip()

    # 코드 펜스 제거
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # 직접 파싱 시도
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # { ... } 추출 시도
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start:end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            # trailing comma 제거 시도
            snippet = re.sub(r",(\s*[}\]])", r"\1", snippet)
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None

    return None


def parse_json_list_response(text: str) -> Optional[list]:
    """LLM 출력에서 JSON 리스트 파싱."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

    return None


# ─────────────────────────────────────────────
# 파일 I/O
# ─────────────────────────────────────────────

def save_jsonl(records: list[dict], path: Path):
    """JSONL 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    """JSONL 로드."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(obj: dict, path: Path):
    """JSON 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    """JSON 로드."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Progress logging
# ─────────────────────────────────────────────

def log_step(msg: str):
    """파이프라인 단계 로그."""
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")
