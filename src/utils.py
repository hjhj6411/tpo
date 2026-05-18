"""
POD-Bench utilities.

GPT-5 reasoning model API notes (critical):
  - Parameter is `max_completion_tokens` (not `max_tokens`).
  - Reasoning tokens count toward `max_completion_tokens`. Reasoning alone
    typically consumes 300-500 tokens, so set ≥ 4096 even for short outputs.
  - `temperature` is NOT supported (only default 1.0). Passing custom values
    returns 400 error.
  - Response parsing: use the Chat Completions endpoint
    (`/v1/chat/completions`) which returns
        {"choices": [{"message": {"content": "..."}}]}
    The Responses API (`/v1/responses`) has a different structure and is
    NOT used here to keep parsing simple.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests


# ─────────────────────────────────────────────
# GPT-5-mini (paid)
# ─────────────────────────────────────────────

def call_gpt5_mini(prompt: str, system: str = "",
                    max_completion_tokens: int = 4096,
                    max_retries: int = 3) -> str:
    """Call GPT-5-mini via Chat Completions endpoint.

    Parameters
    ----------
    prompt : str
        User message.
    system : str
        Optional system message.
    max_completion_tokens : int
        Must include budget for reasoning tokens. Default 4096 is the safe
        floor (covers 300-500 reasoning + up to 3500 output tokens).
    max_retries : int

    Returns
    -------
    str
        Response text. Empty string if the model returned no content
        (e.g., reasoning ran out of token budget before producing output).
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
        "max_completion_tokens": max_completion_tokens,
        # NOTE: do NOT pass `temperature` — gpt-5 family rejects non-default values.
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload, headers=headers, timeout=180,
            )
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                # 4xx errors: don't retry except 429
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise RuntimeError(last_err)
                time.sleep(2 ** attempt)
                continue

            data = resp.json()
            return _extract_chat_completion_content(data)

        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
        except Exception as e:
            last_err = str(e)
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError(f"GPT-5-mini call failed after {max_retries} retries: {last_err}")


def _extract_chat_completion_content(data: dict) -> str:
    """Safely extract content from Chat Completions response.

    Handles:
      - normal response with content
      - empty content (reasoning consumed all tokens)
      - missing 'content' key
      - refusal in content
    """
    choices = data.get("choices") or []
    if not choices:
        return ""

    msg = choices[0].get("message") or {}

    # Refusal field (sometimes present instead of content)
    if msg.get("refusal"):
        return ""

    content = msg.get("content")
    if content is None:
        # finish_reason == 'length' often means reasoning hit cap
        finish = choices[0].get("finish_reason", "")
        if finish == "length":
            return ""  # caller should retry with higher max_completion_tokens
        return ""

    # content may be a list of parts in some API versions
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text") or part.get("content") or ""
                if t:
                    texts.append(t)
            elif isinstance(part, str):
                texts.append(part)
        return "".join(texts)

    return str(content) if content else ""


# ─────────────────────────────────────────────
# Local vLLM (free)
# ─────────────────────────────────────────────

def call_local_llm(prompt: str, system: str = "",
                    api_base: str = "http://localhost:8000/v1",
                    model: str = "Qwen/Qwen2.5-7B-Instruct",
                    max_tokens: int = 2048,
                    temperature: float = 0.3,
                    max_retries: int = 3) -> str:
    """Call a locally-served vLLM model (OpenAI-compatible)."""
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

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            text = _extract_chat_completion_content(resp.json())
            # Strip Qwen3 <think> block if present
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            return text
        except Exception as e:
            last_err = str(e)
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Local LLM call failed: {last_err}")


def call_local_vlm(prompt: str, image_paths: list[str], system: str = "",
                    api_base: str = "http://localhost:8000/v1",
                    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                    max_tokens: int = 512,
                    temperature: float = 0.0) -> str:
    """Call a local VLM with image(s) + text."""
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
    text = _extract_chat_completion_content(resp.json())
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


# ─────────────────────────────────────────────
# JSON parsing helpers
# ─────────────────────────────────────────────

def parse_json_response(text: str) -> Optional[dict]:
    """Extract JSON object from LLM output, tolerant to code fences."""
    if not text:
        return None
    text = text.strip()

    # strip code fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # bracketed
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start:end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            # trailing comma fix
            snippet = re.sub(r",(\s*[}\]])", r"\1", snippet)
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None
    return None


def parse_json_list_response(text: str) -> Optional[list]:
    """Extract JSON array from LLM output."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

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
# File I/O
# ─────────────────────────────────────────────

def save_jsonl(records: list, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def log_step(msg: str):
    print(f"\n{'='*64}\n  {msg}\n{'='*64}")
