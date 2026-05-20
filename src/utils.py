"""
POD-Bench utilities — unified provider abstraction.

Single entry points:
  - call_llm(prompt, stage=..., ...)        → text LLM (gpt5_mini OR vllm)
  - call_vlm(prompt, images, stage=..., ...) → multimodal VLM

PROVIDERS in configs/config.py maps each pipeline stage to a provider name,
so flipping a stage from free vllm to paid gpt5_mini requires no code change.

GPT-5 reasoning model quirks (handled here):
  - 'max_completion_tokens' (not 'max_tokens')
  - reasoning tokens count toward the cap; default 4096
  - 'temperature' is not supported; we omit it
  - Defensive content extraction handles empty content / refusals
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import PROVIDERS, PROVIDER_ENDPOINTS


def resolve_provider(stage: str, override: Optional[str] = None) -> dict:
    if override:
        provider_name = override
    else:
        if stage not in PROVIDERS:
            raise ValueError(f"Unknown stage '{stage}'. Add to PROVIDERS.")
        provider_name = PROVIDERS[stage]["provider"]
    if provider_name not in PROVIDER_ENDPOINTS:
        raise ValueError(f"Unknown provider '{provider_name}'.")
    return PROVIDER_ENDPOINTS[provider_name] | {"_name": provider_name}


def call_llm(prompt: str, stage: str, system: str = "",
              max_tokens: Optional[int] = None, temperature: float = 0.3,
              max_retries: int = 3, provider_override: Optional[str] = None) -> str:
    endpoint = resolve_provider(stage, provider_override)
    return _call_openai_compatible(
        endpoint=endpoint, prompt=prompt, system=system, images=None,
        max_tokens=max_tokens, temperature=temperature, max_retries=max_retries,
    )


def call_vlm(prompt: str, image_paths: list, stage: str, system: str = "",
              max_tokens: Optional[int] = None, temperature: float = 0.0,
              max_retries: int = 3, provider_override: Optional[str] = None) -> str:
    endpoint = resolve_provider(stage, provider_override)
    return _call_openai_compatible(
        endpoint=endpoint, prompt=prompt, system=system, images=image_paths,
        max_tokens=max_tokens, temperature=temperature, max_retries=max_retries,
    )


def _call_openai_compatible(endpoint, prompt, system="", images=None,
                              max_tokens=None, temperature=0.3, max_retries=3):
    kind = endpoint.get("kind", "openai_compat")

    if images:
        import base64
        content = []
        for img_path in images:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({"type": "image_url",
                              "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        user_msg = {"role": "user", "content": content}
    else:
        user_msg = {"role": "user", "content": prompt}

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(user_msg)

    if max_tokens is None:
        max_tokens = endpoint.get("default_max_tokens", 2048)

    payload = {"model": endpoint["model_name"], "messages": messages}
    if endpoint.get("uses_max_completion_tokens"):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens

    if endpoint.get("supports_temperature", True):
        payload["temperature"] = temperature

    headers = {"Content-Type": "application/json"}
    if kind == "openai_api":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY required for gpt-5 stages")
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{endpoint['api_base'].rstrip('/')}/chat/completions"

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=180)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise RuntimeError(last_err)
                time.sleep(2 ** attempt)
                continue
            text = _extract_content(resp.json())
            if text and "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            return text
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
        except Exception as e:
            last_err = str(e)
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError(f"LLM call failed after {max_retries} retries "
                        f"({endpoint['_name']}): {last_err}")


def _extract_content(data):
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    if msg.get("refusal"):
        return ""
    content = msg.get("content")
    if content is None:
        return ""
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


def parse_json_response(text):
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
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start:end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            snippet = re.sub(r",(\s*[}\]])", r"\1", snippet)
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None
    return None


def parse_json_list_response(text):
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


def save_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
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


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def log_step(msg):
    print(f"\n{'='*64}\n  {msg}\n{'='*64}")
