"""
common/clients.py — one OpenAI-compatible chat client for vLLM AND closed APIs.

Objective : identical request path for every provider so model comparisons are
            apples-to-apples. vLLM (:8002~) and OpenAI/Gemini OpenAI-compat
            endpoints differ only in base_url / api_key / guided support.
Controls baked in (closed-model blueprint):
  * uniform image preprocessing: every image resized to --image-max-side and
    sent as base64 JPEG (providers never see different resolutions),
  * temperature 0, model_id snapshot in manifest,
  * idempotent disk cache keyed by sha256(model, messages-with-image-hashes,
    params) -> reruns and retries are free,
  * retry with exponential backoff on 429/5xx/connection errors,
  * guided_choice passthrough for vLLM; closed models fall back to free-form
    text + common/parsing.py (parse failure rate is REPORTED, not hidden).
Input     : messages from common/prompts.py (image markers __IMG__<src>).
Output    : {"text", "cached", "error"} per call.
Metrics   : n/a (infrastructure); latency recorded per record.
Run       : imported. Smoke test against a live vLLM:
              python -c "from common.clients import ChatClient; ..."
"""
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import requests
from PIL import Image

from .prompts import IMG_MARK
from .parsing import extract_choice

_IMG_CACHE = {}
UA = {"User-Agent": "Mozilla/5.0 (compatible; POD-Bench-exp/1.0)"}


def encode_image(src, max_side=768):
    key = (src, max_side)
    if key in _IMG_CACHE:
        return _IMG_CACHE[key]
    if src.startswith("http://") or src.startswith("https://"):
        r = requests.get(src, timeout=20, headers=UA)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    else:
        img = Image.open(src).convert("RGB")
    if max(img.size) > max_side:
        sc = max_side / max(img.size)
        img = img.resize((round(img.size[0] * sc), round(img.size[1] * sc)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    if len(_IMG_CACHE) > 512:
        _IMG_CACHE.clear()
    _IMG_CACHE[key] = url
    return url


def _resolve(messages, max_side):
    out = []
    for m in messages:
        cont = []
        for b in m["content"]:
            if b.get("type") == "image_url" and \
               b["image_url"]["url"].startswith(IMG_MARK):
                src = b["image_url"]["url"][len(IMG_MARK):]
                cont.append({"type": "image_url",
                             "image_url": {"url": encode_image(src, max_side)}})
            else:
                cont.append(b)
        out.append({"role": m["role"], "content": cont})
    return out


def _cache_key(model, messages, params):
    """Image markers hashed by SRC STRING (identifies content, keeps key tiny)."""
    slim = json.dumps([model, messages, params], sort_keys=True,
                      ensure_ascii=False)
    return hashlib.sha256(slim.encode()).hexdigest()


class ChatClient:
    def __init__(self, base_url, model_id, api_key=None, guided=False,
                 cache_dir=None, max_tokens=16, temperature=0.0,
                 image_max_side=768, timeout=180, max_retries=4):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key = api_key
        self.guided = guided
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.image_max_side = image_max_side
        self.timeout = timeout
        self.max_retries = max_retries
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def chat(self, messages, allowed=None, task=None):
        params = {"max_tokens": self.max_tokens,
                  "temperature": self.temperature,
                  "guided": bool(self.guided and allowed),
                  "allowed": allowed if self.guided else None}
        key = _cache_key(self.model_id, messages, params)
        if self.cache_dir:
            p = self.cache_dir / f"{key}.json"
            if p.exists():
                out = json.loads(p.read_text())
                out["cached"] = True
                return out
        payload = {"model": self.model_id,
                   "messages": _resolve(messages, self.image_max_side),
                   "max_tokens": self.max_tokens,
                   "temperature": self.temperature}
        if self.guided and allowed:
            payload["guided_choice"] = list(allowed)     # vLLM extension
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(f"{self.base_url}/chat/completions",
                                  json=payload, headers=headers,
                                  timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"] or ""
                out = {"text": text, "cached": False, "error": None}
                if self.cache_dir:
                    (self.cache_dir / f"{key}.json").write_text(
                        json.dumps(out, ensure_ascii=False))
                return out
            except Exception as e:                        # noqa: BLE001
                last_err = str(e)[:200]
                time.sleep(min(2 ** attempt, 20))
        return {"text": "", "cached": False, "error": last_err}


class FakeClient:
    """Test/dry-run client. policy(task, messages, allowed) -> text.
    Lets tests simulate content-aware models (policy can read task meta)."""
    def __init__(self, policy=lambda task, m, a: ""):
        self.policy = policy
        self.calls = []

    def chat(self, messages, allowed=None, task=None):
        self.calls.append({"task": task, "allowed": allowed,
                           "messages": messages})
        return {"text": self.policy(task, messages, allowed),
                "cached": False, "error": None}


def execute_tasks(client, tasks, progress=True):
    """Shared eval loop. task = {**meta, 'messages', 'letters'}.
    Returns records = meta + {text, choice_display, parse_method, error,
    latency_s, cached}."""
    records = []
    n = len(tasks)
    for i, t in enumerate(tasks):
        meta = {k: v for k, v in t.items() if k not in ("messages",)}
        t0 = time.time()
        out = client.chat(t["messages"], allowed=list(t["letters"]), task=t)
        letter, method = extract_choice(out["text"], t["letters"])
        rec = dict(meta)
        rec.pop("letters", None)
        rec.update({"letters": list(t["letters"]), "text": out["text"],
                    "choice_display": letter, "parse_method": method,
                    "error": out.get("error"), "cached": out.get("cached"),
                    "latency_s": round(time.time() - t0, 3)})
        records.append(rec)
        if progress and (i + 1) % 25 == 0:
            print(f"    {i+1}/{n} done")
    return records
