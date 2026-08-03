#!/usr/bin/env python3
"""대화 표면 다양화 — LLM 패러프레이즈 + 검증 루프.

LLM 의 권한은 **턴 텍스트뿐**이다. evidence·이미지·턴 수·화자 순서·시간은
계획 층(dialog_planner)이 확정했고 여기서 건드리지 않는다. 그래서 환각이
정답지를 오염시킬 통로가 구조적으로 없다.

    템플릿 대화 → 사용자별 목소리 파라미터로 패러프레이즈
                → 앵커 검사 + 누설 검사 + validate_dialogs 재실행
                → 통과하면 채택 / 실패하면 재시도(N회) / 최종 실패 시 템플릿 유지

폴백이 명시적인 것이 중요하다(작업 원칙 §8): 검증에 걸린 턴은 조용히 통과하지
않고 템플릿 문장으로 되돌아가며, 그 사실이 산출물과 리포트에 남는다.

text 쌍둥이는 패러프레이즈된 image 대화에서 다시 파생시킨다. 그래야 두 변형의
유일한 차이가 값의 모달리티라는 성질이 유지된다.

    POD_VARIANT=wacv_scenario_v5 python -m dialog.paraphrase \
        --model gpt-5-mini --api-key-env OPENAI_API_KEY
    python -m dialog.paraphrase --dry-run     # 프롬프트만 출력, 호출 없음
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.config import DATA_DIR                                # noqa: E402
from configs.dialog_templates import (AXIS_WORD, COUNT_WORD,        # noqa: E402
                                      ORDINAL)
from dialog.expression import EXPRESSIONS                           # noqa: E402
from dialog.image_picker import load_available_cells                # noqa: E402
from dialog.renderer import text_twin                               # noqa: E402
from dialog.validate_dialogs import find_leaks, validate            # noqa: E402

LIBRARY = ROOT / "annotation" / "attribute_library.json"

# 사용자마다 다른 목소리를 주는 파라미터. 시드 고정이라 재현된다.
VOICE_AXES = {
    "register": ["breathless and lowercase", "warm and chatty",
                 "dry and clipped", "precise and a little formal",
                 "wry, self-deprecating"],
    "punctuation": ["uses ellipses and dashes", "uses exclamation marks freely",
                    "minimal punctuation, short sentences",
                    "occasional ALL-CAPS for emphasis"],
    "recurring_cast": ["a younger sister who disagrees with everything",
                       "a roommate who lends things unasked",
                       "an aunt with unpredictable taste",
                       "a coworker who shops the same stores"],
    "verbosity": ["one short sentence per turn", "two sentences per turn",
                  "one long sentence with a clause"],
}

SYSTEM = """You rewrite chat messages to sound like a specific real person, \
without changing what they mean.

You will be given turns from a conversation between a user and an assistant, \
each with a list of ANCHORS: exact substrings that must appear verbatim in \
your rewrite. Anchors carry the meaning the conversation is built on.

Hard rules:
1. Keep every anchor verbatim. Do not pluralize, capitalize differently, or \
paraphrase an anchor.
2. Never name a specific colour, garment type, or pattern anywhere. The user \
refers to these only through the anchor words (e.g. "the color", "that cut"). \
Writing an actual colour name, garment name, or pattern name is a failure.
3. Preserve the claim structure exactly: if the original says two items differ \
in exactly one way, your rewrite must say the same. If it says several items \
share exactly one thing, keep that. If it states the user loves or avoids \
something, keep the same direction.
4. Do not mention occasions, events, dress codes, weather, or places where \
clothes would be worn.
5. Return JSON only: {"turns": [{"turn_id": <int>, "text": "<rewrite>"}, ...]} \
with one entry per input turn, same order. No commentary."""


def voice_for(user_id: str, seed: int) -> dict:
    rng = random.Random(f"{user_id}|voice|{seed}")
    return {k: rng.choice(v) for k, v in VOICE_AXES.items()}


def anchors_for(dialog: dict) -> dict[int, list[str]]:
    """턴별로 반드시 보존되어야 할 문자열.

    앵커는 **실제 문장에 들어 있는 것만** 요구한다. 템플릿 변형마다 표현이
    달라(어떤 share 문장은 "share" 라는 단어를 쓰지 않는다) 고정 목록을 쓰면
    존재하지도 않는 문자열을 요구하게 되고, 모든 턴이 검증에 걸려 템플릿으로
    폴백해 버린다. 그래서 원문을 보고 결정한다.

    재시도 라운드에서는 일부 턴만 넘어오므로 없는 턴은 건너뛴다.
    """
    out: dict[int, list[str]] = {t["turn_id"]: [] for t in dialog["turns"]}
    text = {t["turn_id"]: t["text"].lower() for t in dialog["turns"]}

    def add(tid, vals):
        if tid not in text:
            return
        out.setdefault(tid, []).extend(
            v for v in vals
            if re.search(rf"\b{re.escape(v.lower())}s?\b", text[tid]))

    for e in dialog["evidence"]:
        expr = EXPRESSIONS[e["expression_type"]]
        a = AXIS_WORD[e["axis"]]
        keep = [a]
        if expr.family == "differ":
            keep.append(ORDINAL[e["evidence_index"]])
        elif expr.family == "share":
            keep += [COUNT_WORD[expr.n_images], "share", "shared"]
        turn = e["revealed_in_turn"]
        add(turn, keep)
        add(turn + 1, [a])            # 어시스턴트 확인 턴도 축을 재확정한다
    return out


def build_prompt(dialog: dict, voice: dict) -> list[dict]:
    anchors = anchors_for(dialog)
    turns = [{"turn_id": t["turn_id"], "speaker": t["speaker"],
              "original": t["text"],
              "anchors": sorted(set(anchors[t["turn_id"]])),
              "has_images": bool(t["images"])}
             for t in dialog["turns"]]
    persona = "\n".join(f"- {k}: {v}" for k, v in voice.items())
    user = (f"The user's voice:\n{persona}\n\n"
            "The assistant stays consistent: complete sentences, a little "
            "drier than the user, never repeats the same acknowledgement "
            "twice.\n\n"
            "Rewrite every turn below. Vary the wording across turns — the "
            "originals are templated and repetitive, and that is what you are "
            "fixing.\n\n"
            + json.dumps({"turns": turns}, ensure_ascii=False, indent=1))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


def check_turn(text: str, original: str, anchors: list[str],
               is_image_variant: bool) -> list[str]:
    """앵커 보존 + 어휘 누설. validate_dialogs 의 턴 단위 선행 검사."""
    errs = []
    low = text.lower()
    for a in anchors:
        # 복수형 허용 ("two cuts" 도 cut 앵커를 만족)
        if not re.search(rf"\b{re.escape(a.lower())}s?\b", low):
            errs.append(f"anchor missing: {a!r}")
    if is_image_variant:
        hits = find_leaks(text)
        if hits:
            errs.append(f"vocab leak: {hits}")
    if len(text.split()) > max(60, len(original.split()) * 3):
        errs.append("far longer than original")
    if not text.strip():
        errs.append("empty")
    return errs


def call_model(session, url, model, messages, api_key, timeout, retries):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {"model": model, "messages": messages}
    if model.split("/")[-1].startswith(("gpt-5", "o1", "o3", "o4")):
        payload["max_completion_tokens"] = 8192
        payload["reasoning_effort"] = "minimal"
    else:
        payload["max_tokens"] = 8192
        payload["temperature"] = 1.0        # 다양성이 목적이므로 0 이 아니다
    last = None
    for attempt in range(retries):
        try:
            r = session.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            resp = getattr(e, "response", None)
            if resp is not None:
                last = RuntimeError(f"{e} :: {resp.text[:300]}")
                if resp.status_code in (400, 404, 422):
                    break
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:300]}")


def parse_turns(raw: str) -> dict[int, str]:
    body = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(),
                  flags=re.MULTILINE).strip()
    data = json.loads(body)
    return {int(t["turn_id"]): t["text"].strip() for t in data["turns"]}


def paraphrase_dialog(dialog, voice, caller, max_rounds=3):
    """턴별로 통과한 것만 채택하고, 실패한 턴만 다시 요청한다."""
    anchors = anchors_for(dialog)
    accepted: dict[int, str] = {}
    report = {"rounds": 0, "fallback_turns": [], "errors": []}
    pending = dialog

    for rnd in range(max_rounds):
        report["rounds"] = rnd + 1
        todo = [t for t in dialog["turns"] if t["turn_id"] not in accepted]
        if not todo:
            break
        sub = dict(dialog, turns=todo)
        try:
            got = parse_turns(caller(build_prompt(sub, voice)))
        except Exception as e:
            report["errors"].append(f"round {rnd + 1}: {type(e).__name__}: {e}")
            break
        for t in todo:
            text = got.get(t["turn_id"])
            if text is None:
                continue
            errs = check_turn(text, t["text"], sorted(set(anchors[t["turn_id"]])),
                              dialog["variant"] == "image")
            if not errs:
                accepted[t["turn_id"]] = text
            elif rnd == max_rounds - 1:
                report["errors"].append(f"turn {t['turn_id']}: {errs[0]}")

    out = json.loads(json.dumps(dialog))
    for t in out["turns"]:
        if t["turn_id"] in accepted:
            t["text"] = accepted[t["turn_id"]]
        else:
            report["fallback_turns"].append(t["turn_id"])
    out["paraphrase"] = report
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogs", type=Path, default=DATA_DIR / "dialogs")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "dialogs_paraphrased")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--users", nargs="*", default=None)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="프롬프트만 출력하고 호출하지 않는다")
    args = ap.parse_args()

    profiles = {}
    with open(DATA_DIR / "profiles" / "profiles.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            profiles[r["user_id"]] = r
    cells = load_available_cells(LIBRARY, DATA_DIR / "images_manifest.json")

    files = sorted(args.dialogs.glob("dlg__*__image.json"))
    if args.users:
        files = [f for f in files if any(u in f.name for u in args.users)]
    if not files:
        raise SystemExit(f"no dialogs in {args.dialogs}")

    if args.dry_run:
        d = json.loads(files[0].read_text())
        msgs = build_prompt(d, voice_for(d["user_id"], args.seed))
        print(msgs[0]["content"])
        print("\n" + "=" * 60 + "\n")
        print(msgs[1]["content"][:2500])
        return 0

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise SystemExit(f"[error] ${args.api_key_env} is empty")
    session = requests.Session()
    url = args.base_url.rstrip("/") + "/chat/completions"
    caller = lambda msgs: call_model(session, url, args.model, msgs, api_key,
                                     args.timeout, 3)

    args.out.mkdir(parents=True, exist_ok=True)
    n_fb = n_turn = 0
    failed = 0
    for path in files:
        d = json.loads(path.read_text())
        voice = voice_for(d["user_id"], args.seed)
        new = paraphrase_dialog(d, voice, caller, args.max_rounds)
        errs = validate(new, profiles[d["user_id"]], cells)
        if errs:
            failed += 1
            print(f"  ! {d['dialog_id']}: {errs[0]}  → 템플릿 유지")
            new = d
        twin = text_twin(new)
        for out in (new, twin):
            (args.out / f"{out['dialog_id']}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=1))
        fb = len(new.get("paraphrase", {}).get("fallback_turns", []))
        n_fb += fb
        n_turn += len(new["turns"])
        print(f"  {d['user_id']}  rounds={new.get('paraphrase', {}).get('rounds', 0)}"
              f"  fallback={fb}/{len(new['turns'])}"
              + ("  voice=" + voice["register"] if fb == 0 else ""))

    print(f"\n{len(files)} dialogs  템플릿 폴백 {n_fb}/{n_turn} 턴 "
          f"({n_fb / max(n_turn, 1):.1%})  검증 실패 {failed}건")
    print("검증: python -m dialog.validate_dialogs "
          f"--dialogs {args.out} --self-test")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())