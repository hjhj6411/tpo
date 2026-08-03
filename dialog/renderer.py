"""STAGE D4 — 계획 → 턴 렌더링 (image 변형 + text 쌍둥이).

두 변형이 **같은 DialogPlan** 에서 나온다. 그래서 턴 수·순서·시간·evidence
배치가 완전히 동일하고, 두 조건의 유일한 차이는 '값이 이미지에 있는가,
텍스트에 있는가'가 된다. 통제 비교가 구조적으로 성립하는 지점이다.

text 쌍둥이는 이미지 자리에 '{blue solid hoodie}' 자리표시자를 넣는다.
문장 자체는 손대지 않으므로 축 힌트("the shared color")도 그대로 유지된다.

LLM 패러프레이즈를 도입한다면 이 모듈에만 들어간다. 계획 층은 그대로 두고
문장만 다시 쓰되, 채택 전 validate_dialogs 를 통과해야 한다.
"""
from __future__ import annotations

import json

import random

from configs.dialog_templates import (AXIS_KIND, AXIS_WORD, COUNT_WORD,
                                      DIALOG_TEMPLATE_VERSION, EPISODES,
                                      EPISODE_CONNECTORS, ORDINAL, ORTHOGRAPHY,
                                      UTTERANCES)
from dialog.dialog_planner import DialogPlan
from dialog.expression import EXPRESSIONS

OPENERS = {ep_id: (opener, ack) for ep_id, opener, ack in EPISODES}


def word(value: str) -> str:
    return ORTHOGRAPHY.get(value, str(value).replace("_", " "))


def describe(attrs: dict) -> str:
    """text 쌍둥이용 자리표시자 — 이미지 내용의 충실한 서술."""
    return ("{" + f"{word(attrs['color'])} {word(attrs['pattern'])} "
            f"{word(attrs['garment_category'])}" + "}")


class VariantPicker:
    """키별로 변형을 소진 순서대로 꺼낸다 → 대화 내 문장 반복 0.

    한 대화에서 같은 키가 최대 4회 쓰이고(실측) 키마다 변형이 4개이므로,
    소진식으로 꺼내면 중복이 구조적으로 발생하지 않는다. 풀 순서는 사용자
    시드로 섞여 사용자마다 다른 조합이 나온다.
    """

    def __init__(self, seed, user_id):
        self.rng = random.Random(f"{user_id}|variants|{seed}")
        self.order: dict[tuple, list[int]] = {}
        self.used: dict[tuple, int] = {}

    def pick(self, key):
        pool = UTTERANCES[key]
        if key not in self.order:
            idx = list(range(len(pool)))
            self.rng.shuffle(idx)
            self.order[key] = idx
            self.used[key] = 0
        i = self.used[key]
        self.used[key] += 1
        return pool[self.order[key][i % len(pool)]]


def _utterances(ev, picker) -> tuple[str, str]:
    """축 종류에 따라 문장이 갈린다.

    의류 축의 differ 는 두 이미지가 서로 다른 옷이므로 "같은 물건 두 개" 라고
    쓸 수 없다. 그렇게 쓰면 문장이 이미지와 어긋나 모델이 둘 중 하나를 버리게
    되고, 그 결과는 모델 능력이 아니라 데이터 결함을 측정한 값이 된다.
    """
    expr = EXPRESSIONS[ev.expression_type]
    kind = AXIS_KIND[ev.axis]
    user, ack = picker.pick((expr.family, kind, ev.polarity))
    fmt = {"A": AXIS_WORD[ev.axis]}
    if expr.family == "differ":
        fmt["P"] = ORDINAL[ev.evidence_index]
    elif expr.family == "share":
        fmt["N"] = COUNT_WORD[expr.n_images]
    return user.format(**fmt), ack.format(**fmt)


def render(plan: DialogPlan) -> dict:
    """image 변형 대화를 만든다."""
    turns, evidence = [], []
    tid = day = 0
    picker = VariantPicker(plan.seed, plan.user_id)
    conn_rng = random.Random(f"{plan.user_id}|connectors|{plan.seed}")

    def add(speaker, text, images=None):
        nonlocal tid
        turns.append({"turn_id": tid, "speaker": speaker,
                      "time": f"2026-07-{6 + day:02d} 19:{10 + tid % 45:02d}",
                      "text": text, "images": images or []})
        tid += 1

    by_episode = {}
    for ev in plan.evidence:
        by_episode.setdefault(ev.episode_id, []).append(ev)

    for ep_id in plan.episodes:
        evs = by_episode.get(ep_id)
        if not evs:
            continue
        opener, ack = OPENERS[ep_id]
        add("user", opener)
        add("assistant", ack)
        for i, ev in enumerate(evs):
            user_text, ack_text = _utterances(ev, picker)
            if i:      # 에피소드의 두 번째 이후 턴에 담화 연결어를 붙인다
                pool = EPISODE_CONNECTORS.get(ep_id) or []
                if pool:
                    user_text = f"{pool[(i - 1) % len(pool)]} {user_text}"
            add("user", user_text, [c["file"] for c in ev.images])
            evidence.append({
                "axis": ev.axis, "value": ev.value, "polarity": ev.polarity,
                "expression_type": ev.expression_type,
                "requested_type": ev.requested_type,
                "evidence_index": ev.evidence_index,
                "episode_id": ev.episode_id,
                "revealed_in_turn": tid - 1, "modality": "image",
                "image_attributes": [c["attributes"] for c in ev.images],
            })
            add("assistant", ack_text)
        day += 2

    return {
        "dialog_id": f"dlg__{plan.user_id}__image",
        "user_id": plan.user_id,
        "variant": "image",
        "seed": plan.seed,
        "template_version": DIALOG_TEMPLATE_VERSION,
        "downgrades": plan.downgrades,
        "evidence": evidence,
        "turns": turns,
    }


def text_twin(dialog: dict) -> dict:
    """이미지를 자리표시자로 치환한 쌍둥이. 그 외 모든 것이 동일하다."""
    twin = json.loads(json.dumps(dialog))
    twin["dialog_id"] = twin["dialog_id"].replace("__image", "__text")
    twin["variant"] = "text"
    by_turn = {e["revealed_in_turn"]: e for e in twin["evidence"]}
    for turn in twin["turns"]:
        if turn["images"]:
            attrs = by_turn[turn["turn_id"]]["image_attributes"]
            turn["text"] += "  " + "  ".join(describe(a) for a in attrs)
            turn["images"] = []
    for e in twin["evidence"]:
        e["modality"] = "text"
    return twin