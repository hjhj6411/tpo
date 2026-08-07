"""STAGE D3 — 대화 계획 (결정론적, 문장 없음).

프로필의 취향 18개(like 9 + dislike 9)를 **전부** 노출하도록 계획한다.
narrative/KV 조건이 18개를 모두 제공하므로, 대화 조건도 전부 실어야 정보량이
같아지고 조건 비교가 공정해진다.

이 층은 문장을 만들지 않는다. (evidence, 방식, 이미지, 에피소드, 위치)까지만
정하고 렌더링은 renderer.py 가 맡는다. 그래야 image 변형과 text 쌍둥이가
**같은 계획**에서 나와 구조 동일성이 보장된다.

커버리지는 경성 제약, 방식 쿼터는 연성 목표다. 셀이 없어 요청 방식이
불가능하면 DOWNGRADE_ORDER 를 따라 완화하고 downgrade 를 기록한다 —
조용한 폴백을 만들지 않는다(작업 원칙 §8).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from dialog.expression import (AXES, DOWNGRADE_ORDER, EXPRESSIONS, TYPE_QUOTA)
from dialog.image_picker import ImagePicker


@dataclass
class EvidencePlan:
    axis: str
    value: str
    polarity: str                 # "like" | "dislike"
    expression_type: str          # 실제 사용된 방식 (downgrade 후)
    requested_type: str           # 원래 배정된 방식
    images: list[dict]            # [{file, attributes}, ...]
    evidence_index: int | None    # differ/single 에서 evidence 이미지 위치
    episode_id: str
    recur_fallback: bool = False  # 회피 집합에서 못 찾아 겹치는 셀을 쓴 경우


@dataclass
class DialogPlan:
    user_id: str
    seed: int
    evidence: list[EvidencePlan] = field(default_factory=list)
    episodes: list[str] = field(default_factory=list)
    downgrades: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    recur_fallbacks: list[str] = field(default_factory=list)


def all_preferences(profile: dict) -> list[tuple[str, str, str]]:
    """(축, 값, 극성) 18개. 프로필 쿼터: 의류 4+4, 색 3+3, 무늬 2+2."""
    attrs = profile["structured_attributes"]
    return [(ax, v, pol)
            for ax in AXES
            for pol in ("like", "dislike")
            for v in attrs[ax][pol + "s"]]


def plan_dialog(profile: dict, cells: set[str], episodes: list[str],
                seed: int, avoid: set[str] | None = None,
                avoid_mode: str = "prefer") -> DialogPlan:
    """avoid: 이 사용자의 문항 선택지가 쓰는 셀. 대화 이미지가 선택지로
    재등장하면 과제가 시각적 재인으로 바뀌므로 피한다.

    avoid_mode
      off     회피하지 않는다 (기존 동작)
      prefer  회피 집합 밖에서 먼저 찾고, 불가능하면 겹침을 허용하되 기록한다
      strict  겹침을 절대 허용하지 않는다. 못 채우면 failures 에 남는다

    single/differ 는 배경축이 좁아 회피 시 조합이 사라지는 취향이 생긴다
    (실측: 24명 중 15명이 18개 중 1~3개). 그래서 기본은 prefer 다 —
    커버리지를 지키고 어긋난 건수를 드러내는 쪽이 조용한 결손보다 낫다.
    """
    rng = random.Random(f"{profile['user_id']}|dialog|{seed}")
    avoid = avoid or set()
    if avoid_mode == "off" or not avoid:
        primary = ImagePicker(profile, cells, rng)
        fallback = None
    else:
        primary = ImagePicker(profile, cells - avoid, rng)
        fallback = None if avoid_mode == "strict" else \
            ImagePicker(profile, cells, rng)

    slots = all_preferences(profile)
    if len(slots) != 18:
        raise ValueError(f"{profile['user_id']}: expected 18 preferences, "
                         f"got {len(slots)}")
    rng.shuffle(slots)

    requested = [t for t, n in TYPE_QUOTA.items() for _ in range(n)]
    rng.shuffle(requested)

    plan = DialogPlan(user_id=profile["user_id"], seed=seed, episodes=episodes)
    flip = 0                       # differ 위치 카운터밸런스

    # 에피소드에 라운드로빈 분배 (에피소드당 3개, 6개 에피소드 기준)
    buckets = [slots[i::len(episodes)] for i in range(len(episodes))]
    req_iter = iter(requested)

    for ep_id, bucket in zip(episodes, buckets):
        for axis, value, polarity in bucket:
            want = next(req_iter)
            order = [want] + [t for t in DOWNGRADE_ORDER if t != want]
            chosen = None
            # 1) 좁은 배경(solid) → 2) 넓은 배경(중립 유무늬 포함) 순으로
            #    회피 집합 밖에서 찾는다. 넓은 배경도 R3 를 만족한다.
            for wide in (False, True):
                for ty in order:
                    got = primary.pick(ty, axis, value, wide_background=wide)
                    if got:
                        chosen = (ty, *got, False)
                        break
                if chosen:
                    break
            if chosen is None and fallback is not None:
                for ty in order:                   # 그래도 없으면 겹침 허용
                    got = fallback.pick(ty, axis, value, wide_background=True)
                    if got:
                        chosen = (ty, *got, True)
                        plan.recur_fallbacks.append(
                            f"{axis}={value}({polarity}): {ty} with option cell")
                        break
            if chosen is None:
                plan.failures.append(
                    f"{axis}={value}({polarity}): no cells for any expression")
                continue
            ty, images, ev_idx, fell_back = chosen
            if EXPRESSIONS[ty].name == "differ":
                ev_idx = flip % 2                 # 위치 카운터밸런스
                flip += 1
                if ev_idx == 1:
                    images = [images[1], images[0]]
            if ty != want:
                plan.downgrades.append(
                    f"{axis}={value}({polarity}): {want} -> {ty}")
            plan.evidence.append(EvidencePlan(
                axis=axis, value=value, polarity=polarity,
                expression_type=ty, requested_type=want,
                images=images, evidence_index=ev_idx, episode_id=ep_id,
                recur_fallback=fell_back))
    return plan