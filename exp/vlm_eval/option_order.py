"""선택지 표시 순서 결정 — 세 평가 스크립트가 공유한다.

eval_multimodal / eval_text / eval_dialog 이 같은 규칙을 써야 문항 단위
짝비교가 성립한다. 그래서 순서 결정을 여기 한 곳에 모은다.

세 가지 방식이 있다.

  balanced (기본)
      정답 위치를 정확히 4등분한다. 전체 plan 목록을 시드로 섞은 뒤 순서대로
      0,1,2,3 슬롯을 배정하고, 나머지 세 선택지만 문항별로 섞는다.
      위치별 정확도를 보고할 때 각 위치의 n 이 같아져 비교가 깔끔하다.

  random
      문항마다 독립적으로 셔플한다. 정답 위치가 이항분포를 따라 완전히
      균등하지는 않다 (2,550문항 실측: A 1,252 / B 1,330 / C 1,318 / D 1,200,
      chi^2=4.3, p=0.23 — 균등과 통계적으로 구별되지는 않는다).

  fixed:A | fixed:B | fixed:C | fixed:D
      정답을 항상 그 위치에 놓는다. 위치 편향을 단독으로 재는 진단용이며,
      본 실험에 쓰면 정답 위치가 상수가 되어 결과가 무의미해진다.

배정은 **파일에 있는 전체 plan 목록** 위에서 계산한다. --limit 이나 --track
으로 부분집합만 평가해도 각 문항의 배치가 달라지지 않게 하기 위해서다.

시드는 항상 필요하다. balanced 는 슬롯 배정 순서를, random 은 셔플을,
fixed 는 나머지 세 선택지의 배열을 시드로 정한다.
"""
from __future__ import annotations

import random

LABELS = ["A", "B", "C", "D"]
MODES = ["balanced", "random", "fixed:A", "fixed:B", "fixed:C", "fixed:D"]


def assign_orders(all_plan_ids, seed: int, mode: str = "balanced"
                  ) -> dict[str, list[str]]:
    """plan_id -> 표시 순서대로 나열한 원본 키 목록.

    반환값 예: {"q0001__...": ["C", "A", "D", "B"]}
    → 화면의 A 에 원본 C 가, B 에 원본 A 가 놓인다.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")

    orders: dict[str, list[str]] = {}

    if mode == "random":
        for pid in all_plan_ids:
            rng = random.Random(f"{seed}|{pid}")
            keys = list(LABELS)
            rng.shuffle(keys)
            orders[pid] = keys
        return orders

    if mode.startswith("fixed:"):
        slot = LABELS.index(mode.split(":")[1])
        for pid in all_plan_ids:
            rng = random.Random(f"{seed}|{pid}")
            others = ["B", "C", "D"]
            rng.shuffle(others)
            orders[pid] = others[:slot] + ["A"] + others[slot:]
        return orders

    # balanced: 정답(원본 A)이 놓일 슬롯을 정확히 4등분한다.
    ids = sorted(all_plan_ids)
    random.Random(f"{seed}|balance").shuffle(ids)
    for i, pid in enumerate(ids):
        slot = i % 4
        rng = random.Random(f"{seed}|{pid}")
        others = ["B", "C", "D"]
        rng.shuffle(others)
        orders[pid] = others[:slot] + ["A"] + others[slot:]
    return orders


def correct_display(order: list[str]) -> str:
    """원본 A 가 놓인 화면 위치."""
    return LABELS[order.index("A")]


def display_to_original(order: list[str]) -> dict[str, str]:
    return {d: k for d, k in zip(LABELS, order)}