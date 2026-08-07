"""STAGE D2 — 방식별 이미지 조합 선택 (가용성 제약).

이 모듈이 셀을 얻는 **유일한 통로**다. 후보는 사람이 블라인드 검수한
annotation/attribute_library.json 의 status=="available" 셀뿐이며,
images_final 을 glob 하지 않는다. 가용성을 사후 필터가 아니라 후보 제약으로
두는 것은 옵션 플래너 v5 와 같은 원칙이다 — 사후에 버리면 버림이 무작위가
아니어서 균형이 조용히 무너진다.

중립성(R3): evidence 축이 아닌 축의 값은 그 사용자에게 취향 중립이어야 한다.
싫어하는 색의 옷을 들고 "완전 내 스타일"이라 말하면 모순된 증거가 된다.
pattern 의 solid 는 무늬 축의 영(zero) 수준이므로 전원 중립으로 취급한다.
"""
from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path

from configs.config import FASHION_ATTRIBUTE_AXES
from dialog.expression import AXES


def load_available_cells(library_path: Path,
                         manifest_path: Path | None = None) -> set[str]:
    """후보 셀 = 검수 통과 ∩ 실제로 materialize 된 이미지.

    두 집합은 다르다. attribute_library 의 available 은 '사람이 이 크롭을
    승인했다'는 뜻이고, images_manifest 는 '그 이미지가 images/ 에 실제로
    복사되었다'는 뜻이다. v5 기준 각각 1,223 / 1,108 로 115개가 어긋난다.
    전자만 제약으로 쓰면 파일 없는 대화가 생성된다 — 승인과 존재를 같은
    것으로 취급한 데서 나오는 오류다.
    """
    lib = json.loads(Path(library_path).read_text())
    cells = {k for k, v in lib["cells"].items() if v.get("status") == "available"}
    if manifest_path is not None and Path(manifest_path).is_file():
        materialized = set(json.loads(Path(manifest_path).read_text())["cells"])
        cells &= materialized
    return cells


class ImagePicker:
    """한 사용자에 대한 셀 선택기."""

    # share 후보 풀 상한. 조합 탐색이 폭발하지 않게 자른다.
    SHARE_POOL_CAP = 14

    def __init__(self, profile: dict, cells: set[str], rng: random.Random):
        self.attrs = profile["structured_attributes"]
        self.cells = cells
        self.rng = rng

    # ── 중립성 ────────────────────────────────────────────────────────────
    def is_neutral(self, axis: str, value: str) -> bool:
        if axis == "pattern" and value == "solid":
            return True                      # 무늬 축의 영 수준
        a = self.attrs[axis]
        return value not in a["likes"] and value not in a["dislikes"]

    def neutral_values(self, axis: str, exclude: tuple = ()) -> list[str]:
        vals = [v for v in FASHION_ATTRIBUTE_AXES[axis]
                if self.is_neutral(axis, v) and v != "solid" and v not in exclude]
        self.rng.shuffle(vals)
        return vals

    # ── 셀 조회 ───────────────────────────────────────────────────────────
    def cell(self, color: str, garment: str, pattern: str) -> dict | None:
        if f"{color}|{garment}|{pattern}" not in self.cells:
            return None                      # available 아니면 후보에서 탈락
        return {"file": f"{color}/{pattern}_{garment}.jpg",
                "attributes": {"color": color, "garment_category": garment,
                               "pattern": pattern}}

    def _bases(self, axis: str, value: str, wide_background: bool = False):
        """evidence 축=value, 나머지는 사용자-중립인 셀 전부.

        wide_background 면 배경 무늬로 solid 뿐 아니라 중립 유무늬도 허용한다.
        R3 가 요구하는 것은 '그 사용자에게 취향 중립'이지 'solid' 가 아니므로
        둘 다 규칙을 만족한다. solid 로 못 박으면 후보가 절반 이하로 줄어
        선택지 셀 회피가 불가능해진다.
        """
        pats = ["solid"] + (self.neutral_values("pattern") if wide_background else [])
        opts = {"color": self.neutral_values("color"),
                "garment_category": self.neutral_values("garment_category"),
                "pattern": pats}
        opts[axis] = [value]
        for c in opts["color"]:
            for g in opts["garment_category"]:
                for p in opts["pattern"]:
                    cell = self.cell(c, g, p)
                    if cell:
                        yield cell

    # ── 방식별 조합 ───────────────────────────────────────────────────────
    def pick(self, expr_name: str, axis: str, value: str,
             wide_background: bool = False
             ) -> tuple[list[dict], int | None] | None:
        """(이미지 목록, evidence_index) 또는 실패 시 None."""
        if expr_name == "single":
            cell = next(self._bases(axis, value, wide_background), None)
            return ([cell], 0) if cell else None
        if expr_name == "differ":
            return self._differ(axis, value, wide_background)
        return self._share(axis, value, 2 if expr_name == "share2" else 3)

    def _differ(self, axis, value, wide_background: bool = False):
        """evidence 축만 다르고 나머지는 동일한 2장."""
        for base in self._bases(axis, value, wide_background):
            for alt in self.neutral_values(axis, exclude=(value,)):
                at = dict(base["attributes"]); at[axis] = alt
                partner = self.cell(at["color"], at["garment_category"],
                                    at["pattern"])
                if partner:
                    return [base, partner], 0     # 위치는 planner 가 뒤집는다
        return None

    def _share(self, axis, value, n):
        """evidence 축을 n장 전부가 공유하고, 다른 축은 전원 공유가 아닌 조합."""
        others = [a for a in AXES if a != axis]
        pool_vals = {}
        for a in others:
            pool_vals[a] = (["solid"] + self.neutral_values("pattern")
                            if a == "pattern" else self.neutral_values(a))
        candidates = []
        for v0 in pool_vals[others[0]]:
            for v1 in pool_vals[others[1]]:
                at = {axis: value, others[0]: v0, others[1]: v1}
                cell = self.cell(at["color"], at["garment_category"],
                                 at["pattern"])
                if cell:
                    candidates.append(cell)
        self.rng.shuffle(candidates)
        for combo in combinations(candidates[:self.SHARE_POOL_CAP], n):
            if all(len({c["attributes"][a] for c in combo}) > 1 for a in others):
                return list(combo), None
        return None