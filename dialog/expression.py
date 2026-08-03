"""STAGE D1 — 표현 방식 정의 (단일 진실 공급원).

취향 사실 하나는 (축, 값, 극성) 셋으로 이루어진다. 표현 방식은 이 셋을
텍스트와 이미지에 어떻게 나눠 싣느냐를 정한다. image 변형에서는 **값이 항상
이미지에만** 있고, 축과 극성은 텍스트가 나른다.

    single  1장   그 축의 값을 읽으면 됨
    differ  2장   evidence 축만 다름. 텍스트가 어느 쪽인지 지정(서수)
    share2  2장   evidence 축만 공유. 두 장을 비교해 공통값을 찾아야 함
    share3  3장   같은 규칙, 세 장. 교집합이 더 좁혀짐

생성기(dialog_planner)와 검증기(validate_dialogs)가 **같은 정의를 읽는다.**
새 방식을 추가할 때 여기만 고치면 양쪽이 함께 따라온다.

share 의 연역 조건에 대한 주의:
  다른 축은 "쌍마다 달라야" 하는 것이 아니라 "전부가 공유하지는 않아야"
  한다. 색이 evidence 일 때 사용자-중립 무늬가 solid + 1종뿐이라(2+2가 취향
  으로 소진) 3장에 서로 다른 무늬 3개를 줄 수 없기 때문이다. solid/solid/
  striped 면 무늬는 공통 원소가 아니므로 '정확히 하나만 공유'는 유지된다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

AXES = ["garment_category", "color", "pattern"]

# 18개 취향에 배정할 방식 쿼터. 합이 18이어야 한다.
# 커버리지(18개 전부 노출)는 경성 제약, 이 쿼터는 연성 목표다 —
# 셀이 없으면 dialog_planner 가 단계적으로 완화하고 downgrade 를 기록한다.
TYPE_QUOTA = {"single": 6, "differ": 6, "share2": 3, "share3": 3}

# 완화 순서: 요구 이미지 수가 많은 쪽에서 적은 쪽으로.
DOWNGRADE_ORDER = ["share3", "share2", "differ", "single"]


@dataclass(frozen=True)
class Expression:
    name: str
    n_images: int
    family: str            # UTTERANCES 조회 키 (share2/share3 -> "share")
    evidence_is_indexed: bool   # 어느 이미지가 evidence 인지 텍스트가 지정하는가

    def check(self, imgs: Sequence[dict], axis: str, value: str,
              evidence_index: int | None, is_neutral: Callable[[str, str], bool]
              ) -> list[str]:
        """이미지 조합이 이 방식의 연역 조건을 만족하는지. 위반 사유 목록 반환."""
        errs: list[str] = []
        others = [a for a in AXES if a != axis]
        if len(imgs) != self.n_images:
            return [f"{self.name}: needs {self.n_images} images, got {len(imgs)}"]

        if self.name == "single":
            if imgs[0][axis] != value:
                errs.append(f"single: image does not carry {axis}={value}")

        elif self.name == "differ":
            if evidence_index is None or not 0 <= evidence_index < 2:
                errs.append("differ: evidence_index missing")
            else:
                if imgs[evidence_index][axis] != value:
                    errs.append(f"differ: evidence image lacks {axis}={value}")
                partner = imgs[1 - evidence_index][axis]
                # 파트너 값은 사용자-중립이어야 한다. 비선호 값을 쓰면
                # 말하지 않은 극성이 우연히 맞아버려 통제가 흐려진다.
                if not is_neutral(axis, partner):
                    errs.append(f"differ: partner {axis}={partner} not neutral")
            varying = [a for a in AXES if imgs[0][a] != imgs[1][a]]
            if varying != [axis]:
                errs.append(f"differ: varies on {varying}, want [{axis}]")

        else:                                   # share2 / share3
            if any(im[axis] != value for im in imgs):
                errs.append(f"share: {axis}={value} not shared by all")
            for a in others:
                if len({im[a] for im in imgs}) == 1:
                    errs.append(f"share: {a} shared by all too "
                                f"→ two common elements, not deductive")

        # 비-evidence 축 값은 모두 사용자-중립이어야 한다 (R3).
        for im in imgs:
            for a in others:
                if not is_neutral(a, im[a]):
                    errs.append(f"{self.name}: non-neutral {a}={im[a]}")
        return errs


EXPRESSIONS = {
    "single": Expression("single", 1, "single", True),
    "differ": Expression("differ", 2, "differ", True),
    "share2": Expression("share2", 2, "share", False),
    "share3": Expression("share3", 3, "share", False),
}

assert sum(TYPE_QUOTA.values()) == 18, "quota must cover all 18 preferences"
assert set(TYPE_QUOTA) == set(EXPRESSIONS) == set(DOWNGRADE_ORDER)
