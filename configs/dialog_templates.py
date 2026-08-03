"""대화 생성용 설정 — 에피소드와 문장 템플릿.

configs/scenarios.py 와 같은 위치의 '설정'이다. 생성 로직(dialog/)이 아니라
여기를 고쳐야 표현이 바뀐다. 코드와 문구를 분리해 두는 이유는 시나리오
카탈로그와 같다: 문구 수정이 생성 알고리즘을 건드리지 않게 하기 위해서다.

규칙 (dialog/validate_dialogs.py 가 기계 검증):
  * image 변형의 턴 텍스트에는 벤치마크 어휘(13색·23의류·6무늬)와 그 렌더링
    별칭이 하나도 등장하면 안 된다. 축 단어(color/print/cut)만 허용된다.
  * 시나리오·TPO 어휘를 쓰지 않는다. 대화는 취향만 전달한다.
"""

DIALOG_TEMPLATE_VERSION = "dlg-2026-08-03"

# 축을 지목하는 힌트 단어. 값 어휘가 아니므로 누설이 아니다.
AXIS_WORD = {
    "color": "color",
    "pattern": "print",
    "garment_category": "cut",
}

ORDINAL = {0: "first", 1: "second"}
COUNT_WORD = {2: "two", 3: "three"}

# 표기 교정(별칭 아님). text 쌍둥이의 자리표시자에만 쓰인다.
ORTHOGRAPHY = {"t_shirt": "t-shirt", "polka_dot": "polka dot"}

# 취향이 자연스럽게 드러나는 일상 에피소드. 옷장 정리·선물 반응은
# 자연 대화에서 희소한 dislike 를 싣기 위해 반드시 포함한다.
EPISODES = [
    ("shopping_haul",
     "Went a little overboard shopping this weekend... want the damage report?",
     "Always. Walk me through it."),
    ("closet_cleanout",
     "Big closet purge tonight. The keep pile and the donate pile are both growing.",
     "Brutal honesty hour. What's surviving?"),
    ("gift_reaction",
     "My aunt's birthday box arrived. Mixed results, as usual.",
     "Aunt boxes are pure gambling. How did the wheel land?"),
    ("packing_trip",
     "Packing for a week away and trying to keep it to one carry-on.",
     "One bag for a week is ambitious. What's going in?"),
    ("online_cart",
     "My saved cart has gotten out of hand. Helping me triage it?",
     "Cart triage is my favorite sport. Show me."),
    ("laundry_day",
     "Laundry day, which means I'm staring at everything I actually wear.",
     "The honest inventory. What's in the pile?"),
]

# (표현 방식 계열, 극성) -> (사용자 발화, 어시스턴트 확인)
# {A}=축 단어, {P}=서수(differ), {N}=개수 단어(share)
UTTERANCES = {
    ("single", "like"): (
        "This is exactly my favorite {A}. Grabbed it without thinking twice.",
        "Noted — that {A} is officially yours."),
    ("single", "dislike"): (
        "This one though: that {A} is the one {A} I always put back down.",
        "Understood — that {A} stays on the rack."),
    ("differ", "like"): (
        "These two are the same thing except for one difference, and the "
        "{P} one's {A} is completely me.",
        "One difference, clear verdict — the {P} one's {A} it is."),
    ("differ", "dislike"): (
        "Same item twice, one thing different. The {P} one's {A} is the {A} "
        "I never touch.",
        "Got it — the {P} one's {A} goes on the no list."),
    ("share", "like"): (
        "These {N} look nothing alike, but they share exactly one thing — "
        "and that shared {A} is my favorite.",
        "Spotted it. Whatever {A} those {N} have in common belongs to you."),
    ("share", "dislike"): (
        "Odd group — they share exactly one thing, and that shared {A} is "
        "precisely what I avoid.",
        "Understood. The one {A} those {N} share goes on the no list."),
}
