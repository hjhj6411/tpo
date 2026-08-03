"""대화 생성용 설정 — 에피소드와 문장 템플릿.

configs/scenarios.py 와 같은 위치의 '설정'이다. 생성 로직(dialog/)이 아니라
여기를 고쳐야 표현이 바뀐다.

규칙 (dialog/validate_dialogs.py 가 기계 검증):
  * image 변형의 턴 텍스트에는 벤치마크 어휘(13색·23의류·6무늬)와 그 렌더링
    별칭이 하나도 등장하면 안 된다. 축 단어(color/print/cut)만 허용된다.
  * 시나리오·TPO 어휘를 쓰지 않는다. 대화는 취향만 전달한다.

## 축 종류에 따른 분기 (사실 정합)

  attribute (색·무늬)   두 이미지가 같은 옷이고 그 속성만 다르다.
                        → "같은 물건, 하나만 다름" 이 참이다.
  garment  (의류)       두 이미지가 다른 옷이고 색·무늬가 같다.
                        → "같은 물건" 은 거짓. 스웨터와 청바지를 "the same
                          thing" 이라 부르면 문장이 이미지와 어긋나고, 모델은
                          텍스트와 이미지 중 하나를 버려야 한다.

## 변형 풀 (표면 다양성)

각 (계열, 축종류, 극성) 키마다 문장을 4개씩 둔다. 한 대화 안에서 같은 키가
최대 4번 쓰이므로(실측), 중복 없이 뽑으면 대화 내 문장 반복이 0이 된다.
renderer 가 키별로 소진 순서대로 꺼내며, 순서는 사용자 시드로 섞여 사용자마다
다른 조합이 나온다.

변형은 어휘만 바꾼 것이 아니라 문장 구조를 바꾼다. 같은 뜻을 다른 통사로
말해야 표면 다양성이 실제로 생긴다.
"""

DIALOG_TEMPLATE_VERSION = "dlg-2026-08-03c"

AXIS_WORD = {"color": "color", "pattern": "print", "garment_category": "cut"}
AXIS_KIND = {"color": "attribute", "pattern": "attribute",
             "garment_category": "garment"}

ORDINAL = {0: "first", 1: "second"}
COUNT_WORD = {2: "two", 3: "three"}
ORTHOGRAPHY = {"t_shirt": "t-shirt", "polka_dot": "polka dot"}

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

# 에피소드 안에서 두 번째 이후 evidence 턴 앞에 붙는 연결어.
# 담화 연결을 만들고 문장 시작의 반복도 줄인다. 어휘 누설이 없어야 한다.
EPISODE_CONNECTORS = {
    "shopping_haul":   ["Same trip:", "Also in the bag:", "And then:"],
    "closet_cleanout": ["Next out of the pile:", "Same sort:", "Still going:"],
    "gift_reaction":   ["She also sent this:", "Next out of the box:",
                        "And one more:"],
    "packing_trip":    ["Next into the bag:", "Also going:", "Still deciding:"],
    "online_cart":     ["Next in the cart:", "Also saved:", "And this one:"],
    "laundry_day":     ["Next off the pile:", "Also here:", "And these:"],
}

# (계열, 축종류, 극성) -> [(사용자 발화, 어시스턴트 확인), x4]
# {A}=축 단어, {P}=서수(differ), {N}=개수 단어(share)
UTTERANCES = {

# ══ single — 이미지 1장 ═══════════════════════════════════════════════════
("single", "attribute", "like"): [
    ("This is exactly my favorite {A}. Grabbed it without thinking twice.",
     "Noted — that {A} is officially yours."),
    ("The {A} on this one is my favorite {A}, full stop.",
     "Logged. That {A} sits at the top of your list."),
    ("If I had to name one {A} I always go for, it's the {A} here.",
     "Understood — that's your default {A}."),
    ("I bought this for the {A} alone. Nothing else about it mattered.",
     "So the {A} carried the whole purchase. Noted."),
],
("single", "attribute", "dislike"): [
    ("This one, though: that {A} is the one {A} I always put back down.",
     "Understood — that {A} stays on the rack."),
    ("The {A} here is the single {A} I will not wear.",
     "Got it. That {A} is off the table."),
    ("Whatever else this has going for it, that {A} rules it out.",
     "Noted — that {A} is disqualifying on its own."),
    ("I've tried this {A} more than once and it has never worked.",
     "Tried and settled, then. That {A} goes on the no list."),
],
("single", "garment", "like"): [
    ("This is exactly my favorite {A}. I own it in every version I can find.",
     "Noted — that {A} is officially yours."),
    ("The {A} here is the one I reach for constantly.",
     "So that {A} is your default shape. Logged."),
    ("If my closet has a signature, it's this {A}.",
     "Understood — that {A} is the through-line."),
    ("I keep replacing this {A} rather than trying anything else.",
     "Replacing rather than switching says it all. That {A} is yours."),
],
("single", "garment", "dislike"): [
    ("This one, though: that {A} is the one {A} that never works on me.",
     "Understood — that {A} stays on the rack."),
    ("The {A} here has never once looked right on me.",
     "Got it. That {A} is off the table."),
    ("I've owned this {A} twice and worn it zero times.",
     "Owned twice, worn never. That {A} goes on the no list."),
    ("No matter how it's made, this {A} is not for me.",
     "Noted — the {A} itself is the problem, not the execution."),
],

# ══ differ — 이미지 2장, 한 축만 다름 ══════════════════════════════════════
# attribute: 같은 옷 두 벌 → "같은 물건" 이 참
("differ", "attribute", "like"): [
    ("Two of the same piece, and the only difference between them is the {A}. "
     "The {P} one's {A} is completely me.",
     "Same piece, one variable — the {P} one's {A} it is."),
    ("Identical apart from the {A}. I'd take the {P} one every time.",
     "One variable, one verdict. The {P} one's {A} is yours."),
    ("Here's the same thing twice, split only by the {A} — and the {P} one "
     "wins easily.",
     "Clean comparison. Logging the {P} one's {A}."),
    ("Same piece, two {A}s. The {P} one is the version I'd actually wear.",
     "Understood — the {P} one's {A} goes on your list."),
],
("differ", "attribute", "dislike"): [
    ("The same piece twice, differing only in the {A}. The {P} one's {A} is "
     "the {A} I never touch.",
     "Same piece, one variable — the {P} one's {A} goes on the no list."),
    ("Identical except for the {A}, and the {P} one is the one I'd leave.",
     "One variable, one verdict. The {P} one's {A} is out."),
    ("Same thing twice, split by the {A}. I'd never take the {P} one.",
     "Clean comparison. The {P} one's {A} is banned."),
    ("Two {A}s of the same piece. The {P} one is the version I avoid.",
     "Understood — the {P} one's {A} goes on the no list."),
],
# garment: 다른 옷 두 벌, 색·무늬 동일 → "같은 물건" 금지
("differ", "garment", "like"): [
    ("Two different pieces here, matched in every other way. The {P} one's "
     "{A} is completely me.",
     "Everything else held equal — the {P} one's {A} it is."),
    ("Different {A}s, everything else the same. I'd wear the {P} one.",
     "With the rest controlled, the {P} one's {A} is yours."),
    ("These two differ only in {A}, and the {P} one is the one I'd keep.",
     "Noted — the {P} one's {A} goes on your list."),
    ("Same everything, two {A}s. The {P} one is my shape.",
     "Understood. The {P} one's {A} is the keeper."),
],
("differ", "garment", "dislike"): [
    ("Two different pieces, identical apart from their shape. The {P} one's "
     "{A} is the {A} I never touch.",
     "Everything else held equal — the {P} one's {A} goes on the no list."),
    ("Different {A}s, everything else matched. I'd leave the {P} one.",
     "With the rest controlled, the {P} one's {A} is out."),
    ("They differ only in {A}, and the {P} one is the one I'd never buy.",
     "Noted — the {P} one's {A} is banned."),
    ("Same everything, two {A}s. The {P} one has never suited me.",
     "Understood. The {P} one's {A} goes on the no list."),
],

# ══ share — 이미지 2~3장, 한 축만 공유 ═════════════════════════════════════
# attribute: 서로 다른 옷들이 색(또는 무늬)만 공유 → "닮은 데 없다" 가 참
("share", "attribute", "like"): [
    ("These {N} look nothing alike, but they share exactly one thing — and "
     "that shared {A} is my favorite.",
     "Spotted it. Whatever {A} those {N} have in common belongs to you."),
    ("Nothing about these {N} matches except one {A}, and that's the {A} I "
     "love.",
     "One point of overlap — that shared {A} is the one that counts."),
    ("Put these {N} side by side and only one thing repeats. That repeat is "
     "my favorite {A}.",
     "Found the overlap — that {A} is yours."),
    ("Completely different pieces, all {N} sharing one {A}. That {A} is why "
     "I own every one of them.",
     "The common thread is the {A}. Understood."),
],
("share", "attribute", "dislike"): [
    ("Odd group of {N}: they have exactly one thing in common, and that "
     "shared {A} is precisely what I avoid.",
     "Understood. The one {A} those {N} share goes on the no list."),
    ("These {N} are unalike in every way but one {A} — the {A} I keep away "
     "from.",
     "One point of overlap — that shared {A} is the disqualifying one."),
    ("Only one thing repeats across these {N}, and that repeat is the {A} I "
     "won't wear.",
     "Found the overlap — that {A} is out."),
    ("All {N} are different pieces with one shared {A}. That {A} is the "
     "reason none of them worked.",
     "The common thread is that {A}. Noted as a no."),
],
# garment: 같은 종류의 옷들 → 서로 닮아 보이므로 문구를 바꾼다
("share", "garment", "like"): [
    ("These {N} differ in every way but one: every one of them is the same "
     "{A}, and that {A} is my favorite.",
     "Understood — the {A} those {N} have in common is yours."),
    ("All {N} of these are the same {A}, and that's the only thing they share. "
     "That {A} is my favorite.",
     "One shared {A}, everything else different. Logged."),
    ("I own these {N} in wildly different versions, but always this {A}.",
     "Same {A}, many versions — that's your shape."),
    ("Whatever else changes, these {N} keep the same {A}. That's on purpose.",
     "Deliberate repetition. The {A} is yours."),
],
("share", "garment", "dislike"): [
    ("These {N} have nothing in common except one thing: every one of them is "
     "the same {A}, and that {A} is precisely what I avoid.",
     "Understood. The {A} those {N} share goes on the no list."),
    ("All {N} are the same {A} and nothing else matches. That {A} is the one "
     "I stay away from.",
     "One shared {A}, and it's the disqualifying one. Logged."),
    ("I've tried this {A} in {N} different versions. None of them worked.",
     "Same {A}, many versions, same result. It's out."),
    ("Whatever else changes about these {N}, the {A} stays — and that's the "
     "problem.",
     "The constant is the {A}. Noted as a no."),
],
}

# ── 무결성 검사 (임포트 시점) ────────────────────────────────────────────
# 앵커가 빠진 변형은 설계 불변식을 깬다. 축 단어가 없으면 모델은 어느 축을
# 판단해야 하는지 알 수 없고, 서수/개수가 없으면 어느 이미지를 가리키는지
# 특정할 수 없다. 문장을 추가·수정할 때 여기서 즉시 걸린다.
for _k, _v in UTTERANCES.items():
    _fam, _kind, _pol = _k
    assert len(_v) == 4, f"{_k}: expected 4 variants, got {len(_v)}"
    for _i, (_u, _a) in enumerate(_v):
        assert "{A}" in _u, f"{_k}#{_i}: user turn has no axis word"
        assert "{A}" in _a, f"{_k}#{_i}: ack has no axis word"
        if _fam == "differ":
            assert "{P}" in _u, f"{_k}#{_i}: differ user turn has no ordinal"
        if _fam == "share":
            assert "{N}" in _u, f"{_k}#{_i}: share user turn has no count"