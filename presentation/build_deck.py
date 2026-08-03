#!/usr/bin/env python3
"""공학연구인턴십 성과발표회 PPT — POD-Bench (조현준 / VisAGI Lab).

Built on the author's own deck as a template so the master, theme, A4 slide
size, the blue dash header rule and the 맑은 고딕 font all carry over.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
import copy

import os
HERE = os.path.dirname(os.path.abspath(__file__))
FIG  = f"{HERE}/figs"
OUT  = f"{HERE}/POD-Bench_성과발표.pptx"
TPL  = f"{HERE}/assets/template.pptx"
RULE = f"{HERE}/assets/header_rule.png"
LOGO = f"{HERE}/assets/uos_logo.png"

# ── palette lifted from the author's deck ────────────────────────────────────
NAVY   = RGBColor(0x00, 0x40, 0x94)
BLUE   = RGBColor(0x0A, 0x4D, 0x9B)
MID    = RGBColor(0x2E, 0x75, 0xB6)
LIGHT  = RGBColor(0x9D, 0xC3, 0xE6)
PALE   = RGBColor(0xE9, 0xF1, 0xFA)
PALER  = RGBColor(0xF5, 0xF8, 0xFC)
RED    = RGBColor(0xC0, 0x00, 0x00)
PURPLE = RGBColor(0x70, 0x30, 0xA0)
INK    = RGBColor(0x27, 0x25, 0x1E)
GREY   = RGBColor(0x76, 0x71, 0x71)
LGREY  = RGBColor(0xD9, 0xD9, 0xD9)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

KFONT = "맑은 고딕"

SW, SH   = 10.8333, 7.5
HDR      = (0.72, 0.44, 10.13, 0.17)      # the blue dash rule, every slide
TITLE_XY = (0.64, 0.96, 9.66, 0.52)
LM       = 0.64                            # left margin
CW       = 9.55                            # content width

prs = Presentation(TPL)
BLANK = prs.slide_layouts[6]               # 빈 화면


# ── helpers ──────────────────────────────────────────────────────────────────
def _ea(run, name=KFONT):
    """python-pptx sets only the latin typeface; Korean needs <a:ea> too."""
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", name)


def style(run, size=16, bold=False, color=INK, font=KFONT, italic=False):
    f = run.font
    f.name, f.size, f.bold, f.italic = font, Pt(size), bold, italic
    f.color.rgb = color
    _ea(run, font)
    return run


def add_slide(page=None):
    s = prs.slides.add_slide(BLANK)
    for ph in list(s.placeholders):                 # blank layout still carries 3
        ph._element.getparent().remove(ph._element)
    s.shapes.add_picture(RULE, Inches(HDR[0]), Inches(HDR[1]),
                         Inches(HDR[2]), Inches(HDR[3]))
    if page is not None:
        tb = s.shapes.add_textbox(Inches(SW - 1.05), Inches(SH - 0.44),
                                  Inches(0.6), Inches(0.28))
        tb.text_frame.word_wrap = False
        p = tb.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
        style(p.add_run(), 10).text = str(page)
        p.runs[0].text = str(page); style(p.runs[0], 10, color=GREY)
    return s


def textbox(s, x, y, w, h, wrap=True, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tb, tf


def title(s, text, sub=None, color=NAVY):
    tb, tf = textbox(s, *TITLE_XY)
    p = tf.paragraphs[0]
    style(p.add_run(), 24, True, color).text = text
    if sub:
        r = p.add_run(); style(r, 14, False, GREY).text = "   " + sub
    return tb


def para(tf, text, size=16, bold=False, color=INK, indent=0, space_before=0,
         space_after=2, first=False, italic=False, line=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_before, p.space_after = Pt(space_before), Pt(space_after)
    if line:
        p.line_spacing = line
    if indent:
        p.level = min(indent, 4)
    p.alignment = PP_ALIGN.LEFT
    style(p.add_run(), size, bold, color, italic=italic).text = text
    return p


def rich(tf, parts, size=16, indent=0, space_before=0, space_after=2,
         first=False, align=None, line=None):
    """parts = [(text, bold, color), ...] inside one paragraph."""
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_before, p.space_after = Pt(space_before), Pt(space_after)
    if line:
        p.line_spacing = line
    if indent:
        p.level = min(indent, 4)
    p.alignment = align or PP_ALIGN.LEFT
    for t, b, c in parts:
        style(p.add_run(), size, b, c).text = t
    return p


def card(s, x, y, w, h, fill=PALER, line=LIGHT, lw=1.0, radius=0.10, shadow=False):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                            Inches(w), Inches(h))
    try:
        sh.adjustments[0] = radius
    except Exception:
        pass
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(lw)
    if not shadow:
        sp = sh._element.spPr
        el = sp.makeelement(qn("a:effectLst"), {})
        sp.append(el)
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.10)
    tf.margin_top = tf.margin_bottom = Inches(0.06)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return sh


def chip(s, x, y, w, h, text, fill, fg=WHITE, size=12, bold=True):
    sh = card(s, x, y, w, h, fill=fill, line=None, radius=0.35)
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = Inches(0.01)
    tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    style(p.add_run(), size, bold, fg).text = text
    return sh


def arrow(s, x, y, w, h=0.18, color=MID, right=True):
    sh = s.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW if right else MSO_SHAPE.DOWN_ARROW,
        Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sp = sh._element.spPr
    sp.append(sp.makeelement(qn("a:effectLst"), {}))
    return sh


def pic(s, name, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return s.shapes.add_picture(f"{FIG}/{name}", Inches(x), Inches(y), **kw)


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def sec_head(tf, text, first=False, space_before=0):
    """the author's signature body header: bold, blue, 16pt."""
    return para(tf, text, 16, True, BLUE, first=first, space_before=space_before,
                space_after=3)


# ═════════════════════════════════════════════════════════════════════════════
# 1 — 표지
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide()
tb, tf = textbox(s, LM, 4.16, 9.6, 0.90)
rich(tf, [("공학연구인턴십 ", True, NAVY), ("성과발표", True, NAVY)], size=40, first=True)

tb, tf = textbox(s, LM, 5.12, 9.7, 0.95)
rich(tf, [("POD-Bench", True, BLUE),
          ("  ·  VLM은 ", False, INK), ("사용자 취향", True, INK),
          ("과 ", False, INK), ("상황(TPO)", True, INK),
          ("을 동시에 만족시킬 수 있는가?", False, INK)],
     size=17, first=True)
para(tf, "Personalized Outfit Decision Benchmark for Vision-Language Models",
     12.5, False, GREY, space_before=4)

tb, tf = textbox(s, LM, 6.44, 4.2, 0.62)
rich(tf, [("VisAGI Lab", True, NAVY)], size=14, first=True)
rich(tf, [("인공지능학과  2023480017  조현준", True, NAVY)], size=14)

tb, tf = textbox(s, 6.6, 6.50, 2.3, 0.30)
rich(tf, [("2026. 08. 05.", False, GREY)], size=12, first=True, align=PP_ALIGN.RIGHT)
s.shapes.add_picture(LOGO, Inches(8.25), Inches(6.74),
                     Inches(1.95), Inches(0.32))
notes(s, "[0:00] 안녕하십니까. 인공지능학과 조현준입니다. "
         "VisAGI Lab에서 진행한 공학연구인턴십 결과를 발표하겠습니다. "
         "주제는 비전-언어 모델이 사용자의 취향과 상황을 동시에 만족시킬 수 있는지를 "
         "측정하는 벤치마크, POD-Bench입니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 2 — 목차
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(2)
title(s, "목차")

items = [("01", "연구실 · 인턴십 개요", "VisAGI Lab / 담당 범위", "0:20"),
         ("02", "문제 정의", "취향(Preference) vs 상황(TPO)", "0:40"),
         ("03", "Method — 벤치마크 구성", "설계 · 데이터 · 이미지 수집", "1:40"),
         ("04", "전공지식 활용 · 문제해결 사례", "국소 패턴 / 제약 배치", "1:20"),
         ("05", "Key Finding · Contribution", "핵심 결과와 기여", "0:50"),
         ("06", "향후 계획 · 소감", "학업 · 진로 계획", "0:30")]

y = 1.86
for i, (num, name, desc, t) in enumerate(items):
    c = card(s, LM, y, CW, 0.72, fill=PALER if i % 2 == 0 else WHITE,
             line=LIGHT, radius=0.16)
    c.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    chip(s, LM + 0.16, y + 0.16, 0.42, 0.40, num, NAVY, size=13)
    tb, tf = textbox(s, LM + 0.76, y + 0.13, 3.58, 0.48)
    rich(tf, [(name, True, BLUE)], size=15.5, first=True)
    tb, tf = textbox(s, LM + 4.40, y + 0.185, 3.95, 0.34)
    rich(tf, [(desc, False, GREY)], size=12.5, first=True)
    tb, tf = textbox(s, LM + CW - 1.05, y + 0.185, 0.85, 0.34)
    rich(tf, [(t, True, MID)], size=12.5, first=True, align=PP_ALIGN.RIGHT)
    y += 0.80

tb, tf = textbox(s, LM, 6.72, CW, 0.34)
rich(tf, [("발표 5분", True, RED), ("  ·  제한 시간 내 핵심 method · key finding · contribution 중심으로 구성", False, GREY)],
     size=12, first=True)
notes(s, "[0:15] 발표는 여섯 부분으로 구성했습니다. 문제 정의, 벤치마크 구성 방법, "
         "전공지식을 적용해 해결한 문제 두 가지, 그리고 핵심 결과와 향후 계획 순입니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 3 — 연구실 · 인턴십 개요
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(3)
title(s, "연구실 · 인턴십 개요")

tb, tf = textbox(s, LM, 1.80, CW, 0.9)
sec_head(tf, "VisAGI Lab  ·  Vision & AGI  (서울시립대학교 인공지능학과)", first=True)
rich(tf, [("Vision-Language Model", True, INK),
          ("의 인식·추론 능력을 평가하고 개선하는 연구를 수행 — ", False, INK),
          ("본 인턴십에서는 ", False, INK),
          ("개인화(Personalization) 평가 벤치마크", True, INK),
          (" 구축을 담당", False, INK)], size=15, space_before=2)

stats = [("2,621", "option plans\n(4지선다 문항)", NAVY),
         ("59", "canonical scenarios\n/ 20 archetypes", MID),
         ("1,661", "이미지 셀\n전수 수작업 검수", MID),
         ("7", "stage pipeline\n구성 → 수집 → 평가", NAVY)]
x = LM
for v, lab, col in stats:
    c = card(s, x, 3.02, 2.24, 1.16, fill=PALE, line=None, radius=0.12)
    tf2 = c.text_frame
    rich(tf2, [(v, True, col)], size=32, first=True, align=PP_ALIGN.CENTER)
    rich(tf2, [(lab, False, INK)], size=11, align=PP_ALIGN.CENTER, line=1.15)
    x += 2.34

tb, tf = textbox(s, LM, 4.44, CW, 0.4)
sec_head(tf, "인턴십에서 제가 담당한 범위", first=True)

work = [("벤치마크 설계", "2×2 factorial 설계, 3-way scoring 정의, 시나리오·프로필 생성 규칙 작성", NAVY),
        ("데이터 파이프라인", "FashionSigLIP 검색 · SAM3 분할 · patch-wise 속성 검증 수집기 구현", MID),
        ("품질 검증", "1,661개 셀 human annotation 도구 제작 및 전수 검수, 재현성 검증 스크립트", MID),
        ("평가 실험", "vLLM 기반 VLM 5개 input format ablation 실행 및 결과 분석", NAVY)]
y = 4.90
for name, desc, col in work:
    chip(s, LM, y, 1.62, 0.34, name, col, size=11.5)
    tb, tf = textbox(s, LM + 1.76, y + 0.045, CW - 1.80, 0.30)
    rich(tf, [(desc, False, INK)], size=13, first=True)
    y += 0.46

tb, tf = textbox(s, LM, 6.86, CW, 0.32)
rich(tf, [("사용 기술  ", True, GREY),
          ("PyTorch · vLLM · FAISS · Marqo-FashionSigLIP · SAM3 · Qwen3-VL · FastAPI", False, GREY)],
     size=11.5, first=True)
notes(s, "[0:20] VisAGI Lab은 비전-언어 모델의 인식과 추론 능력을 다루는 연구실입니다. "
         "저는 개인화 평가 벤치마크 구축을 맡아, 설계부터 데이터 수집 파이프라인 구현, "
         "품질 검증, 평가 실험까지 전 과정을 담당했습니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 4 — 문제 정의
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(4)
title(s, "Problem Definition", sub="취향과 상황은 다른 축이다")

tb, tf = textbox(s, LM, 1.74, CW, 0.60)
rich(tf, [("AI는 사용자의 ", True, INK), ("취향(Preference)", True, RED),
          ("과 ", True, INK), ("상황 기반 선택(TPO)", True, MID),
          ("을 구별하고, ", True, INK), ("동시에", True, PURPLE),
          (" 만족시킬 수 있는가?", True, INK)], size=18, first=True)

# 입력
card(s, LM, 2.42, 4.55, 1.52, fill=PALER, line=LIGHT, radius=0.10)
tb, tf = textbox(s, LM + 0.18, 2.54, 4.2, 1.30)
rich(tf, [("Profile", True, NAVY), ("  사용자 취향", False, GREY)], size=13, first=True)
rich(tf, [("navy·beige를 선호하고, orange·red는 피함", False, INK)], size=12.5,
     space_before=1)
rich(tf, [("Query", True, NAVY), ("  상황 (Time · Place · Occasion)", False, GREY)],
     size=13, space_before=7)
rich(tf, [("“한겨울 야외 홀리데이 마켓에서 하루 종일 있을 예정”", False, INK)],
     size=12.5, space_before=1)

# 실패 모드
card(s, 5.55, 2.42, 4.64, 1.52, fill=WHITE, line=RED, radius=0.10)
tb, tf = textbox(s, 5.73, 2.54, 4.3, 1.30)
rich(tf, [("기존 개인화 평가의 공백", True, RED)], size=13, first=True)
rich(tf, [("취향만 반영", True, INK), (" → 파란 원피스 (한겨울에 부적합)", False, INK)],
     size=12.5, space_before=4)
rich(tf, [("상황만 반영", True, INK), (" → 주황 후드 (사용자가 싫어하는 색)", False, INK)],
     size=12.5, space_before=2)
rich(tf, [("두 응답 모두 기존 기준에서는 “개인화된 응답”으로 채점됨", False, RED)],
     size=12.5, space_before=5)

arrow(s, 5.28, 4.02, 0.24, 0.26, color=LIGHT, right=False)
pic(s, "fig_abcd.png", 0.92, 4.34, w=9.00)
notes(s, "[0:40] 문제 정의입니다. 사용자는 파란색을 좋아하고 주황색을 싫어합니다. "
         "그리고 한겨울 야외라는 상황이 주어집니다. 취향만 반영하면 파란 원피스를, "
         "상황만 반영하면 주황 후드를 답하게 되는데, 기존 개인화 평가는 두 응답을 모두 "
         "'개인화됐다'고 채점합니다. 저희는 이 둘을 분리해 측정하기 위해 아래처럼 "
         "네 개의 선지를 2×2로 구성했습니다. A만이 정답입니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 5 — 벤치마크 설계 (2x2 → 3-way scoring)
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(5)
title(s, "Method ①  벤치마크 설계", sub="2×2 factorial → 3-way scoring")

tb, tf = textbox(s, LM, 1.78, CW, 0.40)
rich(tf, [("두 축을 ", False, INK), ("직교", True, BLUE),
          ("시켜 4지선다를 구성 — 어느 축에서 실패했는지가 답 하나로 분리 측정됨", False, INK)],
     size=15, first=True)

# 2x2 table
rows, cols = 3, 3
tbl_sh = s.shapes.add_table(rows, cols, Inches(LM), Inches(2.30),
                            Inches(5.62), Inches(1.62))
tbl = tbl_sh.table
tbl.columns[0].width = Inches(1.86)
tbl.columns[1].width = Inches(1.88)
tbl.columns[2].width = Inches(1.88)
for r in range(rows):
    tbl.rows[r].height = Inches(0.54)

cells = [["", "TPO 적합 garment", "TPO 부적합 garment"],
         ["선호 값", "A", "C"],
         ["비선호 값", "B", "D"]]
tags = {(1, 1): ("정답", PURPLE), (1, 2): ("취향만", BLUE),
        (2, 1): ("상황만", RED), (2, 2): ("둘 다 위반", GREY)}
for r in range(rows):
    for c in range(cols):
        cell = tbl.cell(r, c)
        cell.margin_left = cell.margin_right = Inches(0.06)
        cell.margin_top = cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf2 = cell.text_frame
        p = tf2.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        head = (r == 0 or c == 0)
        cell.fill.solid()
        cell.fill.fore_color.rgb = PALE if head else (RGBColor(0xF3, 0xEE, 0xF8)
                                                      if (r, c) == (1, 1) else WHITE)
        txt = cells[r][c]
        if (r, c) in tags:
            style(p.add_run(), 17, True, tags[(r, c)][1]).text = txt
            r2 = p.add_run(); style(r2, 11.5, False, GREY).text = "  " + tags[(r, c)][0]
        elif txt:
            style(p.add_run(), 13, True, NAVY if head else INK).text = txt

# scoring
card(s, 6.46, 2.30, 3.73, 1.62, fill=PALER, line=LIGHT, radius=0.10)
tb, tf = textbox(s, 6.64, 2.42, 3.4, 1.40)
rich(tf, [("3-way scoring", True, NAVY)], size=14, first=True)
for lab, expr, col, note in [("Strict", "A", PURPLE, "두 축 모두 성공"),
                             ("TPO", "A or B", MID, "상황 축만 채점"),
                             ("Preference", "A or C", LIGHT, "취향 축만 채점")]:
    rich(tf, [(f"{lab} ", True, col), (f"= {expr}", True, INK),
              (f"   {note}", False, GREY)], size=12.5, space_before=4)
rich(tf, [("→ 하나의 답에서 두 능력을 분해", False, INK)], size=12,
     space_before=6)

tb, tf = textbox(s, LM, 4.16, CW, 0.4)
sec_head(tf, "설계에서 통제한 교란 요인", first=True)

conf = [("Counterbalance", "모든 값이 A(선호)·B(비선호)에 균등 배치 →\n선호 무관 추측 정확도 0.50 (1,964 plans)"),
        ("Confusability", "시각적으로 혼동되는 쌍(black↔navy, jeans↔slacks)에\n큰 페널티 — 혼동쌍 비율 45% → 19%"),
        ("Background axis", "제3의 축은 A/B/C/D에서 동일하게 고정 →\n정답 식별의 단서가 될 수 없음")]
x = LM
for name, desc in conf:
    c = card(s, x, 4.62, 3.09, 1.22, fill=WHITE, line=LIGHT, radius=0.10)
    tf2 = c.text_frame
    rich(tf2, [(name, True, BLUE)], size=13, first=True, align=PP_ALIGN.CENTER)
    rich(tf2, [(desc, False, INK)], size=11, align=PP_ALIGN.CENTER, line=1.18)
    x += 3.23

tb, tf = textbox(s, LM, 5.98, CW, 0.9)
rich(tf, [("Track 분리  ", True, NAVY),
          ("물리적 부적합(Physical, 24개)과 사회적 복장 규범(Dress-code, 35개)은 ", False, INK),
          ("근거의 종류가 다르므로", True, INK), (" 별도 데이터셋으로 구성·채점하고 절대 합산하지 않음", False, INK)],
     size=13, first=True, line=1.25)
rich(tf, [("Physical", True, MID), ("은 물리적 위험·기능적 불가능에, ", False, GREY),
          ("Dress-code", True, MID), ("는 사회적 관습에 근거 — 후자는 평가 프롬프트에서 문화권을 명시적으로 한정", False, GREY)],
     size=11.5, space_before=3, line=1.2)
notes(s, "[1:10] 설계의 핵심입니다. 취향 축과 상황 축을 직교시켜 네 선지를 만들면, "
         "모델이 어느 축에서 실패했는지가 답 하나로 분리됩니다. Strict, TPO, Preference "
         "세 지표로 분해합니다. 또한 counterbalance로 추측 정확도를 0.5로 맞추고, "
         "혼동되는 색·의류 쌍에 페널티를 줘 시각적 교란을 통제했습니다. "
         "물리적 부적합과 사회적 복장 규범은 근거의 종류가 다르므로 절대 합산하지 않습니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 6 — 파이프라인
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(6)
title(s, "Method ②  전체 파이프라인", sub="구성 → 수집 → 평가")

tb, tf = textbox(s, LM, 1.76, CW, 0.36)
rich(tf, [("Stage 1–3은 ", False, INK), ("LLM을 전혀 쓰지 않는 결정론적 생성", True, BLUE),
          (" — seed 42로 SHA256까지 bit-identical 재현", False, INK)], size=14, first=True)

# row 1 — deterministic construction
card(s, LM, 2.24, CW, 1.42, fill=PALER, line=LIGHT, radius=0.06)
tb, tf = textbox(s, LM + 0.14, 2.32, 4.0, 0.28)
rich(tf, [("Stage 1–3   결정론적 구성 (no LLM)", True, NAVY)], size=12.5, first=True)
st1 = [("1  Profile", "24명 · 축별 quota\ncolor 3+3 / pattern 2+2 / garment 4+4"),
       ("2  Query", "user × scenario 적합성 검사\n2,561 queries"),
       ("3  Option Plan", "A/B/C/D 배치 + counterbalance\n2,621 plans")]
x = LM + 0.16
for name, desc in st1:
    c = card(s, x, 2.66, 2.94, 0.86, fill=WHITE, line=MID, radius=0.10)
    tf2 = c.text_frame
    rich(tf2, [(name, True, NAVY)], size=13, first=True, align=PP_ALIGN.CENTER)
    rich(tf2, [(desc, False, INK)], size=10.5, align=PP_ALIGN.CENTER, line=1.15)
    x += 3.10
arrow(s, 3.76, 3.02, 0.13, 0.16)
arrow(s, 6.86, 3.02, 0.13, 0.16)

# feedback edge from annotation into stage 3
card(s, 6.30, 3.80, 3.89, 0.46, fill=RGBColor(0xFF, 0xF3, 0xF3), line=RED, radius=0.16)
tb, tf = textbox(s, 6.42, 3.90, 3.70, 0.28)
rich(tf, [("attribute_library.json", True, RED),
          ("  →  Stage 3 생성 입력", False, INK)], size=11, first=True)
ua = s.shapes.add_shape(MSO_SHAPE.UP_ARROW, Inches(7.30), Inches(3.58),
                        Inches(0.20), Inches(0.22))
ua.fill.solid(); ua.fill.fore_color.rgb = RED; ua.line.fill.background()
ua._element.spPr.append(ua._element.spPr.makeelement(qn("a:effectLst"), {}))

# row 2 — collection + eval
card(s, LM, 4.42, CW, 1.62, fill=PALER, line=LIGHT, radius=0.06)
tb, tf = textbox(s, LM + 0.14, 4.50, 5.0, 0.28)
rich(tf, [("Stage 4–7   이미지 수집 · 검증 · 평가", True, NAVY)], size=12.5, first=True)
st2 = [("4  Image\nCollection", "FashionSigLIP 651k\n+ SAM3 + human 검수", MID),
       ("5  Label\nVerification", "규칙 기반 검증 +\nmulti-judge LLM", MID),
       ("6  Quality\nAudit", "조립 검사 +\nvision-essentiality", MID),
       ("7  Evaluation", "multimodal / text-only\n5 input formats", NAVY)]
x = LM + 0.16
for name, desc, col in st2:
    c = card(s, x, 4.86, 2.16, 1.02, fill=WHITE, line=col, radius=0.10)
    tf2 = c.text_frame
    rich(tf2, [(name, True, col)], size=12.5, first=True, align=PP_ALIGN.CENTER, line=1.05)
    rich(tf2, [(desc, False, INK)], size=10, align=PP_ALIGN.CENTER, line=1.12)
    x += 2.32
for ax_ in (2.99, 5.31, 7.63):
    arrow(s, ax_, 5.30, 0.12, 0.16)

tb, tf = textbox(s, LM, 6.22, CW, 0.86)
rich(tf, [("재현성 검증  ", True, NAVY),
          ("scripts/verify_release.sh — 생성 입력 해시 고정 → 재생성 → SHA256 3종 비교 → 검증기 실행", False, INK)],
     size=12.5, first=True)
rich(tf, [("+ mutation test 6종 × 8 plan = ", False, GREY),
          ("48/48 검출 필수", True, RED),
          ("  (실패할 수 없는 검증기는 증거가 아니라는 원칙)", False, GREY)],
     size=11.5, space_before=3)
notes(s, "[1:35] 전체 파이프라인입니다. Stage 1에서 3까지는 LLM을 전혀 쓰지 않는 "
         "결정론적 생성이라, 시드만 같으면 해시까지 동일하게 재현됩니다. "
         "Stage 4부터는 실제 이미지를 수집하고 검증합니다. 빨간 화살표가 뒤에서 설명할 "
         "두 번째 문제해결 사례입니다. 검증기 자체도 mutation test로 검증했습니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 7 — 데이터 구성
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(7)
title(s, "Method ③  데이터 구성", sub="24 users × 59 scenarios")

tb, tf = textbox(s, LM, 1.76, CW, 0.36)
rich(tf, [("시나리오는 ", False, INK),
          ("물리적 위험 · 기능적 불가능 · 널리 공유된 관습", True, BLUE),
          ("에 근거한 것만 채택 — 저자의 취향이 정답이 되지 않도록", False, INK)],
     size=14, first=True)

nums = [("24", "user profiles", "규칙 생성 · 축별 quota"),
        ("59", "scenarios", "20 archetypes · 2 tracks"),
        ("2,621", "option plans", "2,253 query 문맥"),
        ("97.3%", "4장 모두 확보", "2,550 / 2,621")]
x = LM
for v, lab, sub in nums:
    c = card(s, x, 2.22, 2.24, 1.06, fill=PALE, line=None, radius=0.12)
    tf2 = c.text_frame
    rich(tf2, [(v, True, NAVY)], size=25, first=True, align=PP_ALIGN.CENTER)
    rich(tf2, [(lab, True, INK)], size=10.5, align=PP_ALIGN.CENTER)
    rich(tf2, [(sub, False, GREY)], size=9, align=PP_ALIGN.CENTER)
    x += 2.34

pic(s, "fig_dataset.png", 0.72, 3.40, w=9.40)

tb, tf = textbox(s, LM, 5.46, CW, 0.42)
sec_head(tf, "설계상 반드시 공개해야 하는 비대칭", first=True)

facts = [("축별 quota가 비대칭", "어휘 크기(23·13·5)가 달라 의도적으로 다르게 설정하고 명시"),
         ("solid은 pattern의 baseline", "‘무늬 없음’은 여섯 번째 값이 아니라 부재 — v5에서 수정"),
         ("Dress-code 라벨은 미검증", "관습 기반 35개는 human validation 전 — 저자 라벨로 표기")]
y = 5.88
for name, desc in facts:
    tb, tf = textbox(s, LM, y, CW, 0.30)
    rich(tf, [("· ", False, MID), (name, True, INK), ("  —  ", False, GREY),
              (desc, False, INK)], size=11.5, first=True)
    y += 0.38
notes(s, "[1:55] 데이터 구성입니다. 24명의 프로필과 59개 시나리오로 2,621개 문항을 "
         "만들었고, 그중 97.3%가 네 장의 이미지를 모두 확보했습니다. "
         "왼쪽은 취향 축, 오른쪽은 상황 위반 축의 분포입니다. "
         "설계상 비대칭이나 아직 검증되지 않은 부분은 숨기지 않고 명시적으로 공개했습니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 8 — 이미지 수집
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(8)
title(s, "Method ④  이미지 수집 파이프라인", sub="Stage 4 — 엔지니어링 핵심")

tb, tf = textbox(s, LM, 1.76, CW, 0.36)
rich(tf, [("문항 하나에는 ", False, INK),
          ("(색 · 의류 · 패턴)이 정확히 통제된 실제 상품 이미지 4장", True, BLUE),
          ("이 필요 — 검색만으로는 절대 확보되지 않음", False, INK)], size=14, first=True)

pic(s, "fig_funnel.png", 1.22, 2.16, w=8.40)

tb, tf = textbox(s, LM, 5.44, CW, 0.4)
sec_head(tf, "축마다 다른 검증기를 붙인 이유", first=True)

axes_info = [("Garment", "closed-vocab VLM judge", "SAM3 마스크는 위치 특정용 —\n분할 점수로는 dress↔tank top 구분 불가", NAVY),
             ("Pattern", "5-density patch 분류", "패턴은 넓은 시각적 맥락이 필요 →\n타일을 크게", MID),
             ("Color", "15-density patch argmax", "단일 임베딩이 놓치는\nNavy↔Blue / Beige↔Brown 혼동 검출", MID)]
x = LM
for name, method, why, col in axes_info:
    c = card(s, x, 5.86, 3.09, 1.14, fill=WHITE, line=LIGHT, radius=0.10)
    tf2 = c.text_frame
    rich(tf2, [(name + "  ", True, col), (method, True, INK)], size=11.5,
         first=True, align=PP_ALIGN.CENTER)
    rich(tf2, [(why, False, GREY)], size=10, align=PP_ALIGN.CENTER, line=1.15)
    x += 3.23
notes(s, "[2:20] 문항 하나에는 색·의류·패턴이 정확히 통제된 실제 상품 이미지 네 장이 "
         "필요합니다. 검색 결과 16만 건에서 시작해 SAM3로 옷을 분할하고, 세 축을 각각 "
         "다른 방식으로 채점한 뒤, 최종적으로 사람이 1,661개 셀을 전수 확인했습니다. "
         "축마다 검증기를 다르게 붙인 이유가 아래에 있습니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 9 — 문제해결 ①
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(9)
title(s, "문제해결 ①  국소 패턴 문제", sub="증상 → 시도 → 원인 규명 → 해결", color=NAVY)

tb, tf = textbox(s, LM, 1.74, CW, 0.36)
rich(tf, [("증상  ", True, RED),
          ("‘plaid beige shirt’로 검색하면 ", False, INK),
          ("포켓에만 체크무늬가 있는 옷", True, RED),
          ("이 통과 — 축을 고정한다는 설계 전제가 깨짐", False, INK)], size=14, first=True)

pic(s, "fig_patch.png", 1.62, 2.10, w=7.60)

tb, tf = textbox(s, LM, 4.72, CW, 0.4)
sec_head(tf, "시도한 해결책과 결과", first=True)

tries = [("Query 증강", "검색어 구체화", "옷 정확도만 상승", False),
         ("Reranker", "Qwen VL 8B reasoning", "효과 없음", False),
         ("VLM-as-judge", "PASS / SELECT 판정", "동일한 한계", False)]
x = LM
for name, how, res, ok in tries:
    c = card(s, x, 5.14, 3.09, 0.66, fill=WHITE, line=LGREY, radius=0.10)
    tf2 = c.text_frame
    rich(tf2, [(name, True, GREY), ("  " + how, False, GREY)], size=11.5,
         first=True, align=PP_ALIGN.CENTER)
    rich(tf2, [("✕  " + res, True, RED)], size=11, align=PP_ALIGN.CENTER)
    x += 3.23

card(s, LM, 5.86, CW, 1.18, fill=PALE, line=NAVY, radius=0.08)
tb, tf = textbox(s, LM + 0.20, 5.96, CW - 0.40, 1.02)
rich(tf, [("원인  ", True, NAVY),
          ("CLIP · SigLIP · reranker 모두 이미지를 ", False, INK),
          ("전역 평균 풀링(global average pooling)", True, NAVY),
          ("으로 벡터 하나에 압축 — 패턴의 공간적 분포가 평균에 묻힘", False, INK)],
     size=12, first=True, line=1.18)
rich(tf, [("근거  ", True, GREY),
          ("FashionVLP (CVPR’22) — global average pooling은 위치 정보를 보존하지 못함", False, GREY)],
     size=11, space_before=4, line=1.18)
rich(tf, [("해결  ", True, NAVY),
          ("SAM3 마스크 위 patch-wise 점수로 대체 — 학습 없이 표현 단위만 교체", True, NAVY)],
     size=11.5, space_before=3, line=1.18)
notes(s, "[2:45] 첫 번째 문제해결 사례입니다. 'plaid beige shirt'를 검색하면 포켓에만 "
         "체크무늬가 있는 옷이 통과했습니다. 검색어 증강, 리랭커, VLM 판정 세 가지를 "
         "시도했지만 모두 실패했습니다. 원인을 파고들어 보니 세 방법 모두 전역 평균 "
         "풀링으로 이미지를 벡터 하나로 압축한다는 공통점이 있었습니다. "
         "패턴의 공간 정보가 평균에 묻힌 겁니다. FashionVLP 논문에서 같은 지적을 "
         "확인했고, 학습 없이 표현 단위만 패치 단위로 바꿔 해결했습니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 10 — 문제해결 ②
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(10)
title(s, "문제해결 ②  제약을 파이프라인 앞으로", sub="필터의 위치가 데이터의 균형을 결정한다")

tb, tf = textbox(s, LM, 1.74, CW, 0.36)
rich(tf, [("증상  ", True, RED),
          ("3,027개 문항 중 ", False, INK), ("1,012개(33.4%)만", True, RED),
          (" 네 장의 이미지를 모두 확보 — 게다가 살아남은 문항의 ", False, INK),
          ("분포가 틀어짐", True, RED)], size=14, first=True)

pic(s, "fig_availability.png", 1.32, 2.22, w=8.20)

tb, tf = textbox(s, LM, 4.74, CW, 0.4)
sec_head(tf, "핵심 통찰 — 잃은 2,015개가 문제가 아니라, 잃은 방식이 문제", first=True)

card(s, LM, 5.14, 4.62, 1.30, fill=WHITE, line=RED, radius=0.08)
tb, tf = textbox(s, LM + 0.16, 5.24, 4.34, 1.10)
rich(tf, [("이전 (v4)", True, RED), ("   제약이 파이프라인 끝", False, GREY)], size=12, first=True)
rich(tf, [("전체 격자에서 균형을 최적화한 뒤, 이미지 없는 문항을 ", False, INK),
          ("사후 삭제", True, RED)], size=11, space_before=3, line=1.18)
rich(tf, [("삭제가 무작위가 아니어서 최적화가 확보한 균형을 파괴 (38:62 → 31:69)",
           False, INK)], size=11, space_before=2, line=1.18)

card(s, 5.57, 5.14, 4.62, 1.30, fill=PALE, line=NAVY, radius=0.08)
tb, tf = textbox(s, 5.73, 5.24, 4.34, 1.10)
rich(tf, [("현재 (v5)", True, NAVY), ("   제약이 생성 입력", False, GREY)], size=12, first=True)
rich(tf, [("네 선지가 모두 ‘사람이 승인한 셀’인 후보만 ", False, INK),
          ("점수 계산 대상", True, NAVY)], size=11, space_before=3, line=1.18)
rich(tf, [("실현 가능 영역 안에서 최적화 → 97.3%, 분포는 설계값 유지",
           False, INK)], size=11, space_before=2, line=1.18)

tb, tf = textbox(s, LM, 6.50, CW - 0.7, 0.72)
rich(tf, [("일반화한 교훈  ", True, NAVY),
          ("사후 필터는 중립적이지 않다 — 이미 구속력을 가진 제약은 최적화 ", False, INK),
          ("이후", True, RED), ("가 아니라 최적화 ", False, INK),
          ("안", True, NAVY), ("으로 들어가야 한다.", False, INK)],
     size=12.5, first=True, line=1.2)
rich(tf, [("제약을 새로 추가한 것이 아니라 이미 존재하던 제약을 코드가 볼 수 있게 만든 것", False, GREY)],
     size=11, space_before=3, line=1.18)
notes(s, "[3:15] 두 번째 사례입니다. 3,027개 문항 중 33.4%만 이미지 네 장을 "
         "확보했습니다. 그런데 진짜 문제는 수율이 아니라, 삭제가 무작위가 아니어서 "
         "설계 단계에서 맞춰 놓은 데이터 균형이 깨졌다는 점이었습니다. "
         "제약을 파이프라인 끝에서 생성 입력으로 옮기자 97.3%로 오르면서 분포도 "
         "설계값을 유지했습니다. 사후 필터는 중립적이지 않다는 것이 핵심 교훈입니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 11 — Key Finding
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(11)
title(s, "Key Finding", sub="Qwen3-VL-4B · multimodal MCQ · 838 plans × 5 input formats")

tb, tf = textbox(s, LM, 1.76, CW, 0.40)
rich(tf, [("각 축은 따로 주면 거의 풀지만, ", True, INK),
          ("함께 주면 상황 축이 무너진다", True, RED)], size=18, first=True)

pic(s, "fig_tradeoff.png", 1.02, 2.18, w=8.80)

tb, tf = textbox(s, LM, 5.86, CW, 0.4)
sec_head(tf, "읽는 법", first=True)

reads = [("설계 검증", "축 하나만 주면 다른 축은 정확히 우연 수준\n→ 두 축이 실제로 분리되어 있다는 증거", MID),
         ("핵심 결과", "프로필 추가 시 TPO 86.3 → 70.0\n취향은 유지 — 상황 쪽이 밀려남", RED),
         ("결론", "Strict 64.0% — 각각은 가능하나\n동시에 적용하는 데에는 실패", NAVY)]
x = LM
for name, desc, col in reads:
    c = card(s, x, 6.24, 3.09, 0.78, fill=WHITE if col != RED else RGBColor(0xFF, 0xF3, 0xF3),
             line=LIGHT if col != RED else RED, radius=0.10)
    tf2 = c.text_frame
    rich(tf2, [(name, True, col)], size=12.5, first=True, align=PP_ALIGN.CENTER)
    rich(tf2, [(desc, False, INK)], size=10.5, align=PP_ALIGN.CENTER, line=1.15)
    x += 3.23
notes(s, "[3:45] 핵심 결과입니다. 상황만 주면 TPO 86.3%, 취향만 주면 Preference "
         "94.7%로 각 축은 거의 풉니다. 그리고 축 하나만 줬을 때 다른 축이 정확히 "
         "우연 수준으로 나온다는 점이 설계가 제대로 분리됐다는 증거입니다. "
         "그런데 둘을 함께 주면 TPO가 86.3에서 70.0으로 16.2포인트 떨어집니다. "
         "취향은 거의 유지되는데 상황 쪽이 밀려납니다. "
         "결과적으로 두 축을 모두 만족한 비율은 64%에 그칩니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 12 — 세부 결과 + Contribution
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(12)
title(s, "Contribution", sub="어디서 무너지는가 · 무엇을 남겼는가")

pic(s, "fig_breakdown.png", 1.12, 1.70, w=8.60)

tb, tf = textbox(s, LM, 4.98, CW, 0.4)
sec_head(tf, "Contribution", first=True)

contrib = [("문제를 분리 가능한 형태로 정식화",
            "‘개인화 실패’를 어느 축의 실패인지까지 분해하는 4지선다 설계"),
           ("가설을 데이터로 확인",
            "각 축 단독 86.3 / 94.7 → 동시 64.0, 문제 설정 시 세운 가설과 일치"),
           ("재현 가능한 데이터셋과 검증 체계",
            "SHA256 재현 · mutation test 48/48 · 1,661 셀 전수 검수"),
           ("검색 단계의 표현 한계를 진단하고 우회",
            "학습 없이 patch-wise 표현으로 국소 패턴 문제 해결")]
y = 5.40
for i, (name, desc) in enumerate(contrib):
    chip(s, LM, y, 0.28, 0.28, str(i + 1), NAVY, size=11)
    tb, tf = textbox(s, LM + 0.40, y + 0.005, CW - 0.44, 0.28)
    rich(tf, [(name, True, BLUE), ("   " + desc, False, INK)], size=11.5, first=True)
    y += 0.38

tb, tf = textbox(s, LM, 6.98, CW - 0.9, 0.3)
rich(tf, [("한계  ", True, RED),
          ("이미지 라벨 검증과 상용 모델 평가가 남아 있어 절대 수치는 잠정값", False, GREY)],
     size=11, first=True)
notes(s, "[4:15] 세부적으로 보면, 두 트랙 모두 같은 패턴을 보이고, 취향 축 중에서는 "
         "시각적 패턴이 가장 약합니다. 색은 74.8%인데 패턴은 51.2%입니다. "
         "기여는 네 가지입니다. 문제를 분리 가능한 형태로 정식화했고, 가설을 "
         "데이터로 확인했으며, 재현 가능한 데이터셋과 검증 체계를 남겼고, "
         "검색 단계의 표현 한계를 진단해 우회했습니다. "
         "다만 이미지 라벨 검증이 남아 있어 절대 수치는 잠정값입니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 13 — 향후 계획 · 소감
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(13)
title(s, "향후 계획 · 소감")

tb, tf = textbox(s, LM, 1.78, 4.62, 0.4)
sec_head(tf, "연구 계획", first=True)

plans = [("라벨 검증 완료", "Dress-code 35개 관습 기반 라벨 human validation"),
         ("상용 모델 평가 확대", "GPT · Gemini 등 closed-source 최소 2종 추가"),
         ("Position bias 실험", "정답 위치를 A/B/C/D로 고정해도 정답률이 같은가"),
         ("A/D 제거 실험", "B와 C만 남기면 취향과 상황 중 무엇을 택하는가")]
y = 2.16
for name, desc in plans:
    c = card(s, LM, y, 4.62, 0.62, fill=PALER, line=LIGHT, radius=0.10)
    c.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    tb, tf = textbox(s, LM + 0.16, y + 0.10, 4.34, 0.44)
    rich(tf, [(name, True, BLUE)], size=12, first=True)
    rich(tf, [(desc, False, INK)], size=10.5, line=1.1)
    y += 0.70

tb, tf = textbox(s, 5.57, 1.78, 4.62, 0.4)
sec_head(tf, "학업 · 진로 계획", first=True)

career = [("단기", "본 연구를 WACV 투고 수준으로 정리 — 라벨 검증과 상용 모델 평가를 마무리"),
          ("중기", "Vision-Language Model의 평가 방법론을 주제로 대학원 진학 준비"),
          ("장기", "‘모델이 무엇을 못하는지’를 측정 가능한 형태로 정의하는 연구자")]
y = 2.18
for name, desc in career:
    chip(s, 5.57, y, 0.66, 0.28, name, MID, size=11)
    tb, tf = textbox(s, 6.36, y - 0.03, 3.83, 0.60)
    rich(tf, [(desc, False, INK)], size=12, first=True, line=1.2)
    y += 0.66

card(s, 5.57, 4.26, 4.62, 2.48, fill=PALE, line=None, radius=0.08)
tb, tf = textbox(s, 5.75, 4.40, 4.28, 2.16)
rich(tf, [("인턴십 소감", True, NAVY)], size=13, first=True)
rich(tf, [("가장 크게 배운 것은 ", False, INK),
          ("‘실패한 결과를 그대로 남기는 법’", True, INK),
          ("이었습니다. 국소 패턴 문제는 세 번 실패한 뒤에야 원인에 도달했고, "
           "수율 33%는 숫자를 고치는 대신 원인을 찾아 파이프라인을 다시 설계해야 했습니다.",
           False, INK)], size=11.5, space_before=5, line=1.25)
rich(tf, [("성능이 아니라 ", False, INK), ("측정이 옳은지", True, NAVY),
          ("를 먼저 의심하는 태도를, 전공 수업에서 배운 실험 설계 개념과 연결해 "
           "직접 적용해 볼 수 있었던 시간이었습니다.", False, INK)],
     size=11.5, space_before=6, line=1.25)

card(s, LM, 5.06, 4.62, 1.68, fill=WHITE, line=NAVY, radius=0.08)
tb, tf = textbox(s, LM + 0.18, 5.18, 4.30, 1.44)
rich(tf, [("전공 교육과의 연결", True, NAVY)], size=12.5, first=True)
for a, b in [("실험설계 · 통계", "2×2 factorial · counterbalance · 교란 통제"),
             ("컴퓨터비전", "SigLIP · SAM3 · patch 표현의 한계 이해"),
             ("소프트웨어공학", "결정론적 파이프라인 · 해시 고정 · mutation test")]:
    rich(tf, [("· ", False, MID), (a, True, INK), ("  →  " + b, False, GREY)],
         size=10.5, space_before=3, line=1.15)
notes(s, "[4:35] 향후 계획입니다. 남은 라벨 검증과 상용 모델 평가를 마무리해 "
         "논문 투고 수준으로 정리하고, 평가 방법론을 주제로 대학원 진학을 준비하려 합니다. "
         "소감을 말씀드리면, 이번 인턴십에서 가장 크게 배운 것은 실패한 결과를 "
         "그대로 남기는 법이었습니다. 성능보다 측정이 옳은지를 먼저 의심하는 태도를 "
         "전공 수업에서 배운 실험 설계 개념과 연결해 직접 적용해 볼 수 있었습니다. "
         "감사합니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 14 — Q&A
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide()
tb, tf = textbox(s, 0, 3.22, SW, 0.9)
rich(tf, [("Q & A", True, BLUE)], size=52, first=True, align=PP_ALIGN.CENTER)
tb, tf = textbox(s, 0, 4.22, SW, 0.4)
rich(tf, [("감사합니다.", False, GREY)], size=15, first=True, align=PP_ALIGN.CENTER)
tb, tf = textbox(s, 0, 4.66, SW, 0.34)
rich(tf, [("인공지능학과  2023480017  조현준   ·   VisAGI Lab", False, GREY)],
     size=12, first=True, align=PP_ALIGN.CENTER)
s.shapes.add_picture(LOGO, Inches(8.25), Inches(6.74),
                     Inches(1.95), Inches(0.32))
notes(s, "[5:00] 감사합니다.")

prs.save(OUT)
print("saved:", OUT, "| slides:", len(prs.slides.__iter__.__self__._sldIdLst))
