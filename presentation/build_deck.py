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
    return para(tf, text, 15.5, True, BLUE, first=first, space_before=space_before,
                space_after=3)


def lead(s, text_parts, y=1.76, size=15):
    """one-line 'why this slide exists' sentence under the title."""
    tb, tf = textbox(s, LM, y, CW, 0.42)
    rich(tf, text_parts, size=size, first=True, line=1.2)
    return tb


# ═════════════════════════════════════════════════════════════════════════════
# 1 — 표지
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide()
tb, tf = textbox(s, LM, 4.16, 9.6, 0.90)
rich(tf, [("공학연구인턴십 ", True, NAVY), ("성과발표", True, NAVY)], size=40, first=True)

tb, tf = textbox(s, LM, 5.12, 9.7, 0.95)
rich(tf, [("POD-Bench", True, BLUE),
          ("  ·  AI는 ", False, INK), ("내 취향", True, INK),
          ("과 ", False, INK), ("지금 상황", True, INK),
          ("을 동시에 만족시킬 수 있는가?", False, INK)],
     size=17, first=True)
para(tf, "Personalized Outfit Decision Benchmark for Vision-Language Models",
     12.5, False, GREY, space_before=4)

tb, tf = textbox(s, LM, 6.44, 4.2, 0.62)
rich(tf, [("VisAGI Lab", True, NAVY)], size=14, first=True)
rich(tf, [("인공지능학과  2023480017  조현준", True, NAVY)], size=14)

tb, tf = textbox(s, 6.6, 6.50, 2.3, 0.30)
rich(tf, [("2026. 08. 05.", False, GREY)], size=12, first=True, align=PP_ALIGN.RIGHT)
s.shapes.add_picture(LOGO, Inches(8.25), Inches(6.74), Inches(1.95), Inches(0.32))
notes(s, "[0:00] 안녕하십니까. 인공지능학과 조현준입니다. VisAGI Lab에서 진행한 "
         "공학연구인턴십 결과를 발표하겠습니다. 주제는 AI가 사용자의 취향과 "
         "그때그때의 상황을 동시에 만족시킬 수 있는지를 재는 시험지를 만드는 "
         "일이었습니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 2 — 목차
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(2)
title(s, "목차")

items = [("01", "연구실 · 인턴십 개요", "0:20"),
         ("02", "무엇을 풀려고 했나", "0:40"),
         ("03", "어떻게 만들었나 — Method", "1:30"),
         ("04", "부딪힌 문제와 해결", "1:20"),
         ("05", "결과와 기여", "0:50"),
         ("06", "향후 계획 · 소감", "0:30")]

y = 1.98
for i, (num, name, t) in enumerate(items):
    card(s, LM, y, CW, 0.72, fill=PALER if i % 2 == 0 else WHITE,
         line=LIGHT, radius=0.16)
    chip(s, LM + 0.18, y + 0.17, 0.44, 0.38, num, NAVY, size=13)
    tb, tf = textbox(s, LM + 0.84, y + 0.18, 6.4, 0.40)
    rich(tf, [(name, True, BLUE)], size=16, first=True)
    tb, tf = textbox(s, LM + CW - 1.10, y + 0.20, 0.90, 0.34)
    rich(tf, [(t, True, MID)], size=13, first=True, align=PP_ALIGN.RIGHT)
    y += 0.80

tb, tf = textbox(s, LM, 6.86, CW, 0.34)
rich(tf, [("발표 5분", True, RED),
          ("  ·  핵심 method와 결과 중심으로 구성했습니다", False, GREY)],
     size=12, first=True)
notes(s, "[0:15] 발표는 여섯 부분입니다. 무엇을 풀려고 했는지, 어떻게 만들었는지, "
         "만들면서 부딪힌 문제를 어떻게 풀었는지, 그리고 결과와 향후 계획입니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 3 — 연구실 · 인턴십 개요
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(3)
title(s, "연구실 · 인턴십 개요")

lead(s, [("VisAGI Lab", True, BLUE),
         ("  ·  AI가 이미지와 언어를 함께 이해하는 능력을 ", False, INK),
         ("평가하고 개선", True, INK), ("하는 연구실", False, INK)])

stats = [("2,621", "만든 문항"),
         ("59", "상황 시나리오"),
         ("1,661", "직접 검수한 칸"),
         ("8주", "인턴십 기간")]
x = LM
for v, lab in stats:
    c = card(s, x, 2.42, 2.24, 1.02, fill=PALE, line=None, radius=0.12)
    tf2 = c.text_frame
    rich(tf2, [(v, True, NAVY)], size=30, first=True, align=PP_ALIGN.CENTER)
    rich(tf2, [(lab, False, INK)], size=11.5, align=PP_ALIGN.CENTER)
    x += 2.34

tb, tf = textbox(s, LM, 3.84, CW, 0.4)
sec_head(tf, "제가 담당한 범위", first=True)

work = [("설계", "무엇을 어떻게 잴지 정하고, 문항 생성 규칙 작성"),
        ("구현", "이미지 수집 파이프라인 전체 개발"),
        ("검수", "1,661칸 검수 도구 제작 및 전수 확인"),
        ("실험", "평가 실행과 결과 분석")]
y = 4.28
for name, desc in work:
    chip(s, LM, y, 0.86, 0.34, name, MID, size=12)
    tb, tf = textbox(s, LM + 1.04, y + 0.05, CW - 1.10, 0.30)
    rich(tf, [(desc, False, INK)], size=13.5, first=True)
    y += 0.50

card(s, LM, 6.34, CW, 0.62, fill=PALER, line=LIGHT, radius=0.10)
tb, tf = textbox(s, LM + 0.20, 6.46, CW - 0.40, 0.40)
rich(tf, [("사용 기술  ", True, GREY),
          ("PyTorch · vLLM · FAISS · FashionSigLIP · SAM3 · Qwen3-VL", False, GREY)],
     size=12, first=True)
notes(s, "[0:20] VisAGI Lab은 AI가 이미지와 언어를 함께 이해하는 능력을 평가하는 "
         "연구실입니다. 저는 8주 동안 설계부터 데이터 수집 구현, 검수, 평가 실험까지 "
         "전 과정을 맡았습니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 4 — 무엇을 풀려고 했나
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(4)
title(s, "무엇을 풀려고 했나", sub="취향과 상황은 서로 다른 문제다")

lead(s, [("AI에게 옷을 추천해 달라고 하면, ", False, INK),
         ("내 취향", True, RED), ("과 ", False, INK),
         ("지금 상황", True, MID), ("을 ", False, INK),
         ("동시에", True, PURPLE), (" 맞출 수 있을까?", False, INK)], size=17)

# the one example that runs through the whole talk
card(s, LM, 2.26, 4.55, 1.24, fill=PALER, line=LIGHT, radius=0.10)
tb, tf = textbox(s, LM + 0.20, 2.40, 4.18, 1.00)
rich(tf, [("이 사람은", True, NAVY), ("   파란색을 좋아하고 주황색을 싫어합니다", False, INK)],
     size=13, first=True, line=1.2)
rich(tf, [("지금 상황은", True, NAVY), ("   한겨울 야외 시장, 하루 종일", False, INK)],
     size=13, space_before=8, line=1.2)

card(s, 5.55, 2.26, 4.64, 1.24, fill=WHITE, line=RED, radius=0.10)
tb, tf = textbox(s, 5.73, 2.40, 4.28, 1.00)
rich(tf, [("한쪽만 맞추면 이렇게 됩니다", True, RED)], size=13, first=True)
rich(tf, [("취향만", True, INK), ("  →  파란 원피스 ", False, INK), ("(한겨울에 춥다)", False, RED)],
     size=12.5, space_before=5, line=1.2)
rich(tf, [("상황만", True, INK), ("  →  주황 후드 ", False, INK), ("(싫어하는 색)", False, RED)],
     size=12.5, space_before=3, line=1.2)

arrow(s, 5.30, 3.58, 0.22, 0.22, color=LIGHT, right=False)

# letters above, plain-Korean labels below — drawn here so they get real bold
FIG_X, FIG_W = 1.42, 8.00
opt_labels = [("A", "둘 다 만족", PURPLE),
              ("B", "상황만 맞음", RED),
              ("C", "취향만 맞음", BLUE),
              ("D", "둘 다 아님", GREY)]
for i, (letter, lab, col) in enumerate(opt_labels):
    cx = FIG_X + FIG_W * (i + 0.5) / 4
    tb, tf = textbox(s, cx - 1.0, 3.92, 2.0, 0.32)
    rich(tf, [(letter, True, col)], size=19, first=True, align=PP_ALIGN.CENTER)

pic(s, "fig_abcd.png", FIG_X, 4.28, w=FIG_W)

for i, (letter, lab, col) in enumerate(opt_labels):
    cx = FIG_X + FIG_W * (i + 0.5) / 4
    tb, tf = textbox(s, cx - 1.0, 6.42, 2.0, 0.34)
    rich(tf, [(lab, True, col)], size=13.5, first=True, align=PP_ALIGN.CENTER)

tb, tf = textbox(s, LM, 6.94, CW, 0.30)
rich(tf, [("A만이 정답", True, NAVY),
          ("입니다 — 나머지 셋은 각각 어디서 틀렸는지를 알려 줍니다.", False, GREY)],
     size=12.5, first=True, align=PP_ALIGN.CENTER)
notes(s, "[0:40] 예를 하나 들겠습니다. 파란색을 좋아하고 주황색을 싫어하는 사람이, "
         "한겨울 야외 시장에 하루 종일 있어야 합니다. 취향만 맞추면 파란 원피스를 "
         "추천하게 되는데 춥습니다. 상황만 맞추면 주황 후드를 추천하는데 싫어하는 "
         "색입니다. 기존 평가는 이 둘을 모두 '개인화된 답'으로 봅니다. "
         "그래서 저는 네 가지 선지를 만들었고, 둘 다 만족하는 A만 정답으로 뒀습니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 5 — Method ①  답 하나로 두 가지를 채점한다
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(5)
title(s, "Method ①  답 하나로 두 가지를 채점", sub="어디서 틀렸는지를 알아야 고칠 수 있다")

lead(s, [("“틀렸다”가 아니라 ", False, INK),
         ("어느 쪽에서 틀렸는지", True, BLUE),
         ("까지 나와야 고칠 수 있습니다.", False, INK)])

# scoring matrix — native table so the Korean renders in PowerPoint
opts = [("A", "좋아하는 색\n+ 따뜻한 옷", PURPLE),
        ("B", "싫어하는 색\n+ 따뜻한 옷", RED),
        ("C", "좋아하는 색\n+ 추운 옷", BLUE),
        ("D", "싫어하는 색\n+ 추운 옷", GREY)]
scores = [("Strict", "둘 다 만족  ← 진짜 정답", [1, 0, 0, 0], NAVY),
          ("상황 점수", "추위는 피했나", [1, 1, 0, 0], MID),
          ("취향 점수", "좋아하는 색인가", [1, 0, 1, 0], RGBColor(0x5B, 0x9B, 0xD5))]

TX, TY, TW = 0.90, 2.36, 9.05
tbl = s.shapes.add_table(4, 5, Inches(TX), Inches(TY), Inches(TW), Inches(2.62)).table
tbl.columns[0].width = Inches(2.65)
for c in range(1, 5):
    tbl.columns[c].width = Inches(1.60)
tbl.rows[0].height = Inches(0.88)
for r in range(1, 4):
    tbl.rows[r].height = Inches(0.58)

for r in range(4):
    for c in range(5):
        cell = tbl.cell(r, c)
        cell.margin_left = cell.margin_right = Inches(0.06)
        cell.margin_top = cell.margin_bottom = Inches(0.03)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE
        tf2 = cell.text_frame
        p = tf2.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER

        if r == 0 and c == 0:
            cell.fill.fore_color.rgb = WHITE
        elif r == 0:
            letter, meaning, col = opts[c - 1]
            cell.fill.fore_color.rgb = PALE if c == 1 else WHITE
            style(p.add_run(), 20, True, col).text = letter
            rich(tf2, [(meaning, False, INK)], size=10, align=PP_ALIGN.CENTER, line=1.1)
        elif c == 0:
            name, why, _hits, col = scores[r - 1]
            cell.fill.fore_color.rgb = PALER
            p.alignment = PP_ALIGN.LEFT
            style(p.add_run(), 14, True, col).text = name
            rich(tf2, [(why, False, GREY)], size=10.5, align=PP_ALIGN.LEFT)
        else:
            hit = scores[r - 1][2][c - 1]
            if hit:
                cell.fill.fore_color.rgb = PALE if r == 1 else WHITE
                style(p.add_run(), 20, True, scores[r - 1][3]).text = "●"
            else:
                style(p.add_run(), 14, False, LGREY).text = "·"

tb, tf = textbox(s, LM, 5.20, CW, 0.40)
sec_head(tf, "이렇게 하면 답 하나에서 두 가지가 따로 보입니다", first=True)

reads = [("A를 골랐다", "취향도 상황도 맞춤", NAVY),
         ("B를 골랐다", "상황은 알지만 취향을 무시", RED),
         ("C를 골랐다", "취향은 알지만 상황을 무시", MID)]
x = LM
for name, desc, col in reads:
    c = card(s, x, 5.64, 3.09, 0.74, fill=WHITE, line=LIGHT, radius=0.10)
    tf2 = c.text_frame
    rich(tf2, [(name, True, col)], size=13, first=True, align=PP_ALIGN.CENTER)
    rich(tf2, [(desc, False, INK)], size=11.5, align=PP_ALIGN.CENTER)
    x += 3.23

card(s, LM, 6.56, CW, 0.54, fill=PALER, line=LIGHT, radius=0.10)
tb, tf = textbox(s, LM + 0.20, 6.66, CW - 0.40, 0.36)
rich(tf, [("찍어서 맞을 확률이 정확히 25%", True, NAVY),
          ("가 되도록 선지를 배치했습니다 — 운으로 오른 점수는 걸러집니다.", False, INK)],
     size=12.5, first=True)
notes(s, "[1:05] 채점 방식이 이 연구의 핵심입니다. 답을 하나만 받아도 두 가지를 "
         "따로 잴 수 있게 했습니다. A를 고르면 둘 다 맞춘 것이고, B를 고르면 상황은 "
         "아는데 취향을 무시한 것, C를 고르면 반대입니다. "
         "그래서 '틀렸다'가 아니라 '어느 쪽에서 틀렸다'까지 나옵니다. "
         "찍어서 맞을 확률은 정확히 25%로 맞춰 뒀습니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 6 — Method ②  문항은 이렇게 만든다
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(6)
title(s, "Method ②  문항 만드는 법", sub="사람 → 상황 → 문항, 세 단계")

lead(s, [("문항을 손으로 쓰면 ", False, INK),
         ("제 취향이 정답이 되어 버립니다", True, RED),
         (". 그래서 규칙으로 만들었습니다.", False, INK)])

steps = [("①  사람 만들기",
          "좋아하는·싫어하는 색·무늬·옷을\n규칙으로 배정",
          "파랑 · 베이지 좋아함\n주황 · 빨강 싫어함",
          "24명"),
         ("②  상황 만들기",
          "누구나 같은 답을 낼 상황만\n채택 (위험하거나 명백한 것)",
          "한겨울 야외\n→ 후드 O, 원피스 X",
          "59개"),
         ("③  문항 만들기",
          "사람 × 상황을 곱해\n4지선다 하나로",
          "파란 후드 / 주황 후드\n파란 원피스 / 주황 원피스",
          "2,621문항")]

x = LM
for i, (name, what, ex, count) in enumerate(steps):
    card(s, x, 2.34, 3.03, 3.20, fill=WHITE, line=MID, radius=0.08)
    tb, tf = textbox(s, x + 0.18, 2.48, 2.67, 0.34)
    rich(tf, [(name, True, NAVY)], size=15, first=True)
    tb, tf = textbox(s, x + 0.18, 2.90, 2.67, 0.70)
    rich(tf, [(what, False, INK)], size=12, first=True, line=1.25)

    card(s, x + 0.18, 3.72, 2.67, 1.02, fill=PALE, line=None, radius=0.10)
    tb, tf = textbox(s, x + 0.30, 3.84, 2.43, 0.80)
    rich(tf, [("예시", True, MID)], size=10, first=True)
    rich(tf, [(ex, False, INK)], size=11, space_before=2, line=1.2)

    tb, tf = textbox(s, x + 0.18, 4.92, 2.67, 0.42)
    rich(tf, [(count, True, NAVY)], size=19, first=True, align=PP_ALIGN.CENTER)
    if i < 2:
        arrow(s, x + 3.09, 3.78, 0.12, 0.16)
    x += 3.23

card(s, LM, 5.80, CW, 0.68, fill=PALER, line=LIGHT, radius=0.10)
tb, tf = textbox(s, LM + 0.20, 5.92, CW - 0.40, 0.46)
rich(tf, [("세 단계 모두 사람 손이 들어가지 않습니다", True, NAVY),
          (" — 같은 씨앗을 넣으면 언제나 똑같은 데이터가 나옵니다.", False, INK)],
     size=13, first=True)

tb, tf = textbox(s, LM, 6.66, CW, 0.44)
rich(tf, [("왜 중요한가  ", True, GREY),
          ("남이 제 결과를 다시 만들어 검증할 수 있고, 제가 데이터를 유리하게 "
           "고르지 않았다는 증거가 됩니다.", False, GREY)], size=11.5, first=True, line=1.2)
notes(s, "[1:30] 문항은 세 단계로 만듭니다. 먼저 사람 24명의 취향을 규칙으로 "
         "배정하고, 다음으로 누구나 같은 답을 낼 상황 59개를 만듭니다. "
         "한겨울 야외라면 후드는 맞고 원피스는 아니다, 이렇게 이견이 없는 것만 "
         "골랐습니다. 마지막으로 둘을 곱해 2,621개 문항이 나옵니다. "
         "세 단계 모두 사람 손이 안 들어가서, 같은 씨앗이면 항상 같은 데이터가 "
         "나옵니다. 제가 유리한 데이터만 골랐을 가능성을 없애는 장치입니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 7 — Method ③  진짜 어려운 건 사진이었다
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(7)
title(s, "Method ③  진짜 어려운 건 사진", sub="문항 하나에 실제 상품 사진 4장")

lead(s, [("검색만으로는 ", False, INK),
         ("원하는 사진이 나오지 않습니다", True, RED),
         (" — 색·옷·무늬가 모두 맞아야 합니다.", False, INK)])

# funnel — five bars, centred, shrinking. Counts from docs/PROCESS.md §3–§4.
FUN = [("검색으로 모은 사진",        "166,100", 6.20, MID,   ""),
       ("옷 부분을 찾아낸 것",       "127,859", 5.50, MID,   "나머지는 옷이 안 보임"),
       ("색·옷·무늬가 맞는 것",      "38,829",  3.40, LIGHT, "셋 다 통과한 것만"),
       ("후보로 저장",              "19,012",  2.70, LIGHT, "칸마다 상위 10장"),
       ("사람이 눈으로 최종 확인",   "1,223",   1.35, NAVY,  "1,661칸 전수 확인")]
CX, FY, BH, GAP = 5.62, 2.40, 0.44, 0.14
for i, (name, val, w_, col, note) in enumerate(FUN):
    x0 = CX - w_ / 2
    c = card(s, x0, FY, w_, BH, fill=col, line=None, radius=0.10)
    rich(c.text_frame, [(val, True, WHITE)], size=14, first=True, align=PP_ALIGN.CENTER)
    tb, tf = textbox(s, LM, FY + 0.10, CX - w_ / 2 - LM - 0.16, 0.28)
    rich(tf, [(name, i in (0, 4), NAVY if i == 4 else INK)], size=12,
         first=True, align=PP_ALIGN.RIGHT)
    if note:
        tb, tf = textbox(s, CX + w_ / 2 + 0.16, FY + 0.11, 2.6, 0.26)
        rich(tf, [(note, False, GREY)], size=10.5, first=True)
    if i < len(FUN) - 1:
        arrow(s, CX - 0.05, FY + BH + 0.01, 0.10, 0.12, color=RGBColor(0xB4, 0xB4, 0xB4), right=False)
    FY += BH + GAP

card(s, LM, 5.62, CW, 0.66, fill=PALER, line=LIGHT, radius=0.10)
tb, tf = textbox(s, LM + 0.20, 5.74, CW - 0.40, 0.44)
rich(tf, [("16만 장을 걸러, 마지막엔 ", False, INK),
          ("제가 직접 1,223칸을 눈으로 확인", True, NAVY),
          ("했습니다.", False, INK)], size=13.5, first=True)

tb, tf = textbox(s, LM, 6.46, CW, 0.60)
rich(tf, [("결과  ", True, GREY), ("전체 문항의 ", False, GREY),
          ("97.3%", True, NAVY),
          ("가 사진 네 장을 모두 확보했습니다.", False, GREY)], size=12.5, first=True)
rich(tf, [("사진이 없어서 버린 문항이 거의 없다는 뜻입니다.", False, GREY)],
     size=11.5, space_before=3)
notes(s, "[1:55] 그런데 진짜 어려운 건 사진이었습니다. 문항 하나에 사진이 네 장 "
         "필요한데, 색과 옷과 무늬가 정확히 맞아야 합니다. '파란 후드'를 검색한다고 "
         "그런 사진이 그냥 나오지 않습니다. 16만 장에서 시작해 옷 부분만 잘라내고 "
         "세 가지를 각각 확인한 뒤, 마지막엔 제가 직접 1,223칸을 눈으로 확인했습니다. "
         "그 결과 문항의 97.3%가 사진 네 장을 모두 확보했습니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 8 — 문제해결 ①
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(8)
title(s, "문제해결 ①  검색이 엉뚱한 옷을 가져온다")

lead(s, [("“체크무늬 베이지 셔츠”를 찾으면 ", False, INK),
         ("주머니에만 체크가 있는 옷", True, RED),
         ("이 통과했습니다.", False, INK)])

pic(s, "fig_patch.png", 1.92, 2.14, w=7.00)      # 7.00 x 2.45

tb, tf = textbox(s, LM, 4.70, CW, 0.4)
sec_head(tf, "세 가지를 시도했고, 셋 다 실패했습니다", first=True)

tries = [("검색어를 더 자세히", "옷 정확도만 조금 올라감"),
         ("재정렬 모델 추가", "효과 없음"),
         ("AI에게 판정 맡기기", "똑같은 한계")]
x = LM
for name, res in tries:
    c = card(s, x, 5.06, 3.09, 0.68, fill=WHITE, line=LGREY, radius=0.10)
    tf2 = c.text_frame
    rich(tf2, [(name, True, GREY)], size=12.5, first=True, align=PP_ALIGN.CENTER)
    rich(tf2, [("✕  " + res, True, RED)], size=11.5, align=PP_ALIGN.CENTER)
    x += 3.23

card(s, LM, 5.88, CW, 1.08, fill=PALE, line=NAVY, radius=0.08)
tb, tf = textbox(s, LM + 0.20, 6.00, CW - 0.40, 0.86)
rich(tf, [("원인  ", True, NAVY),
          ("세 방법 모두 사진 한 장을 ", False, INK), ("숫자 하나로 요약", True, NAVY),
          ("해서 비교합니다. 무늬가 어디에 있는지가 평균에 묻힙니다.", False, INK)],
     size=12.5, first=True, line=1.2)
rich(tf, [("해결  ", True, NAVY),
          ("옷을 잘라 격자로 나눈 뒤 칸마다 따로 확인 — 모델을 새로 학습시키지 않고 "
           "보는 단위만 바꿨습니다.", False, INK)],
     size=12.5, space_before=5, line=1.2)
notes(s, "[2:20] 첫 번째 문제입니다. 체크무늬 셔츠를 찾으면 주머니에만 체크가 있는 "
         "옷이 통과했습니다. 검색어를 자세히 쓰고, 재정렬 모델을 붙이고, AI에게 판정을 "
         "맡겨 봤지만 셋 다 실패했습니다. 원인을 파보니 세 방법 모두 사진 한 장을 "
         "숫자 하나로 요약해서 비교한다는 공통점이 있었습니다. 무늬가 어디 있는지가 "
         "평균에 묻힌 겁니다. 그래서 옷을 격자로 나눠 칸마다 따로 확인하도록 바꿨고, "
         "모델을 새로 학습시키지 않고 해결했습니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 9 — 문제해결 ②
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(9)
title(s, "문제해결 ②  거르는 순서를 바꿨다")

lead(s, [("문항 3,027개를 만들었는데 ", False, INK),
         ("사진 네 장이 다 있는 건 33%뿐", True, RED),
         ("이었습니다.", False, INK)])

# before / after — 7.2in of bar track == 100%
TRACK_X, TRACK_W = 2.30, 7.20
for lab, pct, col, y_, note in [
        ("이전", 33.4, LGREY, 2.64, "다 만들고 나서 사진 없는 걸 버림"),
        ("현재", 97.3, NAVY,  3.54, "만들 때부터 사진 있는 것만 후보로")]:
    tb, tf = textbox(s, LM, y_ - 0.30, 7.0, 0.26)
    rich(tf, [(lab, True, RED if col == LGREY else NAVY),
              ("   " + note, False, GREY)], size=12, first=True)
    card(s, TRACK_X, y_, TRACK_W, 0.46, fill=RGBColor(0xF2, 0xF2, 0xF2),
         line=None, radius=0.06)
    fill_w = TRACK_W * pct / 100
    card(s, TRACK_X, y_, fill_w, 0.46, fill=col, line=None, radius=0.06)
    if pct > 60:                                   # long bar: label sits inside
        tb, tf = textbox(s, TRACK_X + fill_w - 1.42, y_ + 0.07, 1.28, 0.34)
        rich(tf, [(f"{pct}%", True, WHITE)], size=17, first=True, align=PP_ALIGN.RIGHT)
    else:
        tb, tf = textbox(s, TRACK_X + fill_w + 0.16, y_ + 0.07, 1.4, 0.34)
        rich(tf, [(f"{pct}%", True, GREY)], size=17, first=True)

tb, tf = textbox(s, 7.10, 3.12, 2.40, 0.34)
rich(tf, [("약 3배", True, RED)], size=17, first=True, align=PP_ALIGN.RIGHT)
tb, tf = textbox(s, LM, 4.14, CW, 0.28)
rich(tf, [("사진 네 장이 모두 있는 문항의 비율", False, GREY)], size=11, first=True)

card(s, LM, 4.62, 4.62, 1.10, fill=WHITE, line=RED, radius=0.08)
tb, tf = textbox(s, LM + 0.18, 4.74, 4.30, 0.88)
rich(tf, [("진짜 문제는 수율이 아니었습니다", True, RED)], size=13, first=True)
rich(tf, [("버려진 문항이 한쪽에 몰려서, 맞춰 놓은 균형이 깨졌습니다.", False, INK)],
     size=12, space_before=6, line=1.25)

card(s, 5.57, 4.62, 4.62, 1.10, fill=PALE, line=NAVY, radius=0.08)
tb, tf = textbox(s, 5.75, 4.74, 4.30, 0.88)
rich(tf, [("순서를 바꾸니 둘 다 해결", True, NAVY)], size=13, first=True)
rich(tf, [("버릴 일이 없으니 균형도 설계한 그대로 남습니다.", False, INK)],
     size=12, space_before=6, line=1.25)

card(s, LM, 5.90, CW, 0.94, fill=PALER, line=LIGHT, radius=0.10)
tb, tf = textbox(s, LM + 0.20, 6.02, CW - 0.40, 0.72)
rich(tf, [("배운 것  ", True, NAVY),
          ("‘나중에 거르면 된다’가 사실이 아니었습니다. 무엇을 버리느냐가 "
           "남은 데이터의 성격을 바꿉니다.", False, INK)], size=13, first=True, line=1.25)
rich(tf, [("조건을 새로 만든 게 아니라, 원래 있던 조건을 앞으로 옮겼을 뿐입니다.",
           False, GREY)], size=11.5, space_before=4)
notes(s, "[2:50] 두 번째 문제입니다. 문항 3,027개를 만들었는데 사진 네 장이 다 있는 "
         "건 33%뿐이었습니다. 그런데 진짜 문제는 수율이 아니었습니다. 버려진 문항이 "
         "한쪽에 몰려 있어서, 설계 단계에서 맞춰 놓은 균형이 깨져 버린 겁니다. "
         "그래서 만들 때부터 사진이 있는 것만 후보로 두도록 순서를 바꿨더니 "
         "97%로 올라가면서 균형도 그대로 남았습니다. "
         "나중에 거르면 된다는 생각이 틀렸다는 걸 배웠습니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 10 — Key Finding
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(10)
title(s, "결과", sub="Qwen3-VL · 사진 4장을 보고 고르는 문제 838개")

tb, tf = textbox(s, LM, 1.76, CW, 0.42)
rich(tf, [("따로 물어보면 잘합니다. ", True, INK),
          ("그런데 같이 물어보면 상황 쪽이 무너집니다", True, RED)], size=17, first=True)

pic(s, "fig_tradeoff.png", 1.17, 2.22, w=8.50)   # 8.50 x 3.42

tb, tf = textbox(s, LM, 5.78, CW, 0.40)
sec_head(tf, "읽는 법", first=True)

reads = [("설계가 맞았다", "한쪽만 주면 다른 쪽은 딱 찍기 수준", MID),
         ("핵심", "취향을 알려주면 상황 점수 16점 하락", RED),
         ("결론", "각각은 되는데 동시에는 안 됨 — 64%", NAVY)]
x = LM
for name, desc, col in reads:
    c = card(s, x, 6.20, 3.09, 0.78,
             fill=RGBColor(0xFF, 0xF3, 0xF3) if col == RED else WHITE,
             line=RED if col == RED else LIGHT, radius=0.10)
    tf2 = c.text_frame
    rich(tf2, [(name, True, col)], size=13, first=True, align=PP_ALIGN.CENTER)
    rich(tf2, [(desc, False, INK)], size=11.5, align=PP_ALIGN.CENTER)
    x += 3.23
notes(s, "[3:20] 결과입니다. 상황만 물어보면 86점, 취향만 물어보면 95점으로 각각은 "
         "잘합니다. 그리고 한쪽만 줬을 때 다른 쪽 점수가 정확히 찍기 수준으로 "
         "나오는데, 이게 두 가지를 제대로 분리했다는 증거입니다. "
         "그런데 둘을 같이 주면 상황 점수가 86에서 70으로 16점 떨어집니다. "
         "취향은 그대로인데 상황 쪽이 밀려납니다. "
         "결국 둘 다 맞춘 비율은 64%에 그칩니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 11 — Contribution
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(11)
title(s, "기여", sub="어디서 무너지는가 · 무엇을 남겼는가")

pic(s, "fig_breakdown.png", 1.12, 1.74, w=8.60)

tb, tf = textbox(s, LM, 5.06, CW, 0.40)
sec_head(tf, "이번 인턴십에서 남긴 것", first=True)

contrib = [("문제를 나눌 수 있는 형태로 바꿈", "‘틀렸다’를 ‘어디서 틀렸다’로"),
           ("세운 가설을 데이터로 확인", "각각 86 / 95점 → 동시에는 64점"),
           ("누구나 다시 만들 수 있는 데이터셋", "씨앗만 같으면 결과가 동일"),
           ("검색 단계의 한계를 진단하고 우회", "학습 없이 보는 단위만 교체")]
y = 5.42
for i, (name, desc) in enumerate(contrib):
    chip(s, LM, y, 0.30, 0.30, str(i + 1), NAVY, size=11)
    tb, tf = textbox(s, LM + 0.44, y + 0.02, CW - 0.48, 0.28)
    rich(tf, [(name, True, BLUE), ("      " + desc, False, INK)], size=12.5, first=True)
    y += 0.40

tb, tf = textbox(s, LM, 6.96, CW - 0.9, 0.30)
rich(tf, [("한계  ", True, RED),
          ("사진 라벨 검증과 상용 모델 평가가 남아 있어 절대 수치는 잠정값입니다.",
           False, GREY)], size=11, first=True)
notes(s, "[3:50] 세부적으로 보면 두 종류의 상황 모두 같은 패턴이고, 취향 중에서는 "
         "무늬가 가장 약합니다. 색은 75점인데 무늬는 51점입니다. "
         "이번 인턴십에서 남긴 것은 네 가지입니다. 문제를 나눌 수 있는 형태로 바꿨고, "
         "세운 가설을 데이터로 확인했고, 누구나 다시 만들 수 있는 데이터셋을 남겼고, "
         "검색 단계의 한계를 진단해 우회했습니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 12 — 향후 계획 · 소감
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide(12)
title(s, "향후 계획 · 소감")

tb, tf = textbox(s, LM, 1.80, 4.62, 0.4)
sec_head(tf, "남은 일", first=True)

plans = [("사진 라벨 검증 마무리", "관습 기반 35개 상황"),
         ("상용 모델로 평가 확대", "GPT · Gemini 등 2종 이상"),
         ("선지 순서 편향 실험", "정답 위치를 바꿔도 같은가")]
y = 2.24
for name, desc in plans:
    c = card(s, LM, y, 4.62, 0.66, fill=PALER, line=LIGHT, radius=0.10)
    tb, tf = textbox(s, LM + 0.18, y + 0.12, 4.30, 0.44)
    rich(tf, [(name, True, BLUE)], size=13, first=True)
    rich(tf, [(desc, False, GREY)], size=11, line=1.1)
    y += 0.76

tb, tf = textbox(s, 5.57, 1.80, 4.62, 0.4)
sec_head(tf, "학업 · 진로 계획", first=True)

career = [("단기", "이 연구를 논문 투고 수준으로 정리"),
          ("중기", "AI 평가 방법론을 주제로 대학원 진학 준비"),
          ("장기", "‘모델이 무엇을 못하는지’를 재는 연구자")]
y = 2.28
for name, desc in career:
    chip(s, 5.57, y, 0.66, 0.30, name, MID, size=11)
    tb, tf = textbox(s, 6.38, y + 0.02, 3.81, 0.54)
    rich(tf, [(desc, False, INK)], size=12.5, first=True, line=1.2)
    y += 0.66

card(s, 5.57, 4.30, 4.62, 2.14, fill=PALE, line=None, radius=0.08)
tb, tf = textbox(s, 5.77, 4.44, 4.24, 1.88)
rich(tf, [("소감", True, NAVY)], size=13.5, first=True)
rich(tf, [("가장 크게 배운 것은 ", False, INK),
          ("실패한 결과를 그대로 남기는 법", True, INK),
          ("이었습니다.", False, INK)], size=12.5, space_before=7, line=1.25)
rich(tf, [("숫자가 나쁘면 숫자를 고치고 싶어지는데, 두 번 다 원인을 찾아 "
           "다시 설계해야 했습니다.", False, INK)], size=12.5, space_before=5, line=1.25)
rich(tf, [("성능보다 ", False, INK), ("측정이 옳은지", True, NAVY),
          ("를 먼저 의심하게 됐습니다.", False, INK)],
     size=12.5, space_before=5, line=1.25)

card(s, LM, 4.66, 4.62, 1.78, fill=WHITE, line=NAVY, radius=0.08)
tb, tf = textbox(s, LM + 0.18, 4.80, 4.30, 1.52)
rich(tf, [("전공 수업이 어디에 쓰였나", True, NAVY)], size=13, first=True)
for a, b in [("실험설계 · 통계", "두 축을 분리하는 설계"),
             ("컴퓨터비전", "검색 모델이 무엇을 놓치는지"),
             ("소프트웨어공학", "다시 만들 수 있는 파이프라인")]:
    rich(tf, [("· ", False, MID), (a, True, INK), ("   " + b, False, GREY)],
         size=11.5, space_before=6, line=1.15)
notes(s, "[4:15] 앞으로는 남은 사진 라벨 검증을 마무리하고 상용 모델로 평가를 "
         "넓혀서 논문 투고 수준으로 정리하려 합니다. 진로로는 AI 평가 방법론을 "
         "주제로 대학원 진학을 준비하고 있습니다. "
         "소감을 말씀드리면, 가장 크게 배운 것은 실패한 결과를 그대로 남기는 "
         "법이었습니다. 숫자가 나쁘면 숫자를 고치고 싶어지는데, 두 번 다 원인을 "
         "찾아 다시 설계해야 했습니다. 성능보다 측정이 옳은지를 먼저 의심하게 됐고, "
         "그게 전공 수업에서 배운 실험 설계를 처음으로 제 손으로 써 본 경험이었습니다. "
         "감사합니다.")

# ═════════════════════════════════════════════════════════════════════════════
# 13 — Q&A
# ═════════════════════════════════════════════════════════════════════════════
s = add_slide()
tb, tf = textbox(s, 0, 3.22, SW, 0.9)
rich(tf, [("Q & A", True, BLUE)], size=52, first=True, align=PP_ALIGN.CENTER)
tb, tf = textbox(s, 0, 4.22, SW, 0.4)
rich(tf, [("감사합니다.", False, GREY)], size=15, first=True, align=PP_ALIGN.CENTER)
tb, tf = textbox(s, 0, 4.66, SW, 0.34)
rich(tf, [("인공지능학과  2023480017  조현준   ·   VisAGI Lab", False, GREY)],
     size=12, first=True, align=PP_ALIGN.CENTER)
s.shapes.add_picture(LOGO, Inches(8.25), Inches(6.74), Inches(1.95), Inches(0.32))
notes(s, "[5:00] 감사합니다.")

prs.save(OUT)

# ── ship-time integrity gate ────────────────────────────────────────────────
# The template is derived from another deck, so guard against inheriting its
# baggage again: a dangling relationship makes PowerPoint offer to repair, and
# a subsetted embedded CJK font makes it stall on open.
import zipfile
from sanitize_template import verify

verify(OUT)
_z = zipfile.ZipFile(OUT)
_pres = _z.read("ppt/presentation.xml").decode("utf-8")
assert "<p:embeddedFontLst>" not in _pres, "embedded fonts leaked back into the deck"
assert not [n for n in _z.namelist() if n.startswith("ppt/fonts/")], "fntdata leaked back in"
_n_slides = len([n for n in _z.namelist() if n.startswith("ppt/slides/slide")])
_app = _z.read("docProps/app.xml").decode("utf-8")
import re as _re
_claim = _re.search(r"<Slides>(\d+)</Slides>", _app)
assert _claim is None or int(_claim.group(1)) == _n_slides, (
    f"docProps/app.xml claims {_claim.group(1)} slides, package has {_n_slides}")
_z.close()

print("saved:", OUT, "| slides:", _n_slides,
      f"| {os.path.getsize(OUT)/1e6:.2f} MB")
