#!/usr/bin/env python3
"""Figures for the POD-Bench internship presentation.

All numbers are computed from the repo, not hard-coded guesses:
  - eval  : data_wacv_scenario_v5/eval/Qwen_Qwen3-VL-4B-Instruct/results.jsonl
  - plans : data_wacv_scenario_v5/options/option_plans.jsonl
  - cells : annotation/attribute_library.json

FONTS — the only Korean-capable face in this environment is WenQuanYi Zen Hei,
which has no bold cut and no U+2212 MINUS SIGN. So: Korean runs in Zen Hei with
colour and size carrying the emphasis, numbers run in DejaVu Sans where bold
actually renders, and every minus is an ASCII hyphen. The funnel and the
before/after bars are built natively in build_deck.py instead — PowerPoint gives
real 맑은 고딕 bold there.
"""
import json, collections, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

# ── palette (taken from the author's own deck) ───────────────────────────────
NAVY   = "#004094"
BLUE   = "#0A4D9B"
MID    = "#2E75B6"
LIGHT  = "#9DC3E6"
RED    = "#C00000"
PURPLE = "#7030A0"
GREY   = "#8C8C8C"
LGREY  = "#D9D9D9"
INK    = "#27251E"

KO = "WenQuanYi Zen Hei"      # Korean-capable, no bold cut
NUM = "DejaVu Sans"           # Latin/numerals, real bold

plt.rcParams.update({
    "font.family": KO,
    "font.size": 12,
    "axes.unicode_minus": False,      # Zen Hei has no U+2212
    "axes.edgecolor": "#BFBFBF",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def num(ax, x, y, s, size=12, color=INK, **kw):
    """Numeric label — DejaVu so that bold actually renders."""
    return ax.text(x, y, s, fontname=NUM, fontweight="bold", fontsize=size,
                   color=color, **kw)


def load_eval():
    p = f"{REPO}/data_wacv_scenario_v5/eval/Qwen_Qwen3-VL-4B-Instruct/results.jsonl"
    return [r for r in (json.loads(l) for l in open(p)) if not r.get("error")]


def agg(rows, keyf):
    d = collections.defaultdict(lambda: dict(n=0, s=0, t=0, p=0))
    for r in rows:
        c = d[keyf(r)]
        c["n"] += 1; c["s"] += r["strict"]; c["t"] += r["tpo"]; c["p"] += r["preference"]
    return {k: dict(n=v["n"], strict=100*v["s"]/v["n"], tpo=100*v["t"]/v["n"],
                    pref=100*v["p"]/v["n"]) for k, v in d.items()}


ROWS = load_eval()
BYFMT = agg(ROWS, lambda r: r["input_format"])


# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — the headline result
# ═════════════════════════════════════════════════════════════════════════════
def fig_tradeoff():
    conds = [("query",           "상황만 알려줌"),
             ("narrative",       "취향만 알려줌"),
             ("narrative+query", "둘 다 알려줌")]
    metrics = [("tpo",    "상황 점수", MID),
               ("pref",   "취향 점수", LIGHT),
               ("strict", "둘 다 맞춤", NAVY)]

    fig, ax = plt.subplots(figsize=(10.4, 4.3))
    w, xs = 0.26, [0, 1, 2]

    ax.axhline(50, color="#C9C9C9", ls=(0, (5, 4)), lw=1.1, zorder=1)
    ax.axhline(25, color="#C9C9C9", ls=(0, (5, 4)), lw=1.1, zorder=1)
    ax.text(-0.60, 50, "찍기 수준\n(한 축)", fontsize=9.5, color=GREY,
            ha="center", va="center", linespacing=1.35)
    ax.text(-0.60, 25, "찍기 수준\n(둘 다)", fontsize=9.5, color=GREY,
            ha="center", va="center", linespacing=1.35)

    tops = {}
    for i, (mk, label, col) in enumerate(metrics):
        vals = [BYFMT[c][mk] for c, _ in conds]
        pos = [x + (i - 1) * w for x in xs]
        tops[mk] = list(zip(pos, vals))
        ax.bar(pos, vals, w, label=label, color=col, edgecolor="white",
               linewidth=1.2, zorder=3)
        for px, v in zip(pos, vals):
            num(ax, px, v + 1.5, f"{v:.1f}", size=12.5,
                color=NAVY if col == NAVY else INK, ha="center", va="bottom",
                zorder=4)

    # the interference, drawn in the empty column left of the "둘 다" group
    t0, t2 = BYFMT["query"]["tpo"], BYFMT["narrative+query"]["tpo"]
    xt = tops["tpo"][2][0]
    ax.plot([xt - 0.40, xt + w/2], [t0, t0], color=RED, lw=1.2, ls=(0, (4, 3)), zorder=5)
    ax.annotate("", xy=(xt - 0.26, t2), xytext=(xt - 0.26, t0),
                arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=2.0,
                                mutation_scale=12, shrinkA=0, shrinkB=0), zorder=5)
    num(ax, xt - 0.33, (t0 + t2)/2 + 1.5, f"-{t0 - t2:.1f}p", size=14, color=RED,
        ha="right", va="center", zorder=6)
    ax.text(xt - 0.33, (t0 + t2)/2 - 7.0, "상황 점수", color=RED, fontsize=11.5,
            ha="right", va="center", zorder=6)

    p1, p2 = BYFMT["narrative"]["pref"], BYFMT["narrative+query"]["pref"]
    ax.text(2.52, 108, f"취향 점수는 거의 그대로 (-{p1 - p2:.1f}p)", color=GREY,
            fontsize=11, ha="right", va="center", zorder=6)

    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l in conds], fontsize=13.5)
    ax.set_xlim(-0.92, 2.55)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    for t in ax.get_yticklabels():
        t.set_fontname(NUM)
    ax.set_ylabel("정답률 (%)", fontsize=12)
    ax.grid(axis="y", color="#F2F2F2", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              frameon=False, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_tradeoff.png", bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("fig_tradeoff", {k: round(v, 1) for k, v in BYFMT["narrative+query"].items()})


# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — where it breaks
# ═════════════════════════════════════════════════════════════════════════════
def fig_breakdown():
    by_track = agg(ROWS, lambda r: (r["track"], r["input_format"]))
    by_axis  = agg(ROWS, lambda r: (r["active_axis"], r["input_format"]))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))

    ax = axes[0]
    conds = ["query", "narrative", "narrative+query"]
    cl = ["상황만", "취향만", "둘 다"]
    tracks = [("physical", "춥다·덥다 같은 상황", MID),
              ("dress_code", "격식 같은 상황", NAVY)]
    w = 0.35
    for i, (tk, lab, col) in enumerate(tracks):
        vals = [by_track[(tk, c)]["strict"] for c in conds]
        pos = [x + (i - 0.5) * w for x in range(len(conds))]
        ax.bar(pos, vals, w, label=lab, color=col, edgecolor="white", zorder=3)
        for px, v in zip(pos, vals):
            num(ax, px, v + 1.4, f"{v:.0f}", size=11, ha="center", va="bottom", zorder=4)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(cl, fontsize=12)
    ax.set_ylim(0, 88); ax.set_yticks([0, 25, 50, 75])
    for t in ax.get_yticklabels():
        t.set_fontname(NUM)
    ax.axhline(25, color=GREY, ls=(0, (4, 4)), lw=1)
    ax.set_ylabel("둘 다 맞춘 비율 (%)", fontsize=11)
    ax.set_title("상황의 종류를 나눠 봐도 결과는 같다", fontsize=12.5,
                 color=NAVY, loc="left")
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.grid(axis="y", color="#EDEDED", zorder=0); ax.set_axisbelow(True)

    ax = axes[1]
    order = [("color", "색"), ("garment_category", "옷 종류"), ("pattern", "무늬")]
    vals = [by_axis[(a, "narrative+query")]["strict"] for a, _ in order]
    cols = [MID, LIGHT, RED]
    ax.bar(range(3), vals, 0.55, color=cols, edgecolor="white", zorder=3)
    for i, v in enumerate(vals):
        num(ax, i, v + 1.4, f"{v:.1f}", size=12.5, ha="center", va="bottom", zorder=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels([l for _, l in order], fontsize=12.5)
    ax.set_ylim(0, 88); ax.set_yticks([0, 25, 50, 75])
    for t in ax.get_yticklabels():
        t.set_fontname(NUM)
    ax.axhline(25, color=GREY, ls=(0, (4, 4)), lw=1)
    ax.set_title("취향 중에서는 무늬가 가장 어렵다", fontsize=12.5,
                 color=NAVY, loc="left")
    ax.grid(axis="y", color="#EDEDED", zorder=0); ax.set_axisbelow(True)
    ax.annotate("", xy=(1.93, vals[2] + 4), xytext=(1.45, 74),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4,
                                connectionstyle="arc3,rad=0.18"))
    ax.text(1.42, 76, "색보다 20점 이상 낮음", fontsize=10.5, color=RED,
            ha="center", va="bottom")

    fig.tight_layout(w_pad=3.0)
    fig.savefig(f"{OUT}/fig_breakdown.png", bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("fig_breakdown ok")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — global pooling loses the pattern; a per-tile check keeps it
# ═════════════════════════════════════════════════════════════════════════════
def fig_patch():
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))

    def garment():
        img = np.ones((60, 44, 3))
        body = np.zeros((60, 44), bool)
        body[6:56, 8:36] = True
        body[10:26, 2:10] = True; body[10:26, 34:42] = True
        img[body] = np.array([0.93, 0.90, 0.83])
        for r in range(34, 46):
            for c in range(14, 26):
                if (r // 2 + c // 2) % 2 == 0:
                    img[r, c] = np.array([0.35, 0.30, 0.25])
        return img, body

    img, body = garment()

    ax = axes[0]
    ax.imshow(img); ax.axis("off")
    ax.set_title("‘체크무늬 베이지 셔츠’로\n검색해서 나온 사진", fontsize=12,
                 color=INK, linespacing=1.35)
    ax.add_patch(plt.Rectangle((13, 33), 13, 13, fill=False, ec=RED, lw=2))
    ax.text(22, 61, "체크가 주머니에만 있다", fontsize=11, color=RED, ha="center")

    ax = axes[1]
    ax.imshow(img); ax.axis("off")
    ax.set_title("기존 방식\n사진 한 장 = 숫자 하나", fontsize=12,
                 color=GREY, linespacing=1.35)
    ax.add_patch(plt.Circle((22, 30), 16, fill=True, fc="#00000018", ec=GREY,
                            lw=1.6, ls="--"))
    ax.text(22, 30, "평균", ha="center", va="center", fontsize=13, color=INK)
    ax.text(22, 61, "어디에 있는지가 묻힌다  →  통과", fontsize=11,
            color=RED, ha="center")

    ax = axes[2]
    ax.imshow(img); ax.axis("off")
    ax.set_title("바꾼 방식\n격자로 나눠 칸마다 확인", fontsize=12,
                 color=NAVY, linespacing=1.35)
    hits = tot = 0
    for r in range(6, 56, 8):
        for c in range(2, 42, 8):
            if not body[r:r+8, c:c+8].any():
                continue
            tot += 1
            plaid = (32 <= r < 48 and 10 <= c < 28)
            hits += plaid
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 8, 8, fill=False,
                                       ec=NAVY if plaid else "#B9CCE4",
                                       lw=1.8 if plaid else 0.9))
    ax.text(22, 61, f"{tot}칸 중 {hits}칸뿐  →  탈락", fontsize=11,
            color=NAVY, ha="center")

    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{OUT}/fig_patch.png", bbox_inches="tight",
                facecolor="white", pad_inches=0.10)
    plt.close(fig)
    print("fig_patch ok")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — the four options as garments (labels are added in build_deck.py so
#          they get real 맑은 고딕 bold)
# ═════════════════════════════════════════════════════════════════════════════
def fig_abcd():
    from matplotlib.patches import Polygon as Poly, FancyBboxPatch
    BLUE_G, ORANGE_G = "#2E5FA3", "#E8801E"

    def hoodie(ax, col):
        ax.add_patch(Poly([[.28,.10],[.72,.10],[.72,.72],[.62,.72],[.62,.82],
                           [.38,.82],[.38,.72],[.28,.72]], closed=True,
                          fc=col, ec="white", lw=1.6))
        ax.add_patch(Poly([[.10,.46],[.28,.72],[.28,.46]], closed=True, fc=col,
                          ec="white", lw=1.6))
        ax.add_patch(Poly([[.90,.46],[.72,.72],[.72,.46]], closed=True, fc=col,
                          ec="white", lw=1.6))
        ax.add_patch(Poly([[.38,.82],[.62,.82],[.56,.92],[.44,.92]], closed=True,
                          fc=col, ec="white", lw=1.6))

    def dress(ax, col):
        ax.add_patch(Poly([[.18,.08],[.82,.08],[.68,.74],[.32,.74]], closed=True,
                          fc=col, ec="white", lw=1.6))
        ax.add_patch(Poly([[.36,.74],[.44,.74],[.44,.90],[.40,.90]], closed=True,
                          fc=col, ec="white", lw=1.6))
        ax.add_patch(Poly([[.56,.74],[.64,.74],[.60,.90],[.56,.90]], closed=True,
                          fc=col, ec="white", lw=1.6))

    items = [(hoodie, BLUE_G, True), (hoodie, ORANGE_G, False),
             (dress, BLUE_G, False), (dress, ORANGE_G, False)]

    fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.5))
    for ax, (draw, col, highlight) in zip(axs, items):
        if highlight:
            ax.add_patch(FancyBboxPatch((0.03, 0.02), 0.94, 0.96,
                                        boxstyle="round,pad=0.01,rounding_size=0.05",
                                        fc="#F3EEF8", ec=PURPLE, lw=2.2,
                                        transform=ax.transAxes, zorder=0))
        draw(ax, col)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(f"{OUT}/fig_abcd.png", bbox_inches="tight", facecolor="white",
                pad_inches=0.06)
    plt.close(fig)
    print("fig_abcd ok  (labels are drawn in build_deck.py)")


if __name__ == "__main__":
    fig_tradeoff()
    fig_breakdown()
    fig_patch()
    fig_abcd()
    for stale in ("fig_funnel.png", "fig_availability.png", "fig_dataset.png"):
        p = os.path.join(OUT, stale)
        if os.path.exists(p):
            os.remove(p)
            print("removed (now drawn natively in the deck):", stale)
    print("\nfigures ->", OUT)
