#!/usr/bin/env python3
"""Figures for the POD-Bench internship presentation.

All numbers are computed from the repo, not hard-coded guesses:
  - eval  : data_wacv_scenario_v5/eval/Qwen_Qwen3-VL-4B-Instruct/results.jsonl
  - plans : data_wacv_scenario_v5/options/option_plans.jsonl
  - cells : annotation/attribute_library.json
  - funnel: docs/PROCESS.md (recorded screen counts)
"""
import json, collections, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

# ── palette (taken from the author's own deck) ───────────────────────────────
NAVY   = "#004094"   # UOS deep blue  – titles
BLUE   = "#0A4D9B"   # section blue
MID    = "#2E75B6"
LIGHT  = "#9DC3E6"
PALE   = "#DEEBF7"
RED    = "#C00000"
PURPLE = "#7030A0"
GREY   = "#8C8C8C"
LGREY  = "#D9D9D9"
INK    = "#27251E"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.edgecolor": "#BFBFBF",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def load_eval():
    p = f"{REPO}/data_wacv_scenario_v5/eval/Qwen_Qwen3-VL-4B-Instruct/results.jsonl"
    rows = [json.loads(l) for l in open(p)]
    return [r for r in rows if not r.get("error")]


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
# FIG 1 — the trade-off (the headline result)
# ═════════════════════════════════════════════════════════════════════════════
def fig_tradeoff():
    conds = [("query",           "Scenario only\n(no profile)"),
             ("narrative",       "Profile only\n(no scenario)"),
             ("narrative+query", "Both\n(profile + scenario)")]
    metrics = [("tpo",    "TPO  (situation)",   MID),
               ("pref",   "Preference (taste)", LIGHT),
               ("strict", "Strict  (both)",     NAVY)]

    fig, ax = plt.subplots(figsize=(10.4, 4.3))
    w = 0.26
    xs = list(range(len(conds)))

    # chance baselines: A-or-B and A-or-C are 2-of-4 -> 50%; A alone -> 25%
    ax.axhline(50, color="#C9C9C9", ls=(0, (5, 4)), lw=1.1, zorder=1)
    ax.axhline(25, color="#C9C9C9", ls=(0, (5, 4)), lw=1.1, zorder=1)
    ax.text(-0.62, 50, "chance\nTPO / Pref", fontsize=9, color=GREY,
            ha="center", va="center", linespacing=1.35)
    ax.text(-0.62, 25, "chance\nStrict", fontsize=9, color=GREY,
            ha="center", va="center", linespacing=1.35)

    tops = {}
    for i, (mk, label, col) in enumerate(metrics):
        vals = [BYFMT[c][mk] for c, _ in conds]
        pos = [x + (i - 1) * w for x in xs]
        tops[mk] = list(zip(pos, vals))
        bars = ax.bar(pos, vals, w, label=label, color=col,
                      edgecolor="white", linewidth=1.2, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=11.5, fontweight="bold",
                    color=NAVY if col == NAVY else INK, zorder=4)

    # the interference: TPO loses 16.2p the moment a profile is added.
    # Drawn only in the empty column left of the "Both" group so nothing overlaps.
    t0, t2 = BYFMT["query"]["tpo"], BYFMT["narrative+query"]["tpo"]
    xt = tops["tpo"][2][0]                      # centre of the "Both" TPO bar
    ax.plot([xt - 0.40, xt + w/2], [t0, t0], color=RED, lw=1.2,
            ls=(0, (4, 3)), zorder=5)
    ax.annotate("", xy=(xt - 0.26, t2), xytext=(xt - 0.26, t0),
                arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=2.0,
                                mutation_scale=12, shrinkA=0, shrinkB=0), zorder=5)
    ax.text(xt - 0.33, (t0 + t2)/2, f"−{t0 - t2:.1f}p", color=RED, fontsize=13.5,
            fontweight="bold", ha="right", va="center", zorder=6)
    ax.text(xt - 0.33, (t0 + t2)/2 - 7.5, "TPO", color=RED, fontsize=11,
            ha="right", va="center", zorder=6)

    p1, p2 = BYFMT["narrative"]["pref"], BYFMT["narrative+query"]["pref"]
    ax.text(2.52, 108, f"Preference barely moves  (−{p1 - p2:.1f}p)",
            color=GREY, fontsize=10.5, ha="right", va="center", zorder=6)

    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l in conds], fontsize=12)
    ax.set_xlim(-0.92, 2.55)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("accuracy (%)", fontsize=11.5)
    ax.grid(axis="y", color="#F2F2F2", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              frameon=False, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_tradeoff.png", bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("fig_tradeoff", {k: round(v, 1) for k, v in BYFMT["narrative+query"].items()})


# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — where it breaks: by track and by preference axis
# ═════════════════════════════════════════════════════════════════════════════
def fig_breakdown():
    by_track = agg(ROWS, lambda r: (r["track"], r["input_format"]))
    by_axis  = agg(ROWS, lambda r: (r["active_axis"], r["input_format"]))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))

    # (a) track × condition, strict only
    ax = axes[0]
    conds = ["query", "narrative", "narrative+query"]
    cl = ["scenario\nonly", "profile\nonly", "both"]
    tracks = [("physical", "Physical", MID), ("dress_code", "Dress-code", NAVY)]
    w = 0.35
    for i, (tk, lab, col) in enumerate(tracks):
        vals = [by_track[(tk, c)]["strict"] for c in conds]
        pos = [x + (i - 0.5) * w for x in range(len(conds))]
        bars = ax.bar(pos, vals, w, label=lab, color=col, edgecolor="white", zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+1.4, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=10.5, fontweight="bold", zorder=4)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(cl, fontsize=10.5)
    ax.set_ylim(0, 85); ax.set_yticks([0, 25, 50, 75])
    ax.axhline(25, color=GREY, ls=(0, (4, 4)), lw=1)
    ax.set_ylabel("Strict accuracy (%)", fontsize=10.5)
    ax.set_title("(a)  by track", fontsize=12, fontweight="bold", color=NAVY, loc="left")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(axis="y", color="#EDEDED", zorder=0); ax.set_axisbelow(True)

    # (b) preference axis, both-inputs condition
    ax = axes[1]
    axes_order = [("color", "color"), ("garment_category", "garment"), ("pattern", "pattern")]
    vals = [by_axis[(a, "narrative+query")]["strict"] for a, _ in axes_order]
    ns   = [by_axis[(a, "narrative+query")]["n"] for a, _ in axes_order]
    cols = [MID, LIGHT, RED]
    bars = ax.bar(range(3), vals, 0.55, color=cols, edgecolor="white", zorder=3)
    for b, v, n in zip(bars, vals, ns):
        ax.text(b.get_x()+b.get_width()/2, v+1.4, f"{v:.1f}", ha="center",
                va="bottom", fontsize=11.5, fontweight="bold", zorder=4)
        ax.text(b.get_x()+b.get_width()/2, 2.5, f"n={n}", ha="center",
                fontsize=9.5, color="white", zorder=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels([l for _, l in axes_order], fontsize=11)
    ax.set_ylim(0, 85); ax.set_yticks([0, 25, 50, 75])
    ax.axhline(25, color=GREY, ls=(0, (4, 4)), lw=1)
    ax.set_title("(b)  by preference axis  (both inputs)", fontsize=12,
                 fontweight="bold", color=NAVY, loc="left")
    ax.grid(axis="y", color="#EDEDED", zorder=0); ax.set_axisbelow(True)
    ax.annotate("pattern is the\nweakest axis", xy=(1.94, vals[2] + 6),
                xytext=(1.58, 82), fontsize=10, color=RED, ha="center",
                va="top", linespacing=1.35,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4,
                                connectionstyle="arc3,rad=0.18"))

    fig.tight_layout(w_pad=3.0)
    fig.savefig(f"{OUT}/fig_breakdown.png", bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("fig_breakdown ok")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — image-collection funnel (recorded counts, docs/PROCESS.md §3–§4)
# ═════════════════════════════════════════════════════════════════════════════
def fig_funnel():
    lib = json.load(open(f"{REPO}/annotation/attribute_library.json"))
    cells = lib["cells"]
    avail = sum(1 for v in cells.values() if v["status"] == "available")

    stages = [
        ("FashionSigLIP  top-100 retrieval", 166_100, "1,661 cells x 100", MID),
        ("SAM3  garment mask localized",     127_859, "mask found",        MID),
        ("3-axis patch score > 0",            38_829, "colour/pattern/garment", LIGHT),
        ("saved candidates",                  19_012, "top-10 per cell",   LIGHT),
        ("human-approved cells",              avail,  f"of {len(cells):,} annotated", NAVY),
    ]
    fig, ax = plt.subplots(figsize=(10.4, 4.0))
    top = stages[0][1]
    y = 0
    H, GAP = 0.62, 0.28
    for i, (name, n, note, col) in enumerate(stages):
        # log-ish width so the last stage is still visible
        wfrac = (n / top) ** 0.34
        x0 = (1 - wfrac) / 2
        ax.add_patch(Polygon([[x0, y], [x0 + wfrac, y], [x0 + wfrac, y - H], [x0, y - H]],
                             closed=True, facecolor=col, edgecolor="white", lw=1.5, zorder=3))
        ax.text(0.5, y - H/2, f"{n:,}", ha="center", va="center", color="white",
                fontsize=14, fontweight="bold", zorder=4)
        ax.text(-0.03, y - H/2, name, ha="right", va="center", fontsize=11.5,
                color=INK, fontweight="bold" if i in (0, 4) else "normal")
        ax.text(1.03, y - H/2, note, ha="left", va="center", fontsize=10, color=GREY)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((0.5, y - H - 0.02), (0.5, y - H - GAP + 0.02),
                                         arrowstyle="-|>", mutation_scale=13,
                                         color="#B4B4B4", lw=1.3, zorder=2))
        y -= (H + GAP)
    ax.set_xlim(-0.42, 1.42); ax.set_ylim(y + GAP - 0.15, 0.15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_funnel.png", bbox_inches="tight",
                facecolor="white", pad_inches=0.10)
    plt.close(fig)
    print("fig_funnel  available cells =", avail)


# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — the availability fix: post-hoc filter -> generation constraint
# ═════════════════════════════════════════════════════════════════════════════
def fig_availability():
    lib = json.load(open(f"{REPO}/annotation/attribute_library.json"))
    avail = {k for k, v in lib["cells"].items() if v["status"] == "available"}
    plans = [json.loads(l) for l in open(
        f"{REPO}/data_wacv_scenario_v5/options/option_plans.jsonl")]

    def cellkey(o):
        a = o["attributes"]
        if not all(a.get(x) for x in ("color", "garment_category", "pattern")):
            return None
        return f"{a['color']}|{a['garment_category']}|{a['pattern']}"

    full = sum(1 for p in plans
               if all(k is not None and k in avail
                      for k in (cellkey(o) for o in p["options"].values())))
    v5 = 100 * full / len(plans)
    v4 = 33.43          # docs/PROCESS.md, 1,012 of 3,027

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # left: before / after
    ax = axes[0]
    ax.barh([1, 0], [v4, v5], 0.34, color=[LGREY, NAVY],
            edgecolor="white", zorder=3)
    ax.text(v4 + 2.5, 1, f"{v4:.1f}%", va="center", fontsize=16,
            fontweight="bold", color=GREY)
    ax.text(v5 - 2.5, 0, f"{v5:.1f}%", va="center", ha="right", fontsize=16,
            fontweight="bold", color="white")
    ax.text(0, 1.30, "v4    availability applied AFTER planning", fontsize=10.5,
            color=GREY, va="bottom")
    ax.text(0, 0.30, "v5    availability applied AS a candidate constraint",
            fontsize=10.5, color=NAVY, va="bottom", fontweight="bold")
    ax.text(v5 + 3.5, 0, f"×{v5/v4:.1f}", color=RED, fontsize=17,
            fontweight="bold", va="center")
    ax.set_yticks([]); ax.set_xlim(0, 122); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("option plans whose 4 images all exist (%)", fontsize=10.5)
    ax.set_ylim(-0.45, 1.72)
    ax.grid(axis="x", color="#EDEDED", zorder=0); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)

    # right: the balance that the post-hoc filter destroyed
    ax = axes[1]
    labels = ["constructed", "v4 survivors", "v5 (released)"]
    phys = [38, 31, 100*sum(1 for p in plans if p["track"] == "physical")/len(plans)]
    ax.barh(range(3), phys, 0.46, color=MID, edgecolor="white", label="Physical", zorder=3)
    ax.barh(range(3), [100 - v for v in phys], 0.46, left=phys, color=LIGHT,
            edgecolor="white", label="Dress-code", zorder=3)
    for i, v in enumerate(phys):
        ax.text(v/2, i, f"{v:.0f}", ha="center", va="center", color="white",
                fontsize=11, fontweight="bold", zorder=4)
        ax.text(v + (100-v)/2, i, f"{100-v:.0f}", ha="center", va="center",
                color=INK, fontsize=11, fontweight="bold", zorder=4)
    ax.text(103, 1, "balance\nbroken", fontsize=10, color=RED, va="center",
            fontweight="bold", linespacing=1.3)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=10.5)
    ax.invert_yaxis(); ax.set_xlim(0, 128); ax.set_xticks([0, 50, 100])
    ax.set_xlabel("track share (%)", fontsize=10.5)
    ax.set_title("a post-hoc filter is not a neutral filter", fontsize=11.5,
                 color=RED, loc="left", fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper center",
              bbox_to_anchor=(0.42, -0.30))
    ax.spines["left"].set_visible(False)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(f"{OUT}/fig_availability.png", bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print(f"fig_availability  v5={v5:.1f}%  ({full}/{len(plans)})")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 — what the benchmark is made of
# ═════════════════════════════════════════════════════════════════════════════
def fig_dataset():
    plans = [json.loads(l) for l in open(
        f"{REPO}/data_wacv_scenario_v5/options/option_plans.jsonl")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 2.5),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    # (a) preference axis share per track
    ax = axes[0]
    tracks = [("physical", "Physical\n24 scenarios"), ("dress_code", "Dress-code\n35 scenarios")]
    axis_order = [("color", "color", MID), ("pattern", "pattern", LIGHT),
                  ("garment_category", "garment", NAVY)]
    for i, (tk, lab) in enumerate(tracks):
        sub = [p for p in plans if p["track"] == tk]
        left = 0
        for a, alab, col in axis_order:
            v = 100 * sum(1 for p in sub if p["active_axis"] == a) / len(sub)
            if v <= 0:
                continue
            ax.barh(i, v, 0.45, left=left, color=col, edgecolor="white", zorder=3)
            ax.text(left + v/2, i, f"{alab}\n{v:.0f}%", ha="center", va="center",
                    fontsize=9.5, color="white" if col != LIGHT else INK,
                    fontweight="bold", zorder=4)
            left += v
        ax.text(101, i, f"{len(sub):,}\nplans", va="center", fontsize=10,
                color=NAVY, fontweight="bold")
    ax.set_yticks(range(2)); ax.set_yticklabels([l for _, l in tracks], fontsize=10.5)
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xticks([])
    ax.set_ylim(1.55, -0.55)
    ax.set_title("preference axis  (per track, never pooled)", fontsize=11.5,
                 color=NAVY, loc="left", fontweight="bold")
    for s in ("left", "bottom"): ax.spines[s].set_visible(False)

    # (b) TPO violation axis
    ax = axes[1]
    for i, (tk, lab) in enumerate(tracks):
        sub = [p for p in plans if p["track"] == tk]
        left = 0
        for a, alab, col in [("garment_category", "garment", NAVY),
                             ("color", "color", MID), ("pattern", "pattern", LIGHT)]:
            v = 100 * sum(1 for p in sub if p["violation_axis"] == a) / len(sub)
            if v <= 0:
                continue
            ax.barh(i, v, 0.45, left=left, color=col, edgecolor="white", zorder=3)
            ax.text(left + v/2, i, f"{alab}\n{v:.0f}%", ha="center", va="center",
                    fontsize=9.5, color="white" if col != LIGHT else INK,
                    fontweight="bold", zorder=4)
            left += v
    ax.set_yticks(range(2)); ax.set_yticklabels(["", ""])
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xticks([])
    ax.set_ylim(1.55, -0.55)
    ax.set_title("TPO violation axis  —  dress-code can put the norm on colour/pattern",
                 fontsize=11.5, color=NAVY, loc="left", fontweight="bold")
    for s in ("left", "bottom"): ax.spines[s].set_visible(False)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{OUT}/fig_dataset.png", bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("fig_dataset ok")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 6 — global pooling loses the pattern; patch grid keeps it
# ═════════════════════════════════════════════════════════════════════════════
def fig_patch():
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.1))
    rng = np.random.default_rng(7)

    # a garment silhouette mask with a plaid patch only on the pocket
    def garment(local=True):
        img = np.ones((60, 44, 3))
        body = np.zeros((60, 44), bool)
        body[6:56, 8:36] = True
        body[10:26, 2:10] = True; body[10:26, 34:42] = True     # sleeves
        img[body] = np.array([0.93, 0.90, 0.83])                # beige
        if local:
            for r in range(34, 46):
                for c in range(14, 26):
                    if (r // 2 + c // 2) % 2 == 0:
                        img[r, c] = np.array([0.35, 0.30, 0.25])
        else:
            for r in range(6, 56):
                for c in range(2, 42):
                    if body[r, c] and (r // 3 + c // 3) % 2 == 0:
                        img[r, c] = np.array([0.35, 0.30, 0.25])
        return img, body

    ax = axes[0]
    img, body = garment(local=True)
    ax.imshow(img); ax.axis("off")
    ax.set_title("retrieved for\n'plaid beige shirt'", fontsize=11,
                 color=INK, fontweight="bold")
    ax.add_patch(plt.Rectangle((13, 33), 13, 13, fill=False, ec=RED, lw=2))
    ax.text(22, 58, "plaid on the pocket only", fontsize=9.5, color=RED, ha="center")

    ax = axes[1]
    ax.imshow(img); ax.axis("off")
    ax.set_title("global average pooling\n(CLIP / SigLIP / reranker)", fontsize=11,
                 color=GREY, fontweight="bold")
    ax.add_patch(plt.Circle((22, 30), 15, fill=True, fc="#00000018", ec=GREY,
                            lw=1.6, ls="--"))
    ax.text(22, 30, "1 vector", ha="center", va="center", fontsize=11,
            color=INK, fontweight="bold")
    ax.text(22, 58, "spatial layout averaged away  ->  PASS", fontsize=9.5,
            color=RED, ha="center")

    ax = axes[2]
    ax.imshow(img); ax.axis("off")
    ax.set_title("per-tile score over the\nSAM3 garment mask", fontsize=11,
                 color=NAVY, fontweight="bold")
    hits = 0; tot = 0
    for r in range(6, 56, 8):
        for c in range(2, 42, 8):
            if not body[r:r+8, c:c+8].any():
                continue
            tot += 1
            plaid = (r >= 32 and r < 48 and c >= 10 and c < 28)
            hits += plaid
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 8, 8, fill=False,
                                       ec=NAVY if plaid else "#B9CCE4",
                                       lw=1.8 if plaid else 0.9))
    ax.text(22, 58, f"target share {hits}/{tot}  ->  REJECT", fontsize=9.5,
            color=NAVY, ha="center", fontweight="bold")

    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{OUT}/fig_patch.png", bbox_inches="tight",
                facecolor="white", pad_inches=0.10)
    plt.close(fig)
    print("fig_patch ok")


if __name__ == "__main__":
    fig_tradeoff()
    fig_breakdown()
    fig_funnel()
    fig_availability()
    fig_dataset()
    fig_patch()
    print("\nall figures ->", OUT)


# ═════════════════════════════════════════════════════════════════════════════
# FIG 7 — one real item: the 2x2 rendered as four garments
# ═════════════════════════════════════════════════════════════════════════════
def fig_abcd():
    import numpy as np
    from matplotlib.patches import Polygon as Poly, FancyBboxPatch

    BLUE_G, ORANGE_G = "#2E5FA3", "#E8801E"

    def hoodie(ax, col):
        ax.add_patch(Poly([[.28,.06],[.72,.06],[.72,.70],[.62,.70],[.62,.80],
                           [.38,.80],[.38,.70],[.28,.70]], closed=True,
                          fc=col, ec="white", lw=1.6))
        ax.add_patch(Poly([[.10,.44],[.28,.70],[.28,.44]], closed=True, fc=col,
                          ec="white", lw=1.6))
        ax.add_patch(Poly([[.90,.44],[.72,.70],[.72,.44]], closed=True, fc=col,
                          ec="white", lw=1.6))
        ax.add_patch(Poly([[.38,.80],[.62,.80],[.56,.90],[.44,.90]], closed=True,
                          fc=col, ec="white", lw=1.6))          # hood

    def dress(ax, col):
        ax.add_patch(Poly([[.18,.04],[.82,.04],[.68,.72],[.32,.72]], closed=True,
                          fc=col, ec="white", lw=1.6))
        ax.add_patch(Poly([[.36,.72],[.44,.72],[.44,.88],[.40,.88]], closed=True,
                          fc=col, ec="white", lw=1.6))
        ax.add_patch(Poly([[.56,.72],[.64,.72],[.60,.88],[.56,.88]], closed=True,
                          fc=col, ec="white", lw=1.6))

    items = [("A", hoodie, BLUE_G,   "liked colour\n+ TPO-fit garment", PURPLE, "both"),
             ("B", hoodie, ORANGE_G, "disliked colour\n+ TPO-fit garment", RED, "TPO only"),
             ("C", dress,  BLUE_G,   "liked colour\n+ TPO-violating garment", BLUE, "Preference only"),
             ("D", dress,  ORANGE_G, "disliked colour\n+ TPO-violating garment", GREY, "neither")]

    fig, axs = plt.subplots(1, 4, figsize=(10.4, 3.5))
    for ax, (letter, draw, col, desc, lcol, tag) in zip(axs, items):
        if letter == "A":
            ax.add_patch(FancyBboxPatch((0.02, 0.0), 0.96, 1.0,
                                        boxstyle="round,pad=0.01,rounding_size=0.04",
                                        fc="#F3EEF8", ec=PURPLE, lw=2.0,
                                        transform=ax.transAxes, zorder=0))
        draw(ax, col)
        ax.set_xlim(0, 1); ax.set_ylim(-0.42, 1.0); ax.axis("off")
        ax.text(.5, .95, letter, ha="center", va="bottom", fontsize=26,
                fontweight="bold", color=lcol)
        ax.text(.5, -.06, tag, ha="center", va="top", fontsize=12,
                fontweight="bold", color=lcol)
        ax.text(.5, -.20, desc, ha="center", va="top", fontsize=10,
                color=INK, linespacing=1.4)
    fig.text(0.5, 0.005, "user likes blue / dislikes orange   ·   blizzard outdoor:  "
                         "hoodie fits, dress does not",
             ha="center", fontsize=10.5, color=GREY, style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=1.0)
    fig.savefig(f"{OUT}/fig_abcd.png", bbox_inches="tight", facecolor="white",
                pad_inches=0.10)
    plt.close(fig)
    print("fig_abcd ok")


fig_abcd()
