#!/usr/bin/env python3
"""
color_expert.py — colorimetric color verification (CIELAB k-means + ΔE2000)
---------------------------------------------------------------------------
Quick-Win #1 from the multi-expert pipeline study: replace the FashionSigLIP
text-anchor color argmax with a classical colorimetric extractor. CPU-only,
no torch, no GPU — pure numpy. The known ViT/SigLIP failure (Navy↔Blue,
Beige↔Brown) is structural for VLM text anchors but deterministic under
CIELAB + ΔE2000, because navy (#000080) and blue (#0000ff) are well separated
on the L* (lightness) axis.

Recipe (after "From Pixels to Posts", arXiv 2511.19149, adapted):
  garment pixels  ->  k-means(k) in RGB
                  ->  drop near-white / near-black clusters (<min_frac)
                  ->  rank remaining clusters by population (primary, secondary, ...)
                  ->  convert each centroid RGB -> CIELAB
                  ->  ΔE2000 nearest color name from a fixed 13-color dictionary
                  ->  ccov = fraction of (kept) pixels whose centroid maps to target

Public API (used by the server route and the collector):
  extract_palette(rgb_pixels, k=4, ...)            -> list[ColorCluster]
  color_coverage(rgb_pixels, target_color, ...)    -> dict(coverage, primary, top_other, ...)
  delta_e_2000(lab1, lab2)                          -> float
  rgb_to_lab(rgb)                                   -> (L, a, b)

The 13-color vocabulary matches POD-Bench's FASHION_ATTRIBUTE_AXES["color"].
All math is vectorized numpy; everything here runs on CPU in microseconds per
candidate, so it adds ~0 latency to the collection pipeline.
"""
from __future__ import annotations

import numpy as np

# ── POD-Bench 13-color vocabulary -> representative sRGB (0-255) ──────────
# Chosen as canonical web/fashion anchors. navy vs blue vs black differ mainly
# on L*, which ΔE2000 captures cleanly (the whole point of this module).
COLOR_RGB = {
    "black":  (0,   0,   0),
    "white":  (255, 255, 255),
    "gray":   (128, 128, 128),
    "navy":   (0,   0,   128),
    "blue":   (0,   0,   255),
    "red":    (220, 20,  40),
    "pink":   (255, 150, 180),
    "orange": (255, 140, 0),
    "yellow": (255, 220, 0),
    "green":  (0,   140, 60),
    "brown":  (120, 75,  40),
    "beige":  (220, 200, 165),
    "purple": (130, 60,  170),
}
DEFAULT_COLORS = list(COLOR_RGB.keys())


# ════════════════════════════════════════════════════════════════════════
#  sRGB -> CIELAB  (D65)  — vectorized, no external color lib required
# ════════════════════════════════════════════════════════════════════════
def _srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb):
    """rgb: (..., 3) array-like in 0-255 -> Lab (..., 3). D65 reference white."""
    rgb = np.asarray(rgb, dtype="float64")
    single = rgb.ndim == 1
    if single:
        rgb = rgb[None, :]
    lin = _srgb_to_linear(rgb)
    # linear sRGB -> XYZ (D65)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T
    # normalize by D65 white
    xyz /= np.array([0.95047, 1.00000, 1.08883])
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    lab = np.stack([L, a, b], axis=-1)
    return lab[0] if single else lab


# ════════════════════════════════════════════════════════════════════════
#  ΔE2000  — CIEDE2000 color difference (vectorized over lab2)
# ════════════════════════════════════════════════════════════════════════
def delta_e_2000(lab1, lab2):
    """ΔE2000 between one Lab color (lab1, shape (3,)) and one-or-many lab2
    (shape (3,) or (N,3)). Returns scalar or (N,) array."""
    lab1 = np.asarray(lab1, dtype="float64")
    lab2 = np.asarray(lab2, dtype="float64")
    single = lab2.ndim == 1
    if single:
        lab2 = lab2[None, :]

    L1, a1, b1 = lab1[0], lab1[1], lab1[2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt((Cbar ** 7) / (Cbar ** 7 + 25.0 ** 7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    # when either chroma is ~0, hue difference is undefined -> 0
    zero_c = (C1p * C2p) == 0
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)
    dHp = np.where(zero_c, 0.0, dHp)

    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0

    hsum = h1p + h2p
    habs = np.abs(h1p - h2p)
    hbarp = np.where(
        zero_c, hsum,
        np.where(habs <= 180, hsum / 2.0,
                 np.where(hsum < 360, (hsum + 360) / 2.0, (hsum - 360) / 2.0)))

    T = (1
         - 0.17 * np.cos(np.radians(hbarp - 30))
         + 0.24 * np.cos(np.radians(2 * hbarp))
         + 0.32 * np.cos(np.radians(3 * hbarp + 6))
         - 0.20 * np.cos(np.radians(4 * hbarp - 63)))

    dtheta = 30 * np.exp(-(((hbarp - 275) / 25.0) ** 2))
    Rc = 2 * np.sqrt((Cbarp ** 7) / (Cbarp ** 7 + 25.0 ** 7))
    Sl = 1 + (0.015 * (Lbarp - 50) ** 2) / np.sqrt(20 + (Lbarp - 50) ** 2)
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc

    de = np.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh))
    return de[0] if single else de


# ════════════════════════════════════════════════════════════════════════
#  k-means (tiny, dependency-free) for palette extraction
# ════════════════════════════════════════════════════════════════════════
def _kmeans(X, k, n_iter=25, seed=0):
    """Minimal k-means on (N,3) RGB. Returns (labels (N,), centers (k,3)).
    Deterministic given seed; k-means++ init for stability."""
    rng = np.random.RandomState(seed)
    n = len(X)
    k = min(k, n)
    # k-means++ init
    centers = [X[rng.randint(n)]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(-1), axis=1)
        s = d2.sum()
        if s <= 0:
            # all points coincide with existing centers (e.g. a solid color
            # image): distance-weighted sampling is undefined -> pick uniformly
            centers.append(X[rng.randint(n)])
            continue
        probs = d2 / s
        probs /= probs.sum()             # guard against float drift (rng.choice is strict)
        centers.append(X[rng.choice(n, p=probs)])
    centers = np.array(centers, dtype="float64")
    labels = np.zeros(n, dtype="int64")
    for _ in range(n_iter):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, labels) and _ > 0:
            labels = new
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                centers[j] = X[m].mean(0)
    return labels, centers


# ════════════════════════════════════════════════════════════════════════
#  Palette extraction + color coverage
# ════════════════════════════════════════════════════════════════════════
class ColorCluster:
    __slots__ = ("rgb", "lab", "frac", "name", "name_delta_e")

    def __init__(self, rgb, lab, frac, name, name_delta_e):
        self.rgb = rgb
        self.lab = lab
        self.frac = frac
        self.name = name
        self.name_delta_e = name_delta_e

    def as_dict(self):
        return {"rgb": [round(float(c), 1) for c in self.rgb],
                "frac": round(float(self.frac), 4),
                "name": self.name,
                "name_delta_e": round(float(self.name_delta_e), 2)}


def _name_of(lab, vocab, vocab_lab):
    des = delta_e_2000(lab, vocab_lab)   # (V,)
    j = int(np.argmin(des))
    return vocab[j], float(des[j])


def extract_palette(rgb_pixels, k=4, min_frac=0.06, drop_extremes=True,
                    vocab=None, seed=0):
    """rgb_pixels: (N,3) garment pixels in 0-255 (already masked / background-
    dropped by the caller). Returns clusters sorted by population (desc), each
    tagged with its nearest vocabulary color name via ΔE2000.

    drop_extremes: remove near-white (L>92) and near-black (L<8) clusters whose
    population < min_frac, since on product shots these are usually residual
    background / shadow rather than the garment's color identity. A dominant
    white/black garment (frac >= min_frac) is KEPT."""
    vocab = vocab or DEFAULT_COLORS
    vocab_lab = rgb_to_lab(np.array([COLOR_RGB[c] for c in vocab], dtype="float64"))

    X = np.asarray(rgb_pixels, dtype="float64").reshape(-1, 3)
    if len(X) == 0:
        return []
    labels, centers = _kmeans(X, k, seed=seed)
    total = len(X)
    clusters = []
    for j in range(len(centers)):
        frac = float((labels == j).mean())
        if frac == 0:
            continue
        rgb = centers[j]
        lab = rgb_to_lab(rgb)
        if drop_extremes and frac < min_frac:
            if lab[0] > 92 or lab[0] < 8:      # near-white / near-black residue
                continue
        name, de = _name_of(lab, vocab, vocab_lab)
        clusters.append(ColorCluster(rgb, lab, frac, name, de))
    # renormalize frac over kept clusters so coverage is over garment-color pixels
    kept = sum(c.frac for c in clusters)
    if kept > 0:
        for c in clusters:
            c.frac = c.frac / kept
    clusters.sort(key=lambda c: -c.frac)
    return clusters


def color_coverage(rgb_pixels, target_color, k=4, min_frac=0.06,
                   vocab=None, seed=0):
    """Colorimetric replacement for /patch-color-coverage.

    coverage = summed population fraction of clusters whose ΔE2000-nearest
    vocabulary name == target_color. Because the nearest-name decision is a
    13-way argmax in perceptual space, a blue cluster loses to 'blue' and
    counts AGAINST a navy target — the exact ViT failure this fixes.

    Returns:
      {coverage, primary (name), top_other (most frequent non-target name),
       top_other_frac, palette (list of cluster dicts)}
    """
    target = (target_color or "").strip().lower().replace("_", " ")
    vocab = vocab or DEFAULT_COLORS
    if target and target not in vocab:
        vocab = [target] + vocab
        if target not in COLOR_RGB:
            # unknown color name with no anchor RGB -> cannot place it; coverage 0
            return {"coverage": 0.0, "primary": None, "top_other": None,
                    "top_other_frac": 0.0, "palette": [],
                    "error": f"no RGB anchor for '{target}'"}

    clusters = extract_palette(rgb_pixels, k=k, min_frac=min_frac,
                               vocab=vocab, seed=seed)
    if not clusters:
        return {"coverage": 0.0, "primary": None, "top_other": None,
                "top_other_frac": 0.0, "palette": []}

    coverage = sum(c.frac for c in clusters if c.name == target)
    others = {}
    for c in clusters:
        if c.name != target:
            others[c.name] = others.get(c.name, 0.0) + c.frac
    if others:
        top_other = max(others, key=others.get)
        top_other_frac = others[top_other]
    else:
        top_other, top_other_frac = None, 0.0
    return {"coverage": round(float(coverage), 4),
            "primary": clusters[0].name,
            "top_other": top_other,
            "top_other_frac": round(float(top_other_frac), 4),
            "palette": [c.as_dict() for c in clusters]}
