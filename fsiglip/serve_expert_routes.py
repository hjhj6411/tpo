#!/usr/bin/env python3
"""
serve_expert_routes.py — multi-expert verification routes (drop-in for the
existing serve_fsiglip_knn.py Flask app)
------------------------------------------------------------------------------
Implements the multi-expert pipeline study's recommended verification stage as
new Flask routes that REUSE the already-loaded FashionSigLIP server state and
add a segmentation expert + a colorimetric color expert. The existing routes
(/knn-service, /patch-coverage, /patch-color-coverage, /health) are left intact;
these are additive and can be A/B'd against them.

How to wire into serve_fsiglip_knn.py
-------------------------------------
At the bottom of serve_fsiglip_knn.py, after `app` and `_state` exist and after
`_load(...)` is defined, add:

    from fsiglip.serve_expert_routes import register_expert_routes
    register_expert_routes(app, _state, load_pil=_load_pil_cached,
                           encode_norm=_encode_norm, img_feats=_img_feats_np,
                           make_tiles=_make_tiles)

and (optionally) construct the segmenter once at startup:

    from fsiglip.segmentation_expert import GarmentSegmenter
    _state["segmenter"] = GarmentSegmenter(backend=os.environ.get("SEG_BACKEND","heuristic"))

NEW ROUTES
----------
POST /seg-color-coverage
    Colorimetric COLOR coverage on garment-masked pixels.
    body: {images|image, color, garment?, concept?, k?, min_frac?}
    -> per image: {coverage, primary, top_other, top_other_frac, palette,
                   mask_used, n_pixels}
    This is the segmentation + CIELAB/ΔE2000 replacement for /patch-color-coverage.

POST /seg-pattern-coverage
    PATTERN coverage on garment-masked tiles (mask-limited grid).
    body: {images|image, pattern, color?, garment?, concept?, grid?, region?}
    -> per image: {coverage, n_pattern, n_valid, region, mask_used}
    Tiles whose center falls outside the garment mask are dropped before the
    SigLIP {pattern}-vs-plain argmax, so background/skin tiles no longer count.

POST /verify-option
    One-call conjunctive verification used by the collector to re-rank.
    body: {images|image, color, pattern, garment, concept?, beta?, grid?}
    -> per image: {pcov, ccov, score, primary_color, top_other,
                   pattern_n, pattern_valid, mask_used}
    score = pcov_t * ccov**beta   (conjunctive, non-compensatory; matches the
    existing pcov_t × ccov^β fusion). If pattern is solid, pcov defaults to 1.0
    so color drives the score; similarity stays the caller's tie-break.
"""
from __future__ import annotations

import numpy as np
from flask import request, jsonify

try:
    from fsiglip.color_expert import color_coverage as _color_coverage
except ModuleNotFoundError:
    from color_expert import color_coverage as _color_coverage


_SOLID_LIKE = {"", "solid", "plain", "none", "unknown"}


def _get_images(body):
    imgs = body.get("images")
    if not imgs:
        one = body.get("image")
        imgs = [one] if one else []
    return imgs


def _segmenter(state):
    return state.get("segmenter")          # may be None -> heuristic fallback


def _garment_pixels(state, pil, concept, load_pil):
    """Return (M,3) RGB garment pixels via the segmenter if present, else the
    foreground heuristic (fail-open)."""
    seg = _segmenter(state)
    if seg is not None:
        return seg.garment_pixels(pil, concept=concept), True
    # no segmenter object at all: inline foreground heuristic
    from fsiglip.segmentation_expert import foreground_mask
    rgb = np.asarray(pil.convert("RGB"))
    m = foreground_mask(pil)
    px = rgb[m]
    if len(px) == 0:
        px = rgb.reshape(-1, 3)
    return px, False


def register_expert_routes(app, state, load_pil, encode_norm, img_feats,
                           make_tiles):
    """Attach the expert routes to an existing Flask `app`, reusing the server's
    helper closures from serve_fsiglip_knn.py so no model is reloaded."""

    # ---- /seg-color-coverage : colorimetric color on masked pixels --------
    @app.route("/seg-color-coverage", methods=["POST"])
    def seg_color_coverage():
        body = request.get_json(force=True)
        imgs = _get_images(body)
        color = (body.get("color") or "").strip().lower().replace("_", " ")
        concept = (body.get("concept") or body.get("garment") or "garment")
        concept = concept.replace("_", " ").strip()
        k = int(body.get("k", 4))
        min_frac = float(body.get("min_frac", 0.06))
        if not color:
            return jsonify([{"error": "missing 'color'"} for _ in imgs] or
                           [{"error": "missing 'color'"}])
        out = []
        for src in imgs:
            try:
                pil = load_pil(src)
            except Exception as e:
                out.append({"src": str(src)[:80], "error": str(e)[:120]})
                continue
            px, mask_used = _garment_pixels(state, pil, concept, load_pil)
            r = _color_coverage(px, color, k=k, min_frac=min_frac)
            r.update({"src": str(src)[:80], "mask_used": mask_used,
                      "n_pixels": int(len(px))})
            out.append(r)
        return jsonify(out)

    # ---- /seg-pattern-coverage : SigLIP pattern argmax on masked tiles ----
    @app.route("/seg-pattern-coverage", methods=["POST"])
    def seg_pattern_coverage():
        body = request.get_json(force=True)
        imgs = _get_images(body)
        pattern = (body.get("pattern") or "striped").replace("_", " ").lower()
        color = (body.get("color") or "").replace("_", " ").lower()
        concept = (body.get("concept") or body.get("garment") or "garment")
        concept = concept.replace("_", " ").strip()
        grid = int(body.get("grid", 5))
        region = body.get("region", "body")

        # text anchors: {pattern} fabric vs plain fabric (shared SigLIP scale)
        c = f"{color} " if color and color not in ("none", "unknown") else ""
        pat_txt = encode_norm([f"a close-up of {c}{pattern} patterned fabric",
                               f"a close-up of {c}plain solid-colored fabric"])
        out = []
        seg = _segmenter(state)
        for src in imgs:
            try:
                pil = load_pil(src)
            except Exception as e:
                out.append({"src": str(src)[:80], "error": str(e)[:120]})
                continue
            if pattern in _SOLID_LIKE:
                out.append({"src": str(src)[:80], "coverage": 1.0,
                            "n_pattern": 0, "n_valid": 0, "region": region,
                            "mask_used": seg is not None, "note": "solid->skip"})
                continue
            # garment mask (None => fail open to all tiles)
            gmask = seg.mask(pil, concept) if seg is not None else None
            tiles, keep = _masked_tiles(pil, grid, gmask, region, make_tiles)
            if not tiles:
                out.append({"src": str(src)[:80], "coverage": 0.0,
                            "n_pattern": 0, "n_valid": 0, "region": region,
                            "mask_used": gmask is not None})
                continue
            tfeat = img_feats(tiles)                 # (P,D) normalized
            sims = tfeat @ pat_txt.T                  # (P,2)
            is_pat = sims[:, 0] > sims[:, 1]
            out.append({"src": str(src)[:80],
                        "coverage": float(is_pat.mean()),
                        "n_pattern": int(is_pat.sum()),
                        "n_valid": int(len(tiles)),
                        "region": region,
                        "mask_used": gmask is not None})
        return jsonify(out)

    # ---- /verify-option : one-call conjunctive pcov*ccov^beta -------------
    @app.route("/verify-option", methods=["POST"])
    def verify_option():
        body = request.get_json(force=True)
        imgs = _get_images(body)
        color = (body.get("color") or "").strip().lower().replace("_", " ")
        pattern = (body.get("pattern") or "solid").strip().lower().replace("_", " ")
        concept = (body.get("garment") or body.get("concept") or "garment")
        concept = concept.replace("_", " ").strip()
        beta = float(body.get("beta", 1.0))
        grid = int(body.get("grid", 5))
        k = int(body.get("k", 4))
        min_frac = float(body.get("min_frac", 0.06))

        pat_is_solid = pattern in _SOLID_LIKE
        if not pat_is_solid:
            c = f"{color} " if color and color not in ("none", "unknown") else ""
            pat_txt = encode_norm([f"a close-up of {c}{pattern} patterned fabric",
                                   f"a close-up of {c}plain solid-colored fabric"])
        seg = _segmenter(state)
        out = []
        for src in imgs:
            try:
                pil = load_pil(src)
            except Exception as e:
                out.append({"src": str(src)[:80], "error": str(e)[:120]})
                continue
            gmask = seg.mask(pil, concept) if seg is not None else None

            # ccov on masked pixels
            if gmask is not None:
                rgb = np.asarray(pil.convert("RGB"))
                px = rgb[gmask]
            else:
                px, _ = _garment_pixels(state, pil, concept, load_pil)
            cc = _color_coverage(px, color, k=k, min_frac=min_frac) if color else \
                {"coverage": 1.0, "primary": None, "top_other": None}
            ccov = cc["coverage"]

            # pcov on masked tiles (solid -> 1.0)
            if pat_is_solid:
                pcov, n_pat, n_val = 1.0, 0, 0
            else:
                tiles, _ = _masked_tiles(pil, grid, gmask, "body", make_tiles)
                if tiles:
                    tfeat = img_feats(tiles)
                    sims = tfeat @ pat_txt.T
                    is_pat = sims[:, 0] > sims[:, 1]
                    pcov, n_pat, n_val = float(is_pat.mean()), int(is_pat.sum()), int(len(tiles))
                else:
                    pcov, n_pat, n_val = 0.0, 0, 0

            score = float(pcov * (ccov ** beta))
            out.append({"src": str(src)[:80],
                        "pcov": round(float(pcov), 4),
                        "ccov": round(float(ccov), 4),
                        "score": round(score, 4),
                        "primary_color": cc.get("primary"),
                        "top_other": cc.get("top_other"),
                        "pattern_n": n_pat, "pattern_valid": n_val,
                        "mask_used": gmask is not None})
        return jsonify(out)

    return app


# ════════════════════════════════════════════════════════════════════════
#  masked tiling helper
# ════════════════════════════════════════════════════════════════════════
def _masked_tiles(pil, grid, gmask, region, make_tiles):
    """Split into grid*grid tiles and keep a tile only if its CENTER lies inside
    the garment mask (when a mask is given) AND inside the requested vertical
    region. With no mask, defer to make_tiles' own near-white drop (fail-open).
    Returns (tiles_list, kept_bool_grid)."""
    W, H = pil.size
    if gmask is None:
        # fail open: reuse the server's existing white-dropping tiler
        return make_tiles(pil, grid, drop_white=True), None

    tw, th = W // grid, H // grid
    r0, r1 = 0, grid
    if region == "upper":
        r1 = max(1, int(round(grid * 0.55)))
    elif region == "lower":
        r0 = int(grid * 0.45)
    tiles = []
    keep = np.zeros((grid, grid), dtype=bool)
    for r in range(grid):
        for c in range(grid):
            if not (r0 <= r < r1):
                continue
            x1, y1 = c * tw, r * th
            x2 = (c + 1) * tw if c < grid - 1 else W
            y2 = (r + 1) * th if r < grid - 1 else H
            cy, cx = min(int((y1 + y2) / 2), H - 1), min(int((x1 + x2) / 2), W - 1)
            if gmask[cy, cx]:
                tiles.append(pil.crop((x1, y1, x2, y2)))
                keep[r, c] = True
    return tiles, keep
