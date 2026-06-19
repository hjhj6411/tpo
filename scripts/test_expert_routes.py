#!/usr/bin/env python3
"""
test_expert_routes.py — mock verification of the multi-expert routes
--------------------------------------------------------------------
No GPU / torch / faiss / SAM: heavy deps are stubbed; encode_norm + img_feats
are deterministic one-hot/encoders; the segmenter is the CPU foreground
heuristic (or None). Exercises the REAL code paths: garment-pixel extraction,
colorimetric color coverage on masked pixels, mask-limited pattern tiling,
the conjunctive /verify-option score, fail-open behavior, and error handling.

Run: python tests/test_expert_routes.py
"""
import sys, types
import numpy as np
from PIL import Image
from flask import Flask

sys.path.insert(0, ".")

# ── stub heavy deps so importing the experts is light ───────────────────
for name in ("torch", "open_clip", "faiss"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

from fsiglip.serve_expert_routes import register_expert_routes
from fsiglip.color_expert import COLOR_RGB
from fsiglip.segmentation_expert import GarmentSegmenter, foreground_mask

ok = 0
def check(name, cond, detail=""):
    global ok
    assert cond, f"FAIL {name}: {detail}"
    ok += 1
    print(f"  ✓ {name}")


# ── deterministic fakes for the SigLIP closures ─────────────────────────
# Pattern anchors are 2 rows: [pattern, plain]. We make a tile "patterned" by
# returning a feature closer to row 0. img_feats returns a queue of matrices.
_tile_queue = []
def fake_encode_norm(texts):
    # 2 anchors -> 2x2 identity-ish (pattern=row0, plain=row1)
    return np.eye(len(texts), 2, dtype="float32") if len(texts) <= 2 else \
           np.eye(len(texts), dtype="float32")
def fake_img_feats(tiles):
    feats = _tile_queue.pop(0)
    assert feats.shape[0] == len(tiles), f"tile count {len(tiles)} != {feats.shape[0]}"
    return feats

# the server's real _make_tiles (white-drop) — reuse a faithful copy
def make_tiles(img, grid, drop_white=True, white_thresh=0.95, white_frac=0.85):
    W, H = img.size
    tw, th = W // grid, H // grid
    arr = np.asarray(img).astype("float32") / 255.0
    tiles = []
    for r in range(grid):
        for c in range(grid):
            x1, y1 = c*tw, r*th
            x2 = (c+1)*tw if c < grid-1 else W
            y2 = (r+1)*th if r < grid-1 else H
            if drop_white:
                sub = arr[y1:y2, x1:x2]
                if (sub.min(axis=-1) > white_thresh).mean() > white_frac:
                    continue
            tiles.append(img.crop((x1, y1, x2, y2)))
    return tiles


def build_app(segmenter=None):
    app = Flask(__name__)
    state = {}
    if segmenter is not None:
        state["segmenter"] = segmenter
    register_expert_routes(app, state, load_pil=Image.open,
                           encode_norm=fake_encode_norm, img_feats=fake_img_feats,
                           make_tiles=make_tiles)
    return app.test_client()


def make_garment_img(path, rgb=(0, 0, 128), bg_white=True, size=100):
    """Left/top 60% = garment color, rest = white background (or black)."""
    bg = (255, 255, 255) if bg_white else (0, 0, 0)
    img = Image.new("RGB", (size, size), bg)
    for x in range(int(size*0.6)):
        for y in range(size):
            img.putpixel((x, y), rgb)
    img.save(path)
    return path


print("== [C] /seg-color-coverage (colorimetric, masked) ==")
client = build_app(segmenter=None)   # no segmenter -> foreground heuristic
navy = make_garment_img("/tmp/navy.jpg", COLOR_RGB["navy"], bg_white=True)
r = client.post("/seg-color-coverage",
                json={"image": navy, "color": "navy", "garment": "coat"}).get_json()
check("navy garment -> high navy coverage", r[0]["coverage"] > 0.8, r)
check("foreground heuristic dropped white bg (mask_used=False, pixels<all)",
      r[0]["mask_used"] is False and r[0]["n_pixels"] > 0, r)
r2 = client.post("/seg-color-coverage",
                 json={"image": navy, "color": "blue", "garment": "coat"}).get_json()
check("navy pixels score ~0 for blue target + top_other=navy (ViT fix)",
      r2[0]["coverage"] < 0.2 and r2[0]["top_other"] == "navy", r2)
r3 = client.post("/seg-color-coverage", json={"image": navy}).get_json()
check("missing color -> error row", "error" in r3[0], r3)

print("\n== [P] /seg-pattern-coverage (mask-limited tiles) ==")
# 2x2 grid over a fully-colored image -> 4 tiles; make 3 patterned, 1 plain
img2 = Image.new("RGB", (100, 100), (60, 60, 120)); img2.save("/tmp/p.jpg")
_tile_queue.append(np.array([[1,0],[1,0],[1,0],[0,1]], dtype="float32"))
r = client.post("/seg-pattern-coverage",
                json={"image": "/tmp/p.jpg", "pattern": "striped",
                      "grid": 2}).get_json()
check("coverage = patterned/valid = 3/4", abs(r[0]["coverage"]-0.75) < 1e-6, r)
check("n_valid counts kept tiles", r[0]["n_valid"] == 4 and r[0]["n_pattern"] == 3, r)
# solid target short-circuits to coverage 1.0 (no tiles consumed)
r = client.post("/seg-pattern-coverage",
                json={"image": "/tmp/p.jpg", "pattern": "solid", "grid": 2}).get_json()
check("solid pattern -> coverage 1.0, no tiles consumed", r[0]["coverage"] == 1.0, r)

print("\n== [V] /verify-option (conjunctive pcov*ccov^beta) ==")
# navy + striped: 4 tiles all patterned -> pcov=1.0 ; ccov(navy) high -> score high
navy2 = Image.new("RGB", (100,100), COLOR_RGB["navy"]); navy2.save("/tmp/ns.jpg")
_tile_queue.append(np.array([[1,0]]*4, dtype="float32"))
r = client.post("/verify-option",
                json={"image": "/tmp/ns.jpg", "color": "navy",
                      "pattern": "striped", "garment": "shirt", "grid": 2}).get_json()
check("navy+striped: pcov=1.0, ccov high, score=pcov*ccov", 
      r[0]["pcov"] == 1.0 and r[0]["ccov"] > 0.8 and 
      abs(r[0]["score"] - r[0]["pcov"]*r[0]["ccov"]) < 1e-6, r)
# conjunctive kill: right pattern but WRONG color -> ccov~0 -> score~0
_tile_queue.append(np.array([[1,0]]*4, dtype="float32"))
r = client.post("/verify-option",
                json={"image": "/tmp/ns.jpg", "color": "red",
                      "pattern": "striped", "garment": "shirt", "grid": 2}).get_json()
check("conjunctive: pcov=1.0 but wrong color -> ccov~0 -> score~0 (non-compensatory)",
      r[0]["pcov"] == 1.0 and r[0]["ccov"] < 0.2 and r[0]["score"] < 0.2, r)
# solid garment: pcov defaults 1.0, color drives score
r = client.post("/verify-option",
                json={"image": "/tmp/ns.jpg", "color": "navy",
                      "pattern": "solid", "garment": "shirt", "grid": 2}).get_json()
check("solid: pcov=1.0 default, score == ccov", 
      r[0]["pcov"] == 1.0 and abs(r[0]["score"]-r[0]["ccov"]) < 1e-6, r)

print("\n== [S] segmentation fail-open ==")
seg = GarmentSegmenter(backend="sam3")    # sam3 not installed -> must fail open
check("unavailable backend falls back to heuristic", seg.backend == "heuristic", seg.backend)
check("load_error recorded", seg.load_error is not None, seg.load_error)
m = seg.mask(Image.open(navy), "coat")
check("heuristic backend returns None mask (signals fail-open)", m is None, m)
px = seg.garment_pixels(Image.open(navy), "coat")
check("garment_pixels still returns navy pixels via foreground fallback",
      len(px) > 0, len(px))
# foreground_mask actually drops white background
fm = foreground_mask(Image.open(navy))
check("foreground_mask keeps garment, drops white bg (0<frac<1)",
      0.0 < fm.mean() < 1.0, float(fm.mean()))

print("\n== [E] error handling ==")
r = client.post("/seg-color-coverage",
                json={"images": ["/nonexistent.jpg", navy], "color": "navy"}).get_json()
check("bad src -> error row; good src still scored",
      "error" in r[0] and r[1]["coverage"] > 0.8, r)

print(f"\nALL {ok} EXPERT-ROUTE CHECKS PASSED")
