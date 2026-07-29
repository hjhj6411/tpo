#!/usr/bin/env python3
"""
test_server_color_route.py — mock verification of /patch-color-coverage
------------------------------------------------------------------------
No GPU / torch / faiss here: torch, open_clip and faiss are stubbed at import
time; _encode_norm and _img_feats_np are monkeypatched with deterministic
one-hot encoders. The REAL code paths exercised are: route parsing, vocab
handling, anchor-prompt construction, _make_tiles white-drop, numpy argmax
coverage math, top_other reporting, error handling, and the URL->PIL cache
helper. Run: python test_server_color_route.py
"""
import sys, types, json
import numpy as np
from PIL import Image

# ── stub heavy deps BEFORE importing the server module ──────────────────
class _NoGrad:
    def __call__(self, fn): return fn          # @torch.no_grad() decorator use
    def __enter__(self): return None
    def __exit__(self, *a): return False

class _ArrayTensor:
    def __init__(self, value): self.value = np.asarray(value)
    def to(self, *a, **k): return self
    @property
    def T(self): return _ArrayTensor(self.value.T)
    def __matmul__(self, other): return _ArrayTensor(self.value @ other.value)
    def __getitem__(self, item): return _ArrayTensor(self.value[item])
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.value

torch_stub = types.ModuleType("torch")
torch_stub.no_grad = _NoGrad
torch_stub.autocast = lambda *a, **k: _NoGrad()
torch_stub.from_numpy = _ArrayTensor
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = torch_stub
sys.modules["open_clip"] = types.ModuleType("open_clip")
sys.modules["faiss"] = types.ModuleType("faiss")

sys.path.insert(0, ".")
import importlib
srv = importlib.import_module("fsiglip.serve_fsiglip_knn")

VOCAB = srv._DEFAULT_COLORS
D = len(VOCAB)
NAVY, BLUE, BEIGE, BROWN = (VOCAB.index(c) for c in ("navy", "blue", "beige", "brown"))

captured_prompts = []

def fake_encode_norm(texts):
    """One-hot row per anchor at its BATCH position. The route builds anchors in
    request-vocab order, so batch position == vocab index; with the default
    13-color vocab this equals the global VOCAB index used by the tile feats."""
    captured_prompts.append(list(texts))
    return np.eye(len(texts), dtype="float32")

# queue of per-image tile-feature matrices the fake image encoder will emit
tile_feats_queue = []

def fake_img_feats(pil_list):
    feats = tile_feats_queue.pop(0)
    assert feats.shape[0] == len(pil_list), \
        f"tile count mismatch: expected {feats.shape[0]} got {len(pil_list)}"
    return feats

srv._encode_norm = fake_encode_norm
srv._img_feats_np = fake_img_feats

def onehot(*idxs):
    m = np.zeros((len(idxs), D), dtype="float32")
    for r, i in enumerate(idxs):
        m[r, i] = 1.0
    return m

def make_img(path, left_rgb=(20, 30, 90), right_white=True):
    """100x100: left half = garment color, right half = pure white background.
    grid=2 -> 4 tiles; with drop_white the 2 right (white) tiles are dropped."""
    img = Image.new("RGB", (100, 100), (255, 255, 255) if right_white else left_rgb)
    for x in range(50):
        for y in range(100):
            img.putpixel((x, y), left_rgb)
    img.save(path)
    return path

client = srv.app.test_client()
ok = 0

def check(name, cond, detail=""):
    global ok
    assert cond, f"FAIL {name}: {detail}"
    ok += 1
    print(f"  ✓ {name}")

print("== /patch-color-coverage mock tests ==")

# 1) navy target, tiles argmax [navy, blue] -> coverage 0.5, top_other=blue
img1 = make_img("/tmp/t1.jpg")
tile_feats_queue.append(onehot(NAVY, BLUE))
captured_prompts.clear()
r = client.post("/patch-color-coverage",
                json={"images": [img1], "color": "navy", "pattern": "solid",
                      "grid": 2, "drop_white": True}).get_json()
check("coverage = n_target/n_valid", abs(r[0]["coverage"] - 0.5) < 1e-9, r)
check("n_valid excludes white background tiles", r[0]["n_valid"] == 2, r)
check("top_other reports the navy<->blue confusion",
      r[0]["top_other"] == "blue" and abs(r[0]["top_other_frac"] - 0.5) < 1e-9, r)
check("solid target -> 'plain {c} colored fabric' anchors",
      all("plain" in p and "colored fabric" in p for p in captured_prompts[0])
      and len(captured_prompts[0]) == D, captured_prompts[0][:2])

# 2) non-solid pattern context: anchors hold the pattern constant, vary color only
img2 = make_img("/tmp/t2.jpg")
tile_feats_queue.append(onehot(NAVY, NAVY))
captured_prompts.clear()
r = client.post("/patch-color-coverage",
                json={"images": [img2], "color": "navy", "pattern": "striped",
                      "grid": 2}).get_json()
check("striped context -> all anchors contain 'striped', one per vocab color",
      all("striped fabric" in p for p in captured_prompts[0])
      and len(captured_prompts[0]) == D, captured_prompts[0][:2])
check("all-target tiles -> coverage 1.0, top_other None",
      r[0]["coverage"] == 1.0 and r[0]["top_other"] is None, r)

# 3) beige target where tiles read brown -> coverage 0 (the ViT failure case
#    this endpoint exists to catch), top_other=brown
img3 = make_img("/tmp/t3.jpg", left_rgb=(150, 120, 90))
tile_feats_queue.append(onehot(BROWN, BROWN))
r = client.post("/patch-color-coverage",
                json={"images": [img3], "color": "beige", "grid": 2}).get_json()
check("brown-reading tiles count AGAINST a beige target",
      r[0]["coverage"] == 0.0 and r[0]["top_other"] == "brown", r)

# 4) underscore + custom vocab + unknown color auto-added to vocab front
img4 = make_img("/tmp/t4.jpg")
tile_feats_queue.append(np.array([[1, 0, 0], [1, 0, 0]], dtype="float32"))  # 2 tiles, idx0 = "ivory" in [ivory,white,beige]
captured_prompts.clear()
r = client.post("/patch-color-coverage",
                json={"images": [img4], "color": "ivory", "pattern": "polka_dot",
                      "colors": ["white", "beige"], "grid": 2}).get_json()
check("unknown target color prepended to custom vocab; pattern underscores cleaned",
      r[0]["coverage"] == 1.0
      and captured_prompts[0][0] == "a close-up of ivory polka dot fabric"
      and len(captured_prompts[0]) == 3, captured_prompts[0])

# 5) missing color -> error rows; bad image src -> per-image error, no crash
r = client.post("/patch-color-coverage",
                json={"images": ["/tmp/t1.jpg"], "pattern": "solid"}).get_json()
check("missing 'color' -> error row", "error" in r[0], r)
tile_feats_queue.append(onehot(NAVY, NAVY))  # only consumed by the GOOD image (2 valid tiles)
img5 = make_img("/tmp/t5.jpg")
r = client.post("/patch-color-coverage",
                json={"images": ["/nonexistent/x.jpg", img5],
                      "color": "navy", "grid": 2, "drop_white": True}).get_json()
check("bad src -> error row, good src still scored",
      "error" in r[0] and r[1].get("n_valid") == 2, r)

# 6) all-white image -> 0 valid tiles -> coverage 0.0 (no div-by-zero)
img6 = "/tmp/t6.jpg"
Image.new("RGB", (100, 100), (255, 255, 255)).save(img6)
r = client.post("/patch-color-coverage",
                json={"images": [img6], "color": "navy", "grid": 2}).get_json()
check("all-background image -> coverage 0.0, n_valid 0",
      r[0]["coverage"] == 0.0 and r[0]["n_valid"] == 0, r)

# 7) URL->PIL cache: second hit on same http URL must NOT re-download
calls = {"n": 0}
def fake_load(src):
    calls["n"] += 1
    return Image.new("RGB", (10, 10), (1, 2, 3))
orig = srv._load_pil
srv._load_pil = fake_load
a = srv._load_pil_cached("http://x/img.jpg")
b = srv._load_pil_cached("http://x/img.jpg")
srv._load_pil_cached("/local/path.jpg") if False else None
srv._load_pil = orig
check("http URL cached (1 download for 2 requests), same object returned",
      calls["n"] == 1 and a is b, calls)

# 8) /patch-coverage route object untouched (still registered, same endpoint fn name)
rules = {r.rule for r in srv.app.url_map.iter_rules()}
check("existing routes intact",
      {"/knn-service", "/knn-service-ensemble", "/patch-coverage",
       "/patch-color-coverage", "/score-image-files", "/health"} <= rules, rules)

# 9) /score-image-files preserves order and per-image errors when batching.
old_load = srv._load_pil_cached
old_encode_images = srv._encode_images_norm
old_encode_text = srv._encode_norm
old_batch_size = srv._SCORE_IMAGE_BATCH_SIZE
old_device = srv._state.get("device")
score_calls = []
score_features = {
    "a": [1.0, 0.0],
    "b": [0.0, 1.0],
    "c": [0.25, 0.75],
}
def fake_score_load(src):
    if src == "bad":
        raise ValueError("bad image")
    return src
def fake_encode_images(images):
    score_calls.append(list(images))
    return _ArrayTensor([score_features[x] for x in images])
srv._load_pil_cached = fake_score_load
srv._encode_images_norm = fake_encode_images
srv._encode_norm = lambda prompts: np.eye(len(prompts), dtype="float32")
srv._SCORE_IMAGE_BATCH_SIZE = 2
srv._state["device"] = "cpu"
r = client.post(
    "/score-image-files",
    json={"texts": {"left": "L", "right": "R"},
          "image_paths": ["a", "b", "bad", "c"]},
)
score_rows = r.get_json()["scores"]
check("score-image batching preserves order and isolates a bad image",
      score_calls == [["a", "b"], ["c"]]
      and score_rows[0] == {"left": 1.0, "right": 0.0}
      and score_rows[1] == {"left": 0.0, "right": 1.0}
      and "error" in score_rows[2]
      and score_rows[3] == {"left": 0.25, "right": 0.75},
      {"calls": score_calls, "rows": score_rows})
srv._load_pil_cached = old_load
srv._encode_images_norm = old_encode_images
srv._encode_norm = old_encode_text
srv._SCORE_IMAGE_BATCH_SIZE = old_batch_size
if old_device is None:
    srv._state.pop("device", None)
else:
    srv._state["device"] = old_device

print(f"\nALL {ok} SERVER CHECKS PASSED")
