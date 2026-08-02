#!/usr/bin/env python3
"""
test_v7_collector_mock.py — synthetic E2E for collect_images_vit_coverage_v7
------------------------------------------------------------------------------
No live servers: knn_search / download_image / vlm_gender / both coverage
scorers are monkeypatched. Verifies:
  [R] coverage_rerank: primary desc -> secondary desc -> sim desc, None == -1
  [O] resolve_option_top1: color-only order, (pattern, color, sim) order,
      fail-open to retrieval order, dedup/gender gates under color order,
      log fields (coverage / cov_reranked / color_coverage / ccov_reranked)
  [M] collect_for_plan --color-cov routing: off / color / all  x  color/pattern axis
  [P] color_coverage_score_urls: HTTP payload fields + fail-open on error
Run: python vit/tests/test_v7_collector_mock.py
"""
import sys, types, json
from pathlib import Path

sys.path.insert(0, ".")
import importlib
v7 = importlib.import_module("vit.collect_images_vit_coverage_v7")

ok = 0
def check(name, cond, detail=""):
    global ok
    assert cond, f"FAIL {name}: {detail}"
    ok += 1
    print(f"  ✓ {name}")

print("== [R] coverage_rerank ==")
cands = [{"similarity": 0.9}, {"similarity": 0.8}, {"similarity": 0.7}, {"similarity": 0.95}]
# primary only: cov desc, then sim desc; None sorts last but keeps sim order
order = v7.coverage_rerank(cands, [0.5, 1.0, None, 0.5])
check("primary desc -> sim tie-break -> None last", order == [1, 3, 0, 2], order)
# primary tie -> secondary breaks it; both None -> sim
order = v7.coverage_rerank(cands, [1.0, 1.0, None, None], [0.2, 0.9, None, None])
check("(primary, secondary, sim) ordering", order == [1, 0, 3, 2], order)
order = v7.coverage_rerank(cands, [None] * 4, [None] * 4)
check("all-None (fail-open) == retrieval/sim order", order == [3, 0, 1, 2], order)

# ── shared fakes ─────────────────────────────────────────────────────────
CANDS = [
    {"url": "http://img/0", "caption": "cand0", "similarity": 0.90},
    {"url": "http://img/1", "caption": "cand1", "similarity": 0.85},
    {"url": "http://img/2", "caption": "cand2", "similarity": 0.80},
    {"url": "http://img/3", "caption": "cand3", "similarity": 0.75},
]
calls = {"pcov": [], "ccov": [], "knn": []}

def fake_knn(client_url, query, top_k=10, indice_name="pod_fashion"):
    calls["knn"].append(query)
    return [dict(c) for c in CANDS]

def fake_download(url, dest, **kw):
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_bytes(b"jpg")
    return True

v7.knn_search = fake_knn
v7.download_image = fake_download

print("\n== [O] resolve_option_top1 ==")
# (O1) color-only re-rank: highest ccov wins even with lowest sim
v7.color_coverage_score_urls = lambda u, urls, color, pattern, garment, grid: \
    (calls["ccov"].append({"color": color, "pattern": pattern}), [0.2, 0.4, 0.0, 1.0])[1]
v7.coverage_score_urls = lambda *a, **k: (_ for _ in ()).throw(AssertionError("pattern cov must not be called"))
res, g, url = v7.resolve_option_top1(
    "q", Path("/tmp/v7t/A.jpg"), "knn://", 4, "pod_fashion",
    None, None, None, set(),
    coverage_url=None, cov_pattern=None, cov_color="navy", cov_garment="coat",
    grid=5, color_coverage_url="cc://", ccov_color="navy", ccov_pattern="solid")
check("color-only: picks max ccov (rank 3, sim 0.75)",
      res["rank_used"] == 3 and url == "http://img/3", res)
check("log fields: color_coverage + ccov_reranked, pattern cov untouched",
      res["color_coverage"] == 1.0 and res["ccov_reranked"] is True
      and res["coverage"] is None and res["cov_reranked"] is False, res)
check("ccov scorer received color + pattern context",
      calls["ccov"][-1] == {"color": "navy", "pattern": "solid"}, calls["ccov"])

# (O2) both scorers: (pattern desc, color desc, sim desc)
v7.coverage_score_urls = lambda u, urls, pattern, color, garment, grid: [1.0, 1.0, 0.5, 1.0]
v7.color_coverage_score_urls = lambda u, urls, color, pattern, garment, grid: [0.2, 0.9, 1.0, 0.9]
res, g, url = v7.resolve_option_top1(
    "q", Path("/tmp/v7t/B.jpg"), "knn://", 4, "pod_fashion",
    None, None, None, set(),
    coverage_url="pc://", cov_pattern="striped", cov_color="navy",
    cov_garment="shirt", grid=5,
    color_coverage_url="cc://", ccov_color="navy", ccov_pattern="striped")
# pcov ties at 1.0 for ranks {0,1,3}; ccov 0.9 ties for {1,3}; sim 0.85>0.75 -> rank 1
check("both: pattern primary, color tie-break, sim last (-> rank 1)",
      res["rank_used"] == 1 and res["coverage"] == 1.0 and res["color_coverage"] == 0.9
      and res["cov_reranked"] and res["ccov_reranked"], res)

# (O3) fail-open: ccov all-None -> retrieval order (rank 0)
v7.color_coverage_score_urls = lambda *a, **k: [None, None, None, None]
res, g, url = v7.resolve_option_top1(
    "q", Path("/tmp/v7t/C.jpg"), "knn://", 4, "pod_fashion",
    None, None, None, set(),
    coverage_url=None, cov_pattern=None, cov_color="navy", cov_garment="coat",
    grid=5, color_coverage_url="cc://", ccov_color="navy", ccov_pattern="solid")
check("fail-open: all-None ccov -> retrieval order (rank 0)",
      res["rank_used"] == 0 and res["color_coverage"] is None, res)

# (O4) dedup + gender gates walk down the COLOR order
v7.color_coverage_score_urls = lambda u, urls, color, pattern, garment, grid: [0.2, 0.4, 0.0, 1.0]
genders = {"http://img/3": "man", "http://img/1": "woman"}  # rank3 conflicts, rank1 ok
v7.vlm_gender = lambda vu, m, p: genders.get(p._test_url, "unclear") if hasattr(p, "_test_url") else "unclear"
# simpler: patch download to tag, and gender to read a queue
gq = ["man", "woman"]                              # 1st checked img -> man (conflict), 2nd -> woman
v7.vlm_gender = lambda vu, m, p: gq.pop(0)
res, g, url = v7.resolve_option_top1(
    "q", Path("/tmp/v7t/D.jpg"), "knn://", 4, "pod_fashion",
    "vlm://", "M", "woman", {"http://img/1"},      # locked woman; rank1 is intra-set dup
    coverage_url=None, cov_pattern=None, cov_color="navy", cov_garment="coat",
    grid=5, color_coverage_url="cc://", ccov_color="navy", ccov_pattern="solid")
# color order = [3, 1, 0, 2]; rank3 gender=man -> skip; rank1 dup -> skip (no VLM call);
# rank0 gender=woman -> PICK
check("gates walk the color order: gender-skip rank3, dup-skip rank1, pick rank0",
      res["rank_used"] == 0 and res["model_gender"] == "woman" and g == "woman", res)
check("dup skipped BEFORE download/VLM (gender queue fully consumed by ranks 3,0)",
      gq == [], gq)

print("\n== [M] collect_for_plan --color-cov routing ==")
seen_kwargs = []
def spy_resolve(query, img_path, *a, **kw):
    seen_kwargs.append(kw)
    return ({"image_path": str(img_path), "source": "vit_covc", "search_query": query,
             "rank_used": 0, "url": f"http://u/{len(seen_kwargs)}", "source_title": "",
             "similarity": 0.9, "coverage": None, "cov_reranked": False,
             "color_coverage": None, "ccov_reranked": False}, "unclear", f"http://u/{len(seen_kwargs)}")
v7.resolve_option_top1 = spy_resolve

def mkplan(axis):
    # color axis: A/C navy vs B/D red, fixed pattern striped (non-solid fixed!)
    # pattern axis: A/C plaid vs B/D solid, fixed color navy
    if axis == "color":
        attrs = {"A": ("navy", "striped"), "B": ("red", "striped"),
                 "C": ("navy", "striped"), "D": ("red", "striped")}
    else:
        attrs = {"A": ("navy", "plaid"), "B": ("navy", "solid"),
                 "C": ("navy", "plaid"), "D": ("navy", "solid")}
    return {"query_id": f"Q_{axis}", "user_id": "U001", "active_axis": axis,
            "options": {k: {"attributes": {"color": c, "pattern": p,
                                           "garment_category": "shirt"},
                            "search_query": "x"} for k, (c, p) in attrs.items()}}

def routing(axis, mode):
    seen_kwargs.clear()
    v7.collect_for_plan(mkplan(axis), "/tmp/v7t/root", "knn://", 4, "pod_fashion",
                        None, "M", coverage_url="pc://", grid=5,
                        color_coverage_url="cc://", color_cov_mode=mode)
    return [(kw.get("ccov_color"), kw.get("cov_pattern"), kw.get("ccov_pattern"))
            for kw in seen_kwargs]   # per option A..D

r = routing("color", "color")
check("mode=color, color axis: every option gets its OWN target color + fixed pattern context",
      [x[0] for x in r] == ["navy", "red", "navy", "red"]
      and all(x[2] == "striped" for x in r) and all(x[1] is None for x in r), r)
r = routing("pattern", "color")
check("mode=color, pattern axis: NO color cov (pure v6), pattern cov per option",
      all(x[0] is None for x in r) and [x[1] for x in r] == ["plaid", "solid", "plaid", "solid"], r)
r = routing("pattern", "all")
check("mode=all, pattern axis: color cov added (fixed color) + own pattern context",
      [x[0] for x in r] == ["navy"] * 4 and [x[2] for x in r] == ["plaid", "solid", "plaid", "solid"], r)
r = routing("color", "off")
check("mode=off: color cov never requested on any axis",
      all(x[0] is None for x in r), r)
r_rec = v7.collect_for_plan(mkplan("color"), "/tmp/v7t/root2", "knn://", 4, "pod_fashion",
                            None, "M", coverage_url="pc://", grid=5,
                            color_coverage_url="cc://", color_cov_mode="color")
check("collection record carries color_cov_mode", r_rec["color_cov_mode"] == "color", r_rec)

print("\n== [P] color_coverage_score_urls HTTP contract ==")
importlib.reload(v7)   # restore unpatched module functions
payloads = []
class FakeResp:
    status_code = 200
    def raise_for_status(self): pass
    def json(self):
        return [{"coverage": 0.7, "top_other": "blue"}, {"error": "x"}, {"coverage": 0.1}]
class FakeSession:
    def post(self, url, json=None, timeout=None):
        payloads.append((url, json)); return FakeResp()
v7._SESSION = FakeSession()
covs = v7.color_coverage_score_urls("cc://x", ["u1", "u2", "u3"], "navy", "striped", "shirt", 5)
check("payload: images/color/pattern/garment/grid/drop_white",
      payloads[0][1] == {"images": ["u1", "u2", "u3"], "color": "navy",
                         "pattern": "striped", "garment": "shirt",
                         "grid": 5, "drop_white": True}, payloads[0])
check("response parsing: coverage kept, error row -> None, aligned to urls",
      covs == [0.7, None, 0.1], covs)
class BoomSession:
    def post(self, *a, **k): raise RuntimeError("server down")
v7._SESSION = BoomSession()
covs = v7.color_coverage_score_urls("cc://x", ["u1", "u2"], "navy", "", "", 5)
check("fail-open on HTTP error -> all None", covs == [None, None], covs)

print(f"\nALL {ok} COLLECTOR CHECKS PASSED")
