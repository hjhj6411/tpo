#!/usr/bin/env python3
"""
diag_topk_dump.py — top-k 전체 이미지를 폴더에 저장하는 diagnostic 스크립트
-----------------------------------------------------------------------
수집기(collect_images_vit_coverage_v7)가 PICK 1장만 저장하는 것과 달리,
이 스크립트는 KNN top-k 전체 후보를 다시 조회·채점·저장합니다.

정렬 모드 (--sort):
  prod     : pcov * ccov 내림차순, 동점 시 sim  [기본]
  prodsim  : pcov * ccov * sim 내림차순
  floor    : ccov >= threshold 인 후보 중 pcov 내림차순,
             미달 후보는 뒤에 ccov 내림차순으로 append
             --floor-ccov 로 threshold 지정 (기본 0.8)

저장 파일명 예시 (sort=prod):
  B_r00_p0.85_c0.95_x0.81_PICK.jpg   <- PICK (수집기가 실제로 고른 rank)
  B_r01_p1.00_c0.09_x0.09.jpg
  B_r09_p0.85_c0.95_x0.81.jpg

  p = pcov, c = ccov, x = combined score (prod/prodsim/floor-score)
  r = 이 sort 기준의 새 순위 (0-base)

Usage:
  # 특정 query 하나, prod sort
  python scripts/diag_topk_dump.py \\
      --plan_path data/options/option_plans.jsonl \\
      --query-id  U001__cold_blizzard_outdoor__pattern__0002 \\
      --client-url http://127.0.0.1:1234/knn-service \\
      --coverage-url http://127.0.0.1:1235/patch-coverage \\
      --color-coverage-url http://127.0.0.1:1235/patch-color-coverage \\
      --image_root data/diag_topk \\
      --sort prod

  # pattern axis 전체 20개, floor sort (ccov >= 0.7)
  python scripts/diag_topk_dump.py \\
      --plan_path data/options/option_plans.jsonl \\
      --axis pattern --limit 20 \\
      --client-url http://127.0.0.1:1234/knn-service \\
      --coverage-url http://127.0.0.1:1235/patch-coverage \\
      --color-coverage-url http://127.0.0.1:1235/patch-color-coverage \\
      --image_root data/diag_topk \\
      --sort floor --floor-ccov 0.7

  # 세 sort 모두 한 번에 (--sort all)
  python scripts/diag_topk_dump.py ... --sort all
"""
import argparse
import io
import json
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from PIL import Image

# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------
UA = {"User-Agent": "Mozilla/5.0 (compatible; POD-Bench-diag/1.0)"}
_SESSION = requests.Session()
_SESSION.headers.update(UA)
_adapter = HTTPAdapter(pool_connections=16, pool_maxsize=32, max_retries=0)
_SESSION.mount("http://", _adapter)
_SESSION.mount("https://", _adapter)

_SOLID_LIKE = {"", "solid", "plain", "none"}


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _clean(s):
    return (s or "").strip().lower().replace("_", " ")


def _cand_url(c):
    return c.get("url") or c.get("image_url")


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------
def knn_search(client_url, query, top_k, indice_name="pod_fashion"):
    try:
        resp = _SESSION.post(client_url, json={
            "text": query, "modality": "image",
            "num_images": top_k, "num_result_ids": top_k,
            "indice_name": indice_name}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"    [knn] FAILED {query!r}: {e}")
        return []


def score_pattern(coverage_url, urls, pattern, color, garment, grid):
    try:
        resp = _SESSION.post(coverage_url, json={
            "images": list(urls), "pattern": pattern, "color": color,
            "garment": garment, "grid": grid, "drop_white": True}, timeout=300)
        resp.raise_for_status()
        res = resp.json()
        out = [r.get("coverage") if isinstance(r, dict) else None for r in res]
        if len(out) < len(urls):
            out += [None] * (len(urls) - len(out))
        return out[:len(urls)]
    except Exception as e:
        print(f"    [pcov] FAILED: {e}")
        return [None] * len(urls)


def score_color(color_cov_url, urls, color, pattern, garment, grid):
    try:
        resp = _SESSION.post(color_cov_url, json={
            "images": list(urls), "color": color, "pattern": pattern,
            "garment": garment, "grid": grid, "drop_white": True}, timeout=300)
        resp.raise_for_status()
        res = resp.json()
        out = [r.get("coverage") if isinstance(r, dict) else None for r in res]
        if len(out) < len(urls):
            out += [None] * (len(urls) - len(out))
        return out[:len(urls)]
    except Exception as e:
        print(f"    [ccov] FAILED: {e}")
        return [None] * len(urls)


def download_image(url, dest_path, min_side=64, timeout=15):
    try:
        resp = _SESSION.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return False
    if min(img.size) < min_side:
        return False
    try:
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(dest_path, "JPEG", quality=92)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# sort modes
# ---------------------------------------------------------------------------
def _fv(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "None"


def _safe(v, fallback=-1.0):
    return v if isinstance(v, (int, float)) else fallback


def _combined_prod(p, c):
    """pcov * ccov; None treated as 0 when the other is valid, -1 if both None."""
    pv, cv = _safe(p, -1.0), _safe(c, -1.0)
    if pv < 0 and cv < 0:
        return -1.0
    if pv < 0:
        return cv          # only ccov available
    if cv < 0:
        return pv          # only pcov available
    return pv * cv


def sort_prod(indices, pcovs, ccovs, sims):
    """Sort by pcov*ccov desc, then sim desc."""
    return sorted(indices,
                  key=lambda i: (-_combined_prod(pcovs[i], ccovs[i]),
                                 -_safe(sims[i], float("-inf"))))


def sort_prodsim(indices, pcovs, ccovs, sims):
    """Sort by pcov*ccov*sim desc."""
    def score(i):
        p = _safe(pcovs[i], -1.0)
        c = _safe(ccovs[i], -1.0)
        s = _safe(sims[i], 0.0)
        if p < 0 or c < 0:
            return -1.0
        return p * c * s
    return sorted(indices, key=lambda i: -score(i))


def sort_floor(indices, pcovs, ccovs, sims, threshold=0.8):
    """
    ccov >= threshold 나머지 중 pcov 내림순, 동점 sim.
    ccov < threshold 나머지는 ccov 내림순으로 뒤에 append.
    pcov가 없는 코드(color axis):
      ccov desc, sim desc로 모두 정렬.
    """
    has_any_pcov = any(isinstance(pcovs[i], (int, float)) for i in indices)
    if not has_any_pcov:
        # color axis: ccov only
        return sorted(indices,
                      key=lambda i: (-_safe(ccovs[i], -1.0),
                                     -_safe(sims[i], float("-inf"))))

    above = [i for i in indices
             if isinstance(ccovs[i], (int, float)) and ccovs[i] >= threshold]
    below = [i for i in indices
             if not (isinstance(ccovs[i], (int, float)) and ccovs[i] >= threshold)]
    above_sorted = sorted(above,
                          key=lambda i: (-_safe(pcovs[i], -1.0),
                                         -_safe(sims[i], float("-inf"))))
    below_sorted = sorted(below,
                          key=lambda i: (-_safe(ccovs[i], -1.0),
                                         -_safe(sims[i], float("-inf"))))
    return above_sorted + below_sorted


SORT_FNS = {
    "prod":    sort_prod,
    "prodsim": sort_prodsim,
}


# ---------------------------------------------------------------------------
# per-option dump
# ---------------------------------------------------------------------------
def specify_query(garment, target_color, target_pattern, axis):
    g   = _clean(garment)
    c   = _clean(target_color)
    tp  = _clean(target_pattern)
    color_adj    = f"{c} " if c and c not in ("none", "unknown") else ""
    garment_core = f"{color_adj}{g}".replace("  ", " ").strip()
    if axis == "pattern" and tp not in _SOLID_LIKE:
        return (
            f"a fashion catalog photo of a person wearing "
            f"a {garment_core} with an all-over {tp} pattern, "
            f"front view, entire garment visible, plain background"
        ).replace("  ", " ")
    return (
        f"a fashion catalog photo of a person wearing "
        f"a {garment_core}, front view, entire garment visible, plain background"
    ).replace("  ", " ")


def dump_option(opt_key, opt_attrs, axis, out_dir,
                client_url, top_k, indice_name,
                coverage_url, color_cov_url, grid,
                sort_mode, floor_threshold,
                pick_rank_used,            # rank_used stored in collection_log
                ):
    """
    Retrieve top-k, score, sort, download all candidates into out_dir.
    Files are named: {opt_key}_r{nr:02d}_p{pcov}_c{ccov}_x{score}[_PICK].jpg
    Returns list of result dicts.
    """
    attrs          = opt_attrs or {}
    target_color   = _clean(attrs.get("color")   or "")
    target_pattern = _clean(attrs.get("pattern") or "")
    target_garment = _clean(attrs.get("garment_category") or "")

    query = specify_query(target_garment, target_color, target_pattern, axis)

    cands = knn_search(client_url, query, top_k, indice_name)
    if not cands:
        print(f"      [{opt_key}] no candidates")
        return []

    urls = [_cand_url(c) or "" for c in cands]

    # --- score ---
    is_nonsolid = (axis == "pattern") and target_pattern not in _SOLID_LIKE
    use_pcov    = bool(coverage_url and is_nonsolid)
    use_ccov    = bool(color_cov_url and target_color and target_color not in ("none", "unknown"))

    pcovs = score_pattern(coverage_url, urls, target_pattern,
                          target_color, target_garment, grid) if use_pcov \
            else [None] * len(cands)
    ccovs = score_color(color_cov_url, urls, target_color,
                        target_pattern, target_garment, grid) if use_ccov \
            else [None] * len(cands)
    sims  = [c.get("similarity") for c in cands]

    # --- sort ---
    idx = list(range(len(cands)))
    if sort_mode == "prod":
        order = sort_prod(idx, pcovs, ccovs, sims)
    elif sort_mode == "prodsim":
        order = sort_prodsim(idx, pcovs, ccovs, sims)
    else:  # floor
        order = sort_floor(idx, pcovs, ccovs, sims, threshold=floor_threshold)

    # --- combined score for filename ---
    def _xscore(rank, mode):
        p = pcovs[rank]; c = ccovs[rank]; s = sims[rank]
        if mode == "prod":
            return _combined_prod(p, c)
        if mode == "prodsim":
            pv, cv, sv = _safe(p, -1), _safe(c, -1), _safe(s, 0)
            return pv * cv * sv if pv >= 0 and cv >= 0 else -1.0
        # floor: use pcov if above threshold, else ccov
        if isinstance(c, (int, float)) and c >= floor_threshold:
            return _safe(p, -1.0)
        return _safe(c, -1.0)

    # --- download ---
    rows = []
    print(f"      [{opt_key}] query: {query[:80]}")
    print(f"         {'nr':>2} {'retr':>4} {'pcov':>5} {'ccov':>5} {'xscore':>7} {'sim':>7}  {'pick':>4}  caption")

    for nr, rank in enumerate(order):
        c   = cands[rank]
        url = _cand_url(c)
        p   = pcovs[rank]
        cv  = ccovs[rank]
        s   = sims[rank]
        xs  = _xscore(rank, sort_mode)
        is_pick = (rank == pick_rank_used)

        pstr  = f"{p:.2f}"  if isinstance(p,  (int, float)) else "None"
        cstr  = f"{cv:.2f}" if isinstance(cv, (int, float)) else "None"
        xstr  = f"{xs:.2f}" if xs >= 0 else "None"
        sstr  = f"{s:.4f}"  if isinstance(s,  (int, float)) else "None"
        mark  = "PICK" if is_pick else ""

        fname = f"{opt_key}_r{nr:02d}_p{pstr}_c{cstr}_x{xstr}"
        if is_pick:
            fname += "_PICK"
        fname += ".jpg"
        dest  = out_dir / fname

        ok = False
        if url:
            ok = download_image(url, dest)

        cap   = (c.get("caption") or "")[:50]
        status = "OK" if ok else "FAIL"
        print(f"         {nr:>2} {rank:>4} {pstr:>5} {cstr:>5} {xstr:>7} {sstr:>7}  {mark:>4}  {cap}  [{status}]")

        rows.append({
            "opt_key": opt_key, "new_rank": nr, "retr_rank": rank,
            "pcov": p, "ccov": cv, "xscore": xs if xs >= 0 else None,
            "sim": s, "is_pick": is_pick, "downloaded": ok,
            "caption": cap, "url": url, "dest": str(dest),
        })
    return rows


# ---------------------------------------------------------------------------
# per-query
# ---------------------------------------------------------------------------
def dump_query(plan, log_rec,
               image_root, sort_mode, floor_threshold,
               client_url, indice_name, top_k, grid,
               coverage_url, color_cov_url):
    qid  = plan["query_id"]
    axis = plan.get("active_axis", "color")
    out_dir = Path(image_root) / sort_mode / qid
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [{qid}]  axis={axis}  sort={sort_mode}")

    all_rows = []
    for k in "ABCD":
        opt = plan["options"].get(k, {})
        attrs = opt.get("attributes") or {}

        # pick_rank_used from collection log (may be None if log unavailable)
        pick_rank = None
        if log_rec:
            log_opt = (log_rec.get("options") or {}).get(k) or {}
            pick_rank = log_opt.get("rank_used")

        rows = dump_option(
            opt_key=k, opt_attrs=attrs, axis=axis, out_dir=out_dir,
            client_url=client_url, top_k=top_k, indice_name=indice_name,
            coverage_url=coverage_url, color_cov_url=color_cov_url, grid=grid,
            sort_mode=sort_mode, floor_threshold=floor_threshold,
            pick_rank_used=pick_rank,
        )
        for r in rows:
            r["query_id"] = qid
            r["axis"]     = axis
            r["sort_mode"] = sort_mode
        all_rows.extend(rows)
        time.sleep(0.05)

    return all_rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Download all top-k candidates per option with pcov/ccov scoring and 3 sort modes.")
    # data
    ap.add_argument("--plan_path", type=Path,
                    default=Path("data/options/option_plans.jsonl"))
    ap.add_argument("--log", type=Path,
                    default=Path("data/images_vit_cov_c/collection_log.jsonl"),
                    help="collection_log.jsonl to read PICK rank_used from; optional")
    ap.add_argument("--image_root", type=Path,
                    default=Path("data/diag_topk"),
                    help="root folder; subfolders: {sort_mode}/{query_id}/")
    ap.add_argument("--out-jsonl", type=Path, default=None,
                    help="save per-candidate summary rows to this jsonl")
    # filter
    ap.add_argument("--query-id",  default=None, help="run only this query_id")
    ap.add_argument("--axis",      default=None, choices=["color", "pattern"])
    ap.add_argument("--limit",     type=int, default=0, help="0 = all")
    # servers
    ap.add_argument("--client-url",
                    default="http://127.0.0.1:1234/knn-service")
    ap.add_argument("--coverage-url",
                    default="http://127.0.0.1:1235/patch-coverage")
    ap.add_argument("--color-coverage-url",
                    default="http://127.0.0.1:1235/patch-color-coverage")
    ap.add_argument("--indice-name", default="pod_fashion")
    ap.add_argument("--top_k",  type=int, default=12)
    ap.add_argument("--grid",   type=int, default=5)
    # sort
    ap.add_argument("--sort",
                    choices=["prod", "prodsim", "floor", "all"],
                    default="prod",
                    help=(
                        "prod     : pcov*ccov desc, sim desc  [default]\n"
                        "prodsim  : pcov*ccov*sim desc\n"
                        "floor    : ccov>=threshold 중 pcov desc, 나머지 ccov desc\n"
                        "all      : 세 종류 모두 실행 (세 하위폴더 생성)"
                    ))
    ap.add_argument("--floor-ccov", type=float, default=0.8,
                    help="floor sort의 ccov threshold (default 0.8)")
    args = ap.parse_args()

    # --- load plans ---
    plans = load_jsonl(args.plan_path)
    if args.query_id:
        plans = [p for p in plans if p["query_id"] == args.query_id]
    if args.axis:
        plans = [p for p in plans if p.get("active_axis") == args.axis]
    if args.limit > 0:
        plans = plans[:args.limit]
    print(f"Plans to dump: {len(plans)}")

    # --- load log (optional) ---
    log_map = {}
    if args.log.exists():
        for r in load_jsonl(args.log):
            log_map[r["query_id"]] = r
        print(f"Collection log loaded: {len(log_map)} entries")
    else:
        print("Collection log not found; PICK markers will be absent.")

    coverage_url   = args.coverage_url.strip() or None
    color_cov_url  = args.color_coverage_url.strip() or None

    sort_modes = ["prod", "prodsim", "floor"] if args.sort == "all" else [args.sort]

    all_rows = []
    for sort_mode in sort_modes:
        print(f"\n{'='*70}")
        print(f"  SORT MODE: {sort_mode}" +
              (f"  (floor_ccov>={args.floor_ccov})" if sort_mode == "floor" else ""))
        print(f"{'='*70}")
        for plan in plans:
            qid     = plan["query_id"]
            log_rec = log_map.get(qid)
            rows = dump_query(
                plan=plan, log_rec=log_rec,
                image_root=args.image_root,
                sort_mode=sort_mode,
                floor_threshold=args.floor_ccov,
                client_url=args.client_url,
                indice_name=args.indice_name,
                top_k=args.top_k, grid=args.grid,
                coverage_url=coverage_url,
                color_cov_url=color_cov_url,
            )
            all_rows.extend(rows)

    # --- summary ---
    n_total  = len(all_rows)
    n_ok     = sum(1 for r in all_rows if r.get("downloaded"))
    n_pick   = sum(1 for r in all_rows if r.get("is_pick"))
    print(f"\nDone. candidates={n_total}, downloaded={n_ok}, PICK_marked={n_pick}")
    print(f"Images in: {args.image_root}/{{sort_mode}}/{{query_id}}/")

    if args.out_jsonl:
        Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_jsonl, "w") as f:
            for r in all_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Summary JSONL -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
