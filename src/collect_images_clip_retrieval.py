#!/usr/bin/env python3
"""
collect_images_clip_retrieval.py
--------------------------------
Image collection for POD-Bench options via rom1504/clip-retrieval KNN service.

SELF-CONTAINED: this script no longer imports from src.image_collector. All the
helpers it needs (download_image, google_image_search, AttributeVerifier,
HomogeneityChecker) are defined inline, so it runs cleanly inside a dedicated
`clip` conda environment WITHOUT pulling the pod env's numpy/torch stack.

Why a separate env: clip-retrieval + img2dataset + autofaiss pin numpy/pyarrow
versions that conflict with the pod analysis env. Keep them isolated; this
script only needs requests + Pillow at runtime (VLM checks go over HTTP to your
vLLM server, so no torch is required here).

Two backends:
  --backend hosted  : public LAION-5B knn-service  (fast unblock, no infra)
  --backend local   : your self-hosted `clip-retrieval back`  (controlled corpus)

Quality gates:
  - AttributeVerifier : VLM closed-set garment/color/pattern check (HTTP to vLLM)
  - HomogeneityChecker: SSIM within matched pairs (no torch needed)
    homogeneity is enforced WITHIN the {A,B} and {C,D} pairs only (same garment,
    differ on the active axis) -- never across the A<->C garment gap.

--rewrite modes:
  on  : use vllm LLM to rewrite each search_query into a CLIP-optimised phrase
        before sending to KNN. Improves recall for hard patterns (graphic, polka
        dot, camouflage, etc.).  Adds ~1 LLM call per option per candidate pass.
  off : original augment() behaviour (simple suffix append, no LLM call).

Install (in the clip env, see SETUP_CLIP_ENV.md):
    conda create -n clip python=3.10 -y && conda activate clip
    pip install clip-retrieval img2dataset autofaiss requests pillow scikit-image

Quick test (no infra, hosted LAION):
    python src/collect_images_clip_retrieval.py --backend hosted --limit 5 --verify off

Self-hosted (after `clip-retrieval back --port 1234 ...`):
    python src/collect_images_clip_retrieval.py --backend local \\
        --client-url http://127.0.0.1:1234/knn-service --indice-name pod_fashion \\
        --limit 50 --verify lenient

--verify modes:
    strict  : VLM must match garment AND color AND pattern
    lenient : VLM must match garment AND color (pattern often mis-detected)
    off     : accept first downloadable image (smoke-test only)
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import save_jsonl, load_jsonl, log_step, call_llm, call_vlm, parse_json_response
from configs.config import IMAGES_DIR, OPTIONS_DIR, IMAGE_COLLECTION

UA = {"User-Agent": "Mozilla/5.0 (compatible; POD-Bench/2.0)"}


# ============================================================
#  Query rewriting (Direction A)
# ============================================================

_REWRITE_SYSTEM = """\
You are a fashion image-search expert.
Your job: rewrite a fashion product description into a short, vivid phrase
that maximises recall when used as a CLIP text query against a fashion image index.

Rules:
- Output ONLY the rewritten query string. No explanation, no punctuation other
  than what belongs in the query itself.
- Keep it under 20 words.
- Prefer concrete visual descriptors (color, texture, garment shape, key pattern
  features) over abstract terms.
- Always append: fashion product photo, plain background
- For patterns, include the most visually distinctive feature:
    graphic print  -> front graphic print tee, bold logo, plain background
    polka dot      -> round dot pattern fabric, polka dot, plain background
    plaid/checkered-> plaid tartan grid pattern, plain background
    camouflage     -> military camo pattern fabric, army green, plain background
    striped        -> horizontal/vertical stripes clothing, plain background
    floral         -> floral print flower pattern clothing, plain background
"""


def rewrite_query(raw_query: str, provider: str = "vllm") -> str:
    """
    Use the vllm LLM to rewrite raw_query into a CLIP-optimised search phrase.
    Falls back to the original augment() result if the LLM call fails.

    Designed to be called once per option per collection run. The result is
    cached on the option dict (key 'rewritten_query') so it is only computed
    once per plan execution, even across retry passes.
    """
    try:
        result = call_llm(
            prompt=f"Rewrite this fashion search query for CLIP image retrieval:\n{raw_query}",
            stage="query_rewrite",
            system=_REWRITE_SYSTEM,
            max_tokens=60,
            temperature=0.2,
            provider_override=provider,
        )
        rewritten = result.strip().strip('"').strip("'")
        if rewritten:
            return rewritten
    except Exception as e:
        print(f"    [rewrite] LLM call failed, falling back: {e}")
    return augment(raw_query)


def augment(search_query: str) -> str:
    """Original lightweight prompt engineering (no LLM). Used as fallback."""
    return f"{search_query}, fashion product photo, plain background"


# ============================================================
#  Inline helpers (formerly in src.image_collector)
# ============================================================

def download_image(url, dest_path, min_side=224, timeout=15):
    """Download + validate an image, save as JPEG. Returns True on success."""
    try:
        resp = requests.get(url, timeout=timeout, headers=UA)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return False
    w, h = img.size
    if min(w, h) < min_side:
        return False
    try:
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(dest_path, "JPEG", quality=92)
        return True
    except Exception:
        return False


def google_image_search(query, top_k=5):
    """Google Custom Search fallback. Needs GOOGLE_API_KEY + GOOGLE_CSE_ID; else []."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    cse_id = os.environ.get("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        return []
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cse_id, "q": query,
                    "searchType": "image", "num": min(top_k, 10),
                    "imgType": "photo", "safe": "active"},
            timeout=15)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [{"image_url": it["link"], "title": it.get("title", "")[:120],
                 "source": "google"} for it in items if it.get("link")]
    except Exception as e:
        print(f"    [google] search failed: {e}")
        return []


_PATTERN_VOCAB = ("solid, striped, checkered (incl. plaid/tartan/gingham), "
                  "polka_dot, floral, graphic_print (incl. logo/text/picture print), "
                  "camouflage, animal_print (incl. leopard/zebra)")

VERIFY_SYSTEM = f"""You are a precise fashion product-image inspector.
You are given ONE product image and a target description. Inspect the MAIN
garment (the largest/most prominent clothing item being sold) and judge three
attributes independently.

IMPORTANT rules for reading PATTERN:
- The pattern label must describe what covers the MAJORITY of the main garment's
  body. If the body is one plain color and a pattern appears ONLY on small parts
  (e.g. only the sleeves, only a small chest graphic, only trim/cuffs), the
  garment is "solid", NOT that pattern. Report this via "pattern_coverage".
- Choose seen_pattern from this closed vocabulary (use the head word):
  {_PATTERN_VOCAB}.
- First LOOK and DESCRIBE what you see, then decide the booleans.

Output ONLY JSON (no prose, no markdown):
{{"seen_garment": "...", "seen_color": "...", "seen_pattern": "<one head word from the vocab>",
 "pattern_coverage": "full"/"partial"/"none",
 "garment_match": true/false, "color_match": true/false, "pattern_match": true/false,
 "model_gender": "man"/"woman"/"unclear", "is_clean_product_shot": true/false}}"""


class AttributeVerifier:
    """VLM closed-set verification over HTTP (no local torch).

    vlm_url: if given, this verifier sends every request to that specific vLLM
    endpoint. Running one vLLM server per GPU and giving each worker thread a
    verifier bound to a different endpoint is how we spread load across GPUs.
    """

    def __init__(self, provider_override=None, vlm_url=None):
        self.provider = provider_override
        self.vlm_url = vlm_url

    def verify(self, image_path, attrs, active_axis=None):
        garment = (attrs.get("garment_category") or "").replace("_", " ")
        color = attrs.get("color") or ""
        pattern = (attrs.get("pattern") or "").replace("_", " ")
        target = f"garment={garment}; color={color}; pattern={pattern or 'solid/plain'}"
        axis_hint = ""
        if active_axis == "pattern":
            axis_hint = (" This item's KEY attribute is its pattern, so be strict: "
                         "set pattern_match=true ONLY if the pattern covers the "
                         "majority of the main garment (pattern_coverage='full').")
        prompt = (f"Target description: {target}\n"
                  f"Focus on the single main garment for sale; ignore background, "
                  f"props, and secondary items. Does it match each attribute? "
                  f"Report what you see, the pattern coverage, and the three "
                  f"match booleans.{axis_hint}")
        try:
            kw = dict(image_paths=[str(image_path)], stage="image_verifier",
                      system=VERIFY_SYSTEM, max_tokens=300, temperature=0.0,
                      provider_override=self.provider)
            if self.vlm_url:
                try:
                    resp = call_vlm(prompt, base_url=self.vlm_url, **kw)
                except TypeError:
                    resp = call_vlm(prompt, **kw)
            else:
                resp = call_vlm(prompt, **kw)
            parsed = parse_json_response(resp) or {}
        except Exception as e:
            parsed = {"error": str(e)[:120]}
        parsed["passes"] = bool(parsed.get("garment_match")
                                and parsed.get("color_match")
                                and parsed.get("pattern_match"))
        return parsed

    def check_set_consistency(self, verifs):
        genders = [v.get("model_gender", "unclear") for v in verifs.values()]
        known = [g for g in genders if g in ("man", "woman")]
        return {"genders": genders, "consistent": len(set(known)) <= 1}


class HomogeneityChecker:
    """SSIM-based pairwise homogeneity (no torch/CLIP needed in the clip env)."""

    def _img(self, path):
        return Image.open(path).convert("RGB")

    def ssim_pair(self, path_a, path_b, size=(256, 256)):
        try:
            from skimage.metrics import structural_similarity as ssim
            import numpy as np
        except Exception:
            return -1.0
        try:
            a = np.array(self._img(path_a).resize(size).convert("L"))
            b = np.array(self._img(path_b).resize(size).convert("L"))
            return float(ssim(a, b))
        except Exception:
            return -1.0

    def pair_homogeneity(self, path_a, path_b):
        s = self.ssim_pair(path_a, path_b)
        th = IMAGE_COLLECTION["min_ssim_for_color_variants"]
        return {"ssim": s, "threshold": th, "passed": (s < 0) or (s >= th)}


# ============================================================
#  clip-retrieval KNN client
# ============================================================

class KnnRetriever:
    """Thin wrapper over clip_retrieval.ClipClient (preferred) or raw HTTP."""

    def __init__(self, backend, client_url, indice_name, num_images=20):
        self.backend = backend
        self.indice_name = indice_name
        self.num_images = num_images
        self._client = None

        if backend == "hosted":
            url = "https://knn.laion.ai/knn-service"
            indice_name = indice_name or "laion5B-L-14"
            print("  [knn] WARNING: hosted LAION knn-service is discontinued "
                  "(offline since 2023-12-19). Use --backend local.")
        else:
            url = client_url
        self._http_url = url
        try:
            from clip_retrieval.clip_client import ClipClient, Modality
            self._client = ClipClient(
                url=url, indice_name=indice_name,
                num_images=num_images, modality=Modality.IMAGE)
            print(f"  [knn] ClipClient ready (backend={backend}, indice={indice_name})")
        except Exception as e:
            print(f"  [knn] ClipClient unavailable ({e}); falling back to raw HTTP")

    def search(self, query, top_k=20, garment_key=None):
        out = []
        if self._client is not None:
            try:
                for r in self._client.query(text=query)[:top_k]:
                    url = r.get("url") or r.get("image_url")
                    if url:
                        out.append({"image_url": url,
                                    "title": r.get("caption", "")[:100],
                                    "source": "clip_retrieval"})
                return out
            except Exception as e:
                print(f"    [knn] client query failed: {e}")
        if self._http_url:
            try:
                resp = requests.post(self._http_url, json={
                    "text": query, "modality": "image",
                    "num_images": top_k, "num_result_ids": top_k,
                    "indice_name": self.indice_name}, timeout=60)
                resp.raise_for_status()
                for r in resp.json()[:top_k]:
                    url = r.get("url") or r.get("image_url")
                    if url:
                        out.append({"image_url": url,
                                    "title": r.get("caption", "")[:100],
                                    "source": "clip_retrieval_http"})
            except Exception as e:
                print(f"    [knn] http query failed: {e}")
        return out


# ============================================================
#  Per-plan collection
# ============================================================

_PATTERN_GROUPS = {
    "solid": {"solid", "plain", "none", "ribbed", "knit", "no pattern", "unpatterned"},
    "striped": {"striped", "stripe", "stripes", "pinstripe", "pinstriped",
                "vertical stripes", "horizontal stripes", "vertical stripe"},
    "checkered": {"checkered", "checked", "check", "plaid", "tartan", "gingham",
                  "buffalo plaid", "houndstooth", "lattice", "latticed"},
    "floral": {"floral", "flower", "flowers", "flower print", "daisy"},
    "polka_dot": {"polka dot", "polka dots", "polka-dot", "dotted", "dots", "dot"},
    "graphic_print": {"graphic print", "graphic", "print", "printed", "graphic tee",
                      "logo", "text", "slogan", "printed graphic"},
    "camouflage": {"camouflage", "camo", "military"},
    "animal_print": {"animal print", "animal", "leopard", "leopard print",
                     "cheetah", "zebra", "snake", "snakeskin", "tiger"},
}


def _canon_pattern(text):
    if not text:
        return None
    t = text.strip().lower()
    for grp, members in _PATTERN_GROUPS.items():
        if t in members:
            return grp
    candidates = []
    for grp, members in _PATTERN_GROUPS.items():
        for m in members:
            if m in t:
                candidates.append((len(m), grp, m))
    if not candidates:
        return None
    for kw, grp in (("leopard", "animal_print"), ("animal", "animal_print"),
                    ("camo", "camouflage"), ("polka", "polka_dot"),
                    ("plaid", "checkered"), ("tartan", "checkered"),
                    ("stripe", "striped"), ("floral", "floral")):
        if kw in t:
            return grp
    candidates.sort(reverse=True)
    return candidates[0][1]


def _pattern_matches(target_pattern, vr, require_full_coverage=False):
    if not target_pattern:
        return True
    tgt = _canon_pattern(target_pattern.replace("_", " "))
    if require_full_coverage and tgt not in (None, "solid"):
        cov = (vr.get("pattern_coverage") or "").strip().lower()
        if cov and cov != "full":
            return False
    seen = _canon_pattern(vr.get("seen_pattern", ""))
    if tgt is not None and seen is not None:
        return tgt == seen
    return bool(vr.get("pattern_match"))


def _verify_pass(vr, mode, active_axis=None, target_pattern=None):
    if mode == "off":
        return True
    g, c = vr.get("garment_match"), vr.get("color_match")
    full_cov = (active_axis == "pattern")
    if mode == "strict":
        return bool(g and c and _pattern_matches(target_pattern, vr, full_cov))
    base = bool(g and c)
    if active_axis == "pattern":
        return base and _pattern_matches(target_pattern, vr, require_full_coverage=True)
    if active_axis == "color":
        return base
    return base


def _gender_ok(vr, target_gender):
    if target_gender is None:
        return True
    g = vr.get("model_gender", "unclear")
    return g == "unclear" or g == target_gender


def _norm_title(t):
    return " ".join((t or "").lower().split())[:80]


def _download_one(cands, img_path, attrs, verifier, mode, top_n=24,
                  active_axis=None, target_gender=None, used_titles=None):
    """Try candidates in order; accept the first that passes verification.

    Returns (ok, title, verification_dict).

    NOTE (Direction B hook): this function currently returns on the FIRST
    passing candidate. To implement top-k best-pick in the future, change
    the logic here to collect up to `best_of` passing candidates and then
    call a VLM ranking step before selecting.
    """
    used_titles = used_titles if used_titles is not None else set()
    tmp = Path(str(img_path) + ".tmp.jpg")
    target_pattern = attrs.get("pattern")
    for cand in cands[:top_n]:
        title = cand.get("title", "")
        if _norm_title(title) in used_titles:
            continue
        if not download_image(cand["image_url"], tmp):
            continue
        if mode == "off":
            tmp.rename(img_path)
            return True, title, {"passes": True, "model_gender": "unclear"}
        vr = verifier.verify(tmp, attrs, active_axis=active_axis)
        if _verify_pass(vr, mode, active_axis, target_pattern) and _gender_ok(vr, target_gender):
            tmp.rename(img_path)
            return True, title, vr
        tmp.unlink(missing_ok=True)
    return False, "", {}


def _resolve_option(k, opt, attrs, knn, verifier, out_dir, mode, use_google,
                    top_k, active_axis, target_gender, used_titles,
                    rewrite=True, rewrite_provider="vllm"):
    """Resolve a single option image.

    Query flow:
      1. rewrite_query() via LLM (if rewrite=True) -> KNN search
      2. If mode==strict and step 1 fails: relax to lenient, same candidates
      3. Google fallback (if use_google)

    The rewritten query is stored back onto opt['rewritten_query'] so it is
    logged in the output and not recomputed on retry passes.
    Returns (ok, source, title, vr, img_path).
    """
    img_path = out_dir / f"{k}.jpg"

    # Build (and cache) the CLIP query for this option
    if rewrite:
        if "rewritten_query" not in opt:
            opt["rewritten_query"] = rewrite_query(
                opt["search_query"], provider=rewrite_provider
            )
        q = opt["rewritten_query"]
    else:
        q = augment(opt["search_query"])

    cands = knn.search(q, top_k=top_k, garment_key=attrs.get("garment_category"))

    ok, title, vr = _download_one(cands, img_path, attrs, verifier, mode,
                                  active_axis=active_axis,
                                  target_gender=target_gender,
                                  used_titles=used_titles)
    if ok:
        return True, "clip_retrieval", title, vr, img_path

    if mode == "strict":
        ok, title, vr = _download_one(cands, img_path, attrs, verifier, "lenient",
                                      active_axis=active_axis,
                                      target_gender=target_gender,
                                      used_titles=used_titles)
        if ok:
            return True, "clip_retrieval(relaxed)", title, vr, img_path

    if use_google:
        ok, title, vr = _download_one(
            google_image_search(opt["search_query"], top_k=8),
            img_path, attrs, verifier, mode, top_n=8,
            active_axis=active_axis, target_gender=target_gender,
            used_titles=used_titles)
        if ok:
            return True, "google", title, vr, img_path

    return False, "FAILED", "", {}, img_path


def _attempt_instance(plan, knn, verifier, out_dir, mode, use_google, top_k,
                      active_axis, forced_gender, skip_incomplete,
                      rewrite, rewrite_provider):
    """One full A->B->C->D pass at a fixed gender preference."""
    options, paths, verifs = {}, {}, {}
    used_titles = set()
    target_gender = forced_gender
    for k in "ABCD":
        opt = plan["options"][k]
        attrs = opt["attributes"]
        img_path = out_dir / f"{k}.jpg"
        ok, src, title, vr, img_path = _resolve_option(
            k, opt, attrs, knn, verifier, out_dir, mode, use_google,
            top_k, active_axis, target_gender, used_titles,
            rewrite=rewrite, rewrite_provider=rewrite_provider)
        if ok:
            paths[k] = img_path
            verifs[k] = vr
            used_titles.add(_norm_title(title))
            if target_gender is None:
                g = vr.get("model_gender", "unclear")
                if g in ("man", "woman"):
                    target_gender = g
            options[k] = {"image_path": str(img_path), "source": src,
                          "search_query": opt["search_query"],
                          "rewritten_query": opt.get("rewritten_query"),
                          "source_title": title, "verification": vr}
        else:
            options[k] = {"image_path": None, "source": "FAILED",
                          "search_query": opt["search_query"],
                          "rewritten_query": opt.get("rewritten_query")}
            if skip_incomplete:
                return options, paths, verifs, target_gender, k
    return options, paths, verifs, target_gender, None


def collect_for_plan(plan, knn, checker, verifier, out_root, mode, use_google,
                     top_k=50, skip_incomplete=True,
                     rewrite=True, rewrite_provider="vllm"):
    qid = plan["query_id"]
    out_dir = Path(out_root) / qid
    out_dir.mkdir(parents=True, exist_ok=True)
    active_axis = plan["active_axis"]

    result = {"query_id": qid, "user_id": plan["user_id"], "domain": "fashion",
              "active_axis": active_axis, "main_category": plan.get("main_category"),
              "scenario_archetype": plan.get("scenario_archetype"),
              "options": {}, "all_collected": False, "skipped": False,
              "homogeneity_AB": None, "homogeneity_CD": None,
              "structure_preserved": None, "set_consistency": None}

    if all((out_dir / f"{k}.jpg").exists() for k in "ABCD"):
        paths = {k: out_dir / f"{k}.jpg" for k in "ABCD"}
        verifs = {k: {"passes": True, "model_gender": "unclear"} for k in "ABCD"}
        for k in "ABCD":
            result["options"][k] = {"image_path": str(paths[k]), "source": "cached",
                                    "search_query": plan["options"][k]["search_query"]}
        result["all_collected"] = True
        result["set_consistency"] = verifier.check_set_consistency(verifs)
        result["homogeneity_AB"] = checker.pair_homogeneity(paths["A"], paths["B"])
        result["homogeneity_CD"] = checker.pair_homogeneity(paths["C"], paths["D"])
        result["structure_preserved"] = result["homogeneity_AB"]
        return result

    options, paths, verifs, locked_gender, failed_key = _attempt_instance(
        plan, knn, verifier, out_dir, mode, use_google, top_k,
        active_axis, None, skip_incomplete, rewrite, rewrite_provider)

    if failed_key is not None and locked_gender in ("man", "woman"):
        other = "woman" if locked_gender == "man" else "man"
        _cleanup_partial(out_dir, paths)
        opts2, paths2, verifs2, g2, fk2 = _attempt_instance(
            plan, knn, verifier, out_dir, mode, use_google, top_k,
            active_axis, other, skip_incomplete, rewrite, rewrite_provider)
        if fk2 is None:
            options, paths, verifs, failed_key = opts2, paths2, verifs2, None
        else:
            _cleanup_partial(out_dir, paths2)

    result["options"] = options
    if failed_key is not None:
        result["skipped"] = True
        _cleanup_partial(out_dir, paths)
        return result

    result["all_collected"] = len(paths) == 4
    if result["all_collected"]:
        result["set_consistency"] = verifier.check_set_consistency(verifs)
        result["homogeneity_AB"] = checker.pair_homogeneity(paths["A"], paths["B"])
        result["homogeneity_CD"] = checker.pair_homogeneity(paths["C"], paths["D"])
        if active_axis in ("color", "pattern"):
            result["structure_preserved"] = result["homogeneity_AB"]
    return result


def _cleanup_partial(out_dir, paths):
    for p in list(paths.values()):
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


def run(plan_path, output_path, image_root, backend, client_url, indice_name,
        limit, mode, use_google, top_k, vlm_urls=None, workers=1,
        skip_incomplete=True, rewrite=True, rewrite_provider="vllm"):
    log_step(f"Image Collection via clip-retrieval "
             f"(backend={backend}, verify={mode}, workers={workers}, "
             f"rewrite={'on' if rewrite else 'off'})")
    plans = load_jsonl(plan_path)
    print(f"  Loaded {len(plans)} plans")

    knn = KnnRetriever(backend, client_url, indice_name)
    checker = HomogeneityChecker()

    vlm_urls = vlm_urls or [None]
    verifiers = [AttributeVerifier(vlm_url=u) for u in vlm_urls]

    if Path(output_path).exists():
        existing = load_jsonl(output_path)
        done = {r["query_id"] for r in existing}
        results = existing
        todo = [p for p in plans if p["query_id"] not in done]
        print(f"  Resuming: {len(done)} already in log")
    else:
        results, todo = [], plans
    if limit > 0:
        todo = todo[:limit]
    print(f"  To collect: {len(todo)} instances across "
          f"{len(vlm_urls)} VLM endpoint(s)")
    if rewrite:
        print(f"  Query rewriting: ON (provider={rewrite_provider})")
    else:
        print(f"  Query rewriting: OFF (using augment() fallback)")

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock()

    def worker(idx_plan):
        idx, plan = idx_plan
        verifier = verifiers[idx % len(verifiers)]
        return collect_for_plan(plan, knn, checker, verifier, image_root,
                                mode, use_google, top_k, skip_incomplete,
                                rewrite=rewrite, rewrite_provider=rewrite_provider)

    n_workers = max(1, workers)
    if n_workers == 1:
        for i, plan in enumerate(todo):
            print(f"  [{i+1}/{len(todo)}] {plan['query_id']} ({plan['active_axis']})")
            try:
                results.append(worker((i, plan)))
                if (i + 1) % 10 == 0:
                    save_jsonl(results, output_path)
            except Exception as e:
                print(f"    ERROR {plan['query_id']}: {e}")
            time.sleep(0.05)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(worker, (i, p)): p for i, p in enumerate(todo)}
            done_n = 0
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    r = fut.result()
                    with lock:
                        results.append(r)
                        done_n += 1
                        tag = ("OK" if r["all_collected"]
                               else ("SKIP" if r.get("skipped") else "PARTIAL"))
                        print(f"  [{done_n}/{len(todo)}] {p['query_id']} "
                              f"({p['active_axis']}) -> {tag}")
                        if done_n % 10 == 0:
                            save_jsonl(results, output_path)
                except Exception as e:
                    print(f"    ERROR {p['query_id']}: {e}")

    save_jsonl(results, output_path)
    n_done = sum(1 for r in results if r.get("all_collected"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    print(f"\n  Saved {len(results)}: complete={n_done} "
          f"({n_done/max(len(results),1):.0%}), skipped={n_skip}")
    src_counter, fail_patterns = {}, {}
    for r in results:
        for k in "ABCD":
            o = r["options"].get(k)
            if not o:
                continue
            s = o.get("source", "?")
            src_counter[s] = src_counter.get(s, 0) + 1
            if s == "FAILED":
                kw = (o.get("search_query", "").split() or ["?"])[0]
                fail_patterns[kw] = fail_patterns.get(kw, 0) + 1
    print("  source breakdown:", src_counter)
    if fail_patterns:
        print("  FAILED by leading query word:",
              dict(sorted(fail_patterns.items(), key=lambda x: -x[1])))


def selftest(backend, client_url, indice_name, top_k):
    print(f"=== KNN self-test (backend={backend}, indice={indice_name}) ===")
    knn = KnnRetriever(backend, client_url, indice_name)
    q = augment("navy wool coat")
    cands = knn.search(q, top_k=top_k)
    print(f"  query: {q!r}")
    print(f"  got {len(cands)} candidates")
    for i, c in enumerate(cands[:5]):
        print(f"   [{i}] {c.get('source')} | {c.get('title','')[:60]} | {c['image_url'][:90]}")
    if not cands:
        print("  -> 0 candidates. Check that `clip-retrieval back` is running on "
              "the given --client-url and that --indice-name matches indices_paths.json.")
    return len(cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hosted", "local"], default="local")
    ap.add_argument("--client-url", default="http://127.0.0.1:1234/knn-service")
    ap.add_argument("--indice-name", default="pod_fashion")
    ap.add_argument("--plan_path", type=Path, default=OPTIONS_DIR / "option_plans.jsonl")
    ap.add_argument("--output", type=Path, default=IMAGES_DIR / "collection_log.jsonl")
    ap.add_argument("--image_root", type=Path, default=IMAGES_DIR)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--verify", choices=["strict", "lenient", "off"], default="lenient")
    ap.add_argument("--google", action="store_true", help="enable Google image fallback")
    ap.add_argument("--vlm-urls", default="",
                    help="comma-separated vLLM endpoints, one per GPU")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel instance workers")
    ap.add_argument("--no-skip-incomplete", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rewrite", choices=["on", "off"], default="on",
                    help="LLM query rewriting for better CLIP recall (default: on). "
                         "Use --rewrite off to restore original augment() behaviour.")
    ap.add_argument("--rewrite-provider", default="vllm",
                    help="provider alias for the rewrite LLM (default: vllm)")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.backend, args.client_url, args.indice_name, args.top_k)
        return
    if args.force and Path(args.output).exists():
        Path(args.output).unlink()
    vlm_urls = [u.strip() for u in args.vlm_urls.split(",") if u.strip()] or None
    run(args.plan_path, args.output, args.image_root, args.backend, args.client_url,
        args.indice_name, args.limit, args.verify, args.google, args.top_k,
        vlm_urls=vlm_urls, workers=args.workers,
        skip_incomplete=not args.no_skip_incomplete,
        rewrite=(args.rewrite == "on"),
        rewrite_provider=args.rewrite_provider)


if __name__ == "__main__":
    main()
