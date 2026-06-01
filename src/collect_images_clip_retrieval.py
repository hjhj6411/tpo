#!/usr/bin/env python3
"""
collect_images_clip_retrieval.py
--------------------------------
Image collection for POD-Bench options via rom1504/clip-retrieval KNN service.

Agentic collection loop (CollectionAgent)
-----------------------------------------
Every plan goes through up to MAX_AGENT_ATTEMPTS passes. After each attempt
the agent diagnoses which of three failure modes occurred and adjusts the
next attempt accordingly:

  Case 1 – DUPLICATE
    Same product image selected for two different options (high SSIM).
    Action: REWRITE_QUERY with negative hint derived from the duplicate's
            VLM caption, forcing the next KNN query away from it.
    Active on: pattern axis only (color axis rarely duplicates meaningfully).

  Case 2 – MISSING
    One or more options could not be filled (FAILED). The agent inspects
    per-candidate verification logs to determine the dominant failure cause:
      garment_mismatch  -> rewrite garment term more explicitly
      color_mismatch    -> rewrite color + garment, drop pattern emphasis
      no_candidates     -> RELAX_CONSTRAINT then Google fallback
      pattern_ambiguous -> upgrade to strict pattern query
    Active on: all axes.

  Case 3 – PATTERN_AMBIGUITY
    Detected pre-emptively for hard patterns (checkered, polka_dot,
    graphic_print, camouflage) when active_axis=="pattern".
    Also triggered post-hoc when collected images have pattern_coverage!='full'.
    Active on: pattern axis only.

Axis-based agent config
-----------------------
  color          : rewrite=off, no duplicate check, no pattern check, 1 attempt
  pattern        : rewrite=on,  duplicate check,    pattern check,    3 attempts
  garment_category: rewrite=on, no duplicate check, no pattern check, 2 attempts

Scenario-aware verification
---------------------------
AttributeVerifier.verify() now accepts an optional scenario_context string.
When provided (typically the plan's scenario description + option role A/B/C/D),
the VLM is asked to also judge whether the image "fits" the scenario as a
soft signal. This does NOT change the pass/fail gate (garment+color+pattern
remain the hard criteria) but the judgment is logged in the verification dict
for downstream analysis.

CLI flags
---------
  --rewrite on|off
  --rewrite-provider ALIAS  (default: vllm_vlm)
  --agent-attempts N        (default: 3)
  --verify strict|lenient|off
  --scenario-verify on|off  inject scenario context into VLM verify (default: on)
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path
from collections import Counter

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import save_jsonl, load_jsonl, log_step, call_llm, call_vlm, parse_json_response
from configs.config import IMAGES_DIR, OPTIONS_DIR, IMAGE_COLLECTION

UA = {"User-Agent": "Mozilla/5.0 (compatible; POD-Bench/2.0)"}

_HARD_PATTERNS  = {"checkered", "plaid", "polka_dot", "graphic_print", "camouflage"}
_DUPLICATE_SSIM = 0.82

# Per-axis agent behaviour
AXIS_AGENT_CONFIG = {
    "color": {
        "rewrite":            False,
        "check_duplicate":    False,
        "check_pattern_cov":  False,
        "max_attempts":       1,
    },
    "pattern": {
        "rewrite":            True,
        "check_duplicate":    True,
        "check_pattern_cov":  True,
        "max_attempts":       3,
    },
    "garment_category": {
        "rewrite":            True,
        "check_duplicate":    False,
        "check_pattern_cov":  False,
        "max_attempts":       2,
    },
}
_DEFAULT_AXIS_CONFIG = {
    "rewrite": True, "check_duplicate": False,
    "check_pattern_cov": False, "max_attempts": 2,
}


# ============================================================
#  Query rewriting
# ============================================================

_REWRITE_SYSTEM = """\
You are a fashion image-search expert.
Rewrite a fashion product description into a short, vivid phrase that maximises
recall when used as a CLIP text query against a fashion image index.

Rules:
- Output ONLY the rewritten query string. No explanation, no extra punctuation.
- Keep it under 20 words.
- Prefer concrete visual descriptors (color, texture, garment shape, pattern features).
- Always end with: fashion product photo, plain background
- For hard patterns use the most visually distinctive description:
    graphic print  -> bold front graphic logo print tee
    polka dot      -> round dot pattern covering full garment body
    plaid/checkered-> large plaid tartan grid covering full garment body
    camouflage     -> military camo pattern covering full garment, army green brown
    striped        -> horizontal or vertical stripes covering full garment
    floral         -> floral print flower pattern covering full garment
"""

_REWRITE_AVOID_SYSTEM = """\
You are a fashion image-search expert.
Rewrite a fashion product description into a CLIP query that finds a DIFFERENT
product from the one described in AVOID.

Rules:
- Output ONLY the rewritten query string.
- Keep it under 20 words.
- Explicitly contrast against the AVOID description.
- Always end with: fashion product photo, plain background
"""

_REWRITE_CAUSE_HINTS = {
    "garment_mismatch":  "Focus on the garment type explicitly; name the exact garment category.",
    "color_mismatch":    "Focus on the exact color; drop pattern emphasis temporarily.",
    "pattern_ambiguous": "Require the pattern to cover the FULL garment body; add 'all-over pattern'.",
    "no_candidates":     "Use simpler, broader terms; drop unusual modifiers.",
}


def rewrite_query(raw_query: str, provider: str = "vllm_vlm",
                  avoid_caption: str = "", cause: str = "") -> str:
    try:
        if avoid_caption:
            system = _REWRITE_AVOID_SYSTEM
            prompt = (f"Target: {raw_query}\n"
                      f"AVOID (already selected): {avoid_caption}\n"
                      f"Rewrite for CLIP to find a visually DIFFERENT product.")
        else:
            system = _REWRITE_SYSTEM
            hint = _REWRITE_CAUSE_HINTS.get(cause, "")
            prompt = (f"Rewrite for CLIP image retrieval:\n{raw_query}"
                      + (f"\nHint: {hint}" if hint else ""))
        result = call_llm(
            prompt=prompt, stage="query_rewrite", system=system,
            max_tokens=60, temperature=0.2, provider_override=provider,
        )
        rewritten = result.strip().strip('"').strip("'")
        if rewritten:
            return rewritten
    except Exception as e:
        print(f"    [rewrite] failed ({e}), using augment()")
    return augment(raw_query)


def augment(search_query: str) -> str:
    return f"{search_query}, fashion product photo, plain background"


# ============================================================
#  Inline helpers
# ============================================================

def download_image(url, dest_path, min_side=224, timeout=15):
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
    api_key = os.environ.get("GOOGLE_API_KEY")
    cse_id  = os.environ.get("GOOGLE_CSE_ID")
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


def ssim_score(path_a, path_b, size=(256, 256)):
    try:
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        a = np.array(Image.open(path_a).convert("L").resize(size))
        b = np.array(Image.open(path_b).convert("L").resize(size))
        return float(ssim(a, b))
    except Exception:
        return -1.0


# ============================================================
#  Verification
# ============================================================

_PATTERN_VOCAB = ("solid, striped, checkered (incl. plaid/tartan/gingham), "
                  "polka_dot, floral, graphic_print (incl. logo/text/picture print), "
                  "camouflage, animal_print (incl. leopard/zebra)")

VERIFY_SYSTEM = f"""You are a precise fashion product-image inspector.
You are given ONE product image and a target description.
Inspect the MAIN garment (the largest/most prominent clothing item).

IMPORTANT:
- PATTERN must cover the MAJORITY of the garment body to count.
  If pattern appears only on sleeves/trim/cuffs the garment is 'solid'.
- Choose seen_pattern from: {_PATTERN_VOCAB}.

Output ONLY JSON:
{{"seen_garment": "...", "seen_color": "...", "seen_pattern": "<vocab word>",
 "pattern_coverage": "full"/"partial"/"none",
 "garment_match": true/false, "color_match": true/false, "pattern_match": true/false,
 "model_gender": "man"/"woman"/"unclear", "is_clean_product_shot": true/false,
 "scenario_fit": "good"/"neutral"/"poor"/"n/a"}}"""

VERIFY_SYSTEM_WITH_SCENARIO = (
    VERIFY_SYSTEM.rstrip("}") +
    ",\n \"scenario_fit_reason\": \"one sentence\"}}"
)


def _build_verify_prompt(attrs, active_axis, scenario_context=""):
    garment = (attrs.get("garment_category") or "").replace("_", " ")
    color   = attrs.get("color") or ""
    pattern = (attrs.get("pattern") or "").replace("_", " ")
    target  = f"garment={garment}; color={color}; pattern={pattern or 'solid/plain'}"

    axis_hint = ""
    if active_axis == "pattern":
        axis_hint = (" KEY axis is PATTERN: pattern_match=true ONLY if pattern "
                     "covers MAJORITY of garment body (pattern_coverage='full').")

    scenario_block = ""
    if scenario_context:
        scenario_block = (
            f"\n\nSCENARIO CONTEXT (soft signal only — does NOT affect garment/color/pattern gates):\n"
            f"{scenario_context}\n"
            f"Judge scenario_fit as 'good' if the garment style/formality suits this scenario, "
            f"'poor' if it clearly clashes, 'neutral' otherwise. "
            f"Provide a one-sentence reason in scenario_fit_reason."
        )

    return (
        f"Target: {target}\n"
        f"Judge the main garment only. Report seen values and match booleans.{axis_hint}"
        f"{scenario_block}"
    )


class AttributeVerifier:
    """
    VLM closed-set verification.

    scenario_verify: if True, injects scenario context into the prompt.
    The scenario_fit field is LOGGED ONLY — it never blocks pass/fail.
    """

    def __init__(self, provider_override=None, vlm_url=None, scenario_verify=True):
        self.provider        = provider_override
        self.vlm_url         = vlm_url
        self.scenario_verify = scenario_verify

    def verify(self, image_path, attrs, active_axis=None, scenario_context=""):
        use_scenario = self.scenario_verify and bool(scenario_context)
        system = VERIFY_SYSTEM_WITH_SCENARIO if use_scenario else VERIFY_SYSTEM
        prompt = _build_verify_prompt(attrs, active_axis,
                                      scenario_context if use_scenario else "")
        try:
            kw = dict(image_paths=[str(image_path)], stage="image_verifier",
                      system=system, max_tokens=350, temperature=0.0,
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
        parsed["passes"] = bool(
            parsed.get("garment_match")
            and parsed.get("color_match")
            and parsed.get("pattern_match"))
        return parsed

    def check_set_consistency(self, verifs):
        genders = [v.get("model_gender", "unclear") for v in verifs.values()]
        known   = [g for g in genders if g in ("man", "woman")]
        return {"genders": genders, "consistent": len(set(known)) <= 1}


class HomogeneityChecker:
    def pair_homogeneity(self, path_a, path_b):
        s  = ssim_score(path_a, path_b)
        th = IMAGE_COLLECTION["min_ssim_for_color_variants"]
        return {"ssim": s, "threshold": th, "passed": (s < 0) or (s >= th)}


# ============================================================
#  Scenario context builder
# ============================================================

# Option roles based on label semantics:
#   A = tpo_and_preference (correct: matches both TPO and preference)
#   B = tpo_only           (TPO ok, preference mismatch)
#   C = preference_only    (preference ok, TPO violation)
#   D = neither            (violates both)
_OPTION_ROLES = {
    "A": "appropriate for BOTH the occasion/TPO and the user's personal preference (ideal choice)",
    "B": "appropriate for the occasion/TPO only (user preference NOT satisfied)",
    "C": "matches user preference only (violates the occasion/TPO requirement)",
    "D": "inappropriate for BOTH the occasion/TPO and user preference (worst choice)",
}


def build_scenario_context(plan, option_key):
    """
    Construct a concise scenario description string for the VLM verifier.
    Includes: scenario archetype, TPO event type (if available), and option role.
    Returns empty string if plan has no scenario info (safe fallback).
    """
    parts = []
    archetype = plan.get("scenario_archetype") or ""
    if archetype:
        parts.append(f"Scenario archetype: {archetype}")

    # Try common field names for the TPO event description
    for field in ("scenario_description", "tpo_description",
                  "occasion", "tpo_context", "event"):
        val = plan.get(field) or ""
        if val:
            parts.append(f"Occasion: {val}")
            break

    role = _OPTION_ROLES.get(option_key, "")
    if role:
        parts.append(f"This image is option {option_key}: {role}")

    return "  ".join(parts)


# ============================================================
#  KNN client
# ============================================================

class KnnRetriever:
    def __init__(self, backend, client_url, indice_name, num_images=20):
        self.backend      = backend
        self.indice_name  = indice_name
        self._client      = None
        url               = ("https://knn.laion.ai/knn-service"
                             if backend == "hosted" else client_url)
        self._http_url    = url
        try:
            from clip_retrieval.clip_client import ClipClient, Modality
            self._client = ClipClient(
                url=url, indice_name=indice_name,
                num_images=num_images, modality=Modality.IMAGE)
            print(f"  [knn] ClipClient ready (backend={backend}, indice={indice_name})")
        except Exception as e:
            print(f"  [knn] ClipClient unavailable ({e}); falling back to raw HTTP")

    def search(self, query, top_k=20):
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
#  Pattern helpers
# ============================================================

_PATTERN_GROUPS = {
    "solid":         {"solid", "plain", "none", "ribbed", "knit", "no pattern", "unpatterned"},
    "striped":       {"striped", "stripe", "stripes", "pinstripe", "pinstriped",
                     "vertical stripes", "horizontal stripes", "vertical stripe"},
    "checkered":     {"checkered", "checked", "check", "plaid", "tartan", "gingham",
                     "buffalo plaid", "houndstooth", "lattice", "latticed"},
    "floral":        {"floral", "flower", "flowers", "flower print", "daisy"},
    "polka_dot":     {"polka dot", "polka dots", "polka-dot", "dotted", "dots", "dot"},
    "graphic_print": {"graphic print", "graphic", "print", "printed", "graphic tee",
                     "logo", "text", "slogan", "printed graphic"},
    "camouflage":    {"camouflage", "camo", "military"},
    "animal_print":  {"animal print", "animal", "leopard", "leopard print",
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
    return base


def _gender_ok(vr, target_gender):
    if target_gender is None:
        return True
    g = vr.get("model_gender", "unclear")
    return g == "unclear" or g == target_gender


def _norm_title(t):
    return " ".join((t or "").lower().split())[:80]


# ============================================================
#  Core retrieval
# ============================================================

def _download_one(cands, img_path, attrs, verifier, mode, top_n=24,
                  active_axis=None, target_gender=None, used_titles=None,
                  reject_paths=None, scenario_context=""):
    """
    Try candidates in order; return (ok, title, vr, vr_log).
    vr_log: list of (title, vr) for ALL verified candidates (for Case-2 diagnosis).
    reject_paths: existing image paths; SSIM >= _DUPLICATE_SSIM -> skip (Case-1 guard).
    scenario_context: passed through to verifier for scenario_fit logging.
    """
    used_titles  = used_titles  if used_titles  is not None else set()
    reject_paths = reject_paths if reject_paths is not None else set()
    tmp          = Path(str(img_path) + ".tmp.jpg")
    target_pat   = attrs.get("pattern")
    vr_log       = []

    for cand in cands[:top_n]:
        title = cand.get("title", "")
        if _norm_title(title) in used_titles:
            continue
        if not download_image(cand["image_url"], tmp):
            continue

        # Case-1 duplicate guard
        dup = False
        for rp in reject_paths:
            if ssim_score(tmp, rp) >= _DUPLICATE_SSIM:
                dup = True
                print(f"    [agent] duplicate vs {Path(rp).name}, skipping")
                break
        if dup:
            tmp.unlink(missing_ok=True)
            continue

        if mode == "off":
            tmp.rename(img_path)
            return True, title, {"passes": True, "model_gender": "unclear"}, vr_log

        vr = verifier.verify(tmp, attrs, active_axis=active_axis,
                              scenario_context=scenario_context)
        vr_log.append((title, vr))
        if _verify_pass(vr, mode, active_axis, target_pat) and _gender_ok(vr, target_gender):
            tmp.rename(img_path)
            return True, title, vr, vr_log
        tmp.unlink(missing_ok=True)

    return False, "", {}, vr_log


def _resolve_option(k, opt, attrs, knn, verifier, out_dir, mode, use_google,
                    top_k, active_axis, target_gender, used_titles,
                    rewrite=True, rewrite_provider="vllm_vlm",
                    avoid_caption="", rewrite_cause="",
                    reject_paths=None, scenario_context=""):
    """
    Resolve one option image.
    Returns (ok, source, title, vr, img_path, vr_log).
    """
    img_path = out_dir / f"{k}.jpg"

    if rewrite:
        cache_key = f"rewritten_query__{avoid_caption[:30]}_{rewrite_cause}"
        if cache_key not in opt:
            opt[cache_key] = rewrite_query(
                opt["search_query"],
                provider=rewrite_provider,
                avoid_caption=avoid_caption,
                cause=rewrite_cause,
            )
        q = opt[cache_key]
        opt["rewritten_query"] = q
    else:
        q = augment(opt["search_query"])

    cands = knn.search(q, top_k=top_k)
    ok, title, vr, vr_log = _download_one(
        cands, img_path, attrs, verifier, mode,
        active_axis=active_axis,
        target_gender=target_gender,
        used_titles=used_titles,
        reject_paths=reject_paths,
        scenario_context=scenario_context,
    )
    if ok:
        return True, "clip_retrieval", title, vr, img_path, vr_log

    if mode == "strict":
        ok, title, vr, vr_log2 = _download_one(
            cands, img_path, attrs, verifier, "lenient",
            active_axis=active_axis, target_gender=target_gender,
            used_titles=used_titles, reject_paths=reject_paths,
            scenario_context=scenario_context)
        vr_log += vr_log2
        if ok:
            return True, "clip_retrieval(relaxed)", title, vr, img_path, vr_log

    if use_google:
        gcands = google_image_search(opt["search_query"], top_k=8)
        ok, title, vr, vr_log2 = _download_one(
            gcands, img_path, attrs, verifier, mode, top_n=8,
            active_axis=active_axis, target_gender=target_gender,
            used_titles=used_titles, reject_paths=reject_paths,
            scenario_context=scenario_context)
        vr_log += vr_log2
        if ok:
            return True, "google", title, vr, img_path, vr_log

    return False, "FAILED", "", {}, img_path, vr_log


# ============================================================
#  CollectionAgent
# ============================================================

class CollectionAgent:
    """
    Agentic wrapper with three failure-case handlers.

    Axis-aware: uses AXIS_AGENT_CONFIG to disable unnecessary logic per axis.
    Scenario-aware: passes scenario context to AttributeVerifier for soft
    scenario_fit logging (does not affect hard pass/fail gate).
    """

    def __init__(self, knn, verifier, checker, mode, use_google, top_k,
                 rewrite, rewrite_provider, max_attempts=3,
                 scenario_verify=True):
        self.knn             = knn
        self.verifier        = verifier
        self.checker         = checker
        self.mode            = mode
        self.use_google      = use_google
        self.top_k           = top_k
        self.rewrite         = rewrite
        self.rewrite_provider = rewrite_provider
        self.max_attempts    = max_attempts
        self.scenario_verify = scenario_verify

    # ----------------------------------------------------------------
    # Axis config helpers
    # ----------------------------------------------------------------

    def _axis_cfg(self, active_axis):
        cfg = AXIS_AGENT_CONFIG.get(active_axis, _DEFAULT_AXIS_CONFIG)
        # CLI --rewrite off overrides axis config
        if not self.rewrite:
            cfg = dict(cfg, rewrite=False)
        # CLI --agent-attempts overrides max_attempts only if user raised it
        return cfg

    def _effective_attempts(self, axis_cfg):
        return min(self.max_attempts, axis_cfg["max_attempts"])

    # ----------------------------------------------------------------
    # Public entry
    # ----------------------------------------------------------------

    def collect(self, plan, out_root):
        qid        = plan["query_id"]
        out_dir    = Path(out_root) / qid
        out_dir.mkdir(parents=True, exist_ok=True)
        active_axis = plan["active_axis"]
        axis_cfg    = self._axis_cfg(active_axis)
        attempts    = self._effective_attempts(axis_cfg)

        result = {
            "query_id": qid, "user_id": plan["user_id"], "domain": "fashion",
            "active_axis": active_axis,
            "main_category":      plan.get("main_category"),
            "scenario_archetype": plan.get("scenario_archetype"),
            "options": {}, "all_collected": False, "skipped": False,
            "homogeneity_AB": None, "homogeneity_CD": None,
            "structure_preserved": None, "set_consistency": None,
            "agent_attempts": 0, "agent_actions": [],
        }

        if all((out_dir / f"{k}.jpg").exists() for k in "ABCD"):
            return self._finalise_cached(plan, out_dir, result)

        # Case 3 pre-flight (pattern axis only)
        if axis_cfg["check_pattern_cov"]:
            hard_keys = self._flag_hard_patterns(plan, active_axis)
            if hard_keys:
                print(f"    [agent] Case3 pre-flag hard pattern keys={hard_keys}")
                result["agent_actions"].append(
                    {"case": 3, "trigger": "pre-flight", "keys": hard_keys})

        override      = {}   # {k: {avoid_caption, cause}}
        locked_gender = None
        options = paths = verifs = {}
        failed_key = None

        for attempt in range(1, attempts + 1):
            result["agent_attempts"] = attempt
            print(f"    [agent] attempt {attempt}/{attempts} "
                  f"qid={qid} axis={active_axis}")

            options, paths, verifs, locked_gender, failed_key = \
                self._attempt(plan, out_dir, active_axis, locked_gender,
                              override, axis_cfg)

            # Gender retry
            if failed_key is not None and locked_gender in ("man", "woman"):
                other = "woman" if locked_gender == "man" else "man"
                _cleanup_partial(out_dir, paths)
                opts2, paths2, verifs2, g2, fk2 = \
                    self._attempt(plan, out_dir, active_axis, other,
                                  override, axis_cfg)
                if fk2 is None:
                    options, paths, verifs, failed_key = opts2, paths2, verifs2, None
                    locked_gender = g2
                else:
                    _cleanup_partial(out_dir, paths2)

            all_ok = (failed_key is None) and len(paths) == 4

            if all_ok:
                # Case 1: duplicate check (pattern axis only)
                if axis_cfg["check_duplicate"]:
                    dup_actions = self._check_duplicates(paths, options, plan)
                    if dup_actions:
                        result["agent_actions"] += dup_actions
                        _cleanup_partial(out_dir, paths)
                        for act in dup_actions:
                            override[act["key"]] = {
                                "avoid_caption": act["avoid_caption"], "cause": ""}
                        print(f"    [agent] Case1 dup retry "
                              f"{[a['key'] for a in dup_actions]}")
                        if attempt < attempts:
                            continue

                # Case 3 post-hoc (pattern axis only)
                if axis_cfg["check_pattern_cov"]:
                    cov_actions = self._check_pattern_coverage(
                        verifs, plan, active_axis)
                    if cov_actions and attempt < attempts:
                        result["agent_actions"] += cov_actions
                        _cleanup_partial(out_dir, paths)
                        for act in cov_actions:
                            override[act["key"]] = {
                                "avoid_caption": "", "cause": "pattern_ambiguous"}
                        print(f"    [agent] Case3 cov retry "
                              f"{[a['key'] for a in cov_actions]}")
                        continue
                break

            else:
                # Case 2: diagnose missing
                miss_actions = self._diagnose_missing(options, plan)
                result["agent_actions"] += miss_actions
                if attempt < attempts and miss_actions:
                    _cleanup_partial(out_dir, paths)
                    for act in miss_actions:
                        override[act["key"]] = {
                            "avoid_caption": "", "cause": act["cause"]}
                    print(f"    [agent] Case2 missing retry "
                          f"{[(a['key'], a['cause']) for a in miss_actions]}")
                    continue
                break

        result["options"] = options
        if failed_key is not None or len(paths) < 4:
            result["skipped"] = True
            _cleanup_partial(out_dir, paths)
            return result

        result["all_collected"] = True
        result["set_consistency"] = self.verifier.check_set_consistency(verifs)
        result["homogeneity_AB"] = self.checker.pair_homogeneity(
            paths["A"], paths["B"])
        result["homogeneity_CD"] = self.checker.pair_homogeneity(
            paths["C"], paths["D"])
        if active_axis in ("color", "pattern"):
            result["structure_preserved"] = result["homogeneity_AB"]
        return result

    # ----------------------------------------------------------------
    # Single attempt
    # ----------------------------------------------------------------

    def _attempt(self, plan, out_dir, active_axis, forced_gender,
                 override, axis_cfg):
        options, paths, verifs = {}, {}, {}
        used_titles   = set()
        target_gender = forced_gender
        failed_key    = None

        for k in "ABCD":
            opt   = plan["options"][k]
            attrs = opt["attributes"]
            ov    = override.get(k, {})
            reject_paths = set(str(p) for p in paths.values())

            # Build scenario context for this specific option key
            sc_ctx = ""
            if self.scenario_verify:
                sc_ctx = build_scenario_context(plan, k)

            ok, src, title, vr, img_path, vr_log = _resolve_option(
                k, opt, attrs,
                self.knn, self.verifier, out_dir,
                self.mode, self.use_google, self.top_k,
                active_axis, target_gender, used_titles,
                rewrite=axis_cfg["rewrite"],
                rewrite_provider=self.rewrite_provider,
                avoid_caption=ov.get("avoid_caption", ""),
                rewrite_cause=ov.get("cause", ""),
                reject_paths=reject_paths,
                scenario_context=sc_ctx,
            )
            opt["_last_vr_log"] = vr_log

            if ok:
                paths[k]  = img_path
                verifs[k] = vr
                used_titles.add(_norm_title(title))
                if target_gender is None:
                    g = vr.get("model_gender", "unclear")
                    if g in ("man", "woman"):
                        target_gender = g
                options[k] = {
                    "image_path":      str(img_path),
                    "source":          src,
                    "search_query":    opt["search_query"],
                    "rewritten_query": opt.get("rewritten_query"),
                    "source_title":    title,
                    "verification":    vr,
                    # scenario_fit is inside vr already
                }
            else:
                options[k] = {
                    "image_path":      None,
                    "source":          "FAILED",
                    "search_query":    opt["search_query"],
                    "rewritten_query": opt.get("rewritten_query"),
                }
                failed_key = k
                break

        return options, paths, verifs, target_gender, failed_key

    # ----------------------------------------------------------------
    # Case 1
    # ----------------------------------------------------------------

    def _check_duplicates(self, paths, options, plan):
        actions = []
        keys    = list(paths.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ka, kb = keys[i], keys[j]
                s = ssim_score(paths[ka], paths[kb])
                if s >= _DUPLICATE_SSIM:
                    vr_ka = options[ka].get("verification", {})
                    avoid = (
                        f"{vr_ka.get('seen_color','')} "
                        f"{vr_ka.get('seen_pattern','')} "
                        f"{vr_ka.get('seen_garment','')} "
                        f"\u2014 {options[ka].get('source_title','')}"
                    ).strip()
                    print(f"    [agent] Case1: {ka}\u2194{kb} "
                          f"SSIM={s:.3f}>={_DUPLICATE_SSIM}")
                    actions.append({
                        "case": 1, "key": kb,
                        "ssim": round(s, 3),
                        "avoid_caption": avoid,
                    })
        return actions

    # ----------------------------------------------------------------
    # Case 2
    # ----------------------------------------------------------------

    def _diagnose_missing(self, options, plan):
        actions = []
        for k, opt_res in options.items():
            if opt_res.get("source") != "FAILED":
                continue
            vr_log = plan["options"][k].get("_last_vr_log", [])
            cause  = self._infer_cause(vr_log)
            print(f"    [agent] Case2: option {k} FAILED cause={cause}")
            actions.append({"case": 2, "key": k, "cause": cause})
        return actions

    @staticmethod
    def _infer_cause(vr_log):
        if not vr_log:
            return "no_candidates"
        garment_fails = sum(1 for _, vr in vr_log
                            if not vr.get("garment_match"))
        color_fails   = sum(1 for _, vr in vr_log
                            if vr.get("garment_match")
                            and not vr.get("color_match"))
        pattern_fails = sum(1 for _, vr in vr_log
                            if vr.get("garment_match")
                            and vr.get("color_match")
                            and not vr.get("pattern_match"))
        cov_fails     = sum(1 for _, vr in vr_log
                            if vr.get("pattern_match")
                            and (vr.get("pattern_coverage") or "") not in ("full", ""))
        counts = {
            "garment_mismatch":  garment_fails,
            "color_mismatch":    color_fails,
            "pattern_ambiguous": pattern_fails + cov_fails,
            "no_candidates":     0,
        }
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else "no_candidates"

    # ----------------------------------------------------------------
    # Case 3
    # ----------------------------------------------------------------

    @staticmethod
    def _flag_hard_patterns(plan, active_axis):
        if active_axis != "pattern":
            return []
        return [k for k in "ABCD"
                if _canon_pattern(
                    plan["options"][k]["attributes"].get("pattern", "")
                ) in _HARD_PATTERNS]

    @staticmethod
    def _check_pattern_coverage(verifs, plan, active_axis):
        if active_axis != "pattern":
            return []
        actions = []
        for k, vr in verifs.items():
            pat = plan["options"][k]["attributes"].get("pattern", "")
            if _canon_pattern(pat) not in _HARD_PATTERNS:
                continue
            cov = (vr.get("pattern_coverage") or "").strip().lower()
            if cov and cov != "full":
                print(f"    [agent] Case3: option {k} "
                      f"pattern_coverage={cov}, retry")
                actions.append({"case": 3, "key": k, "pattern_coverage": cov})
        return actions

    # ----------------------------------------------------------------
    # Cached fast path
    # ----------------------------------------------------------------

    def _finalise_cached(self, plan, out_dir, result):
        paths  = {k: out_dir / f"{k}.jpg" for k in "ABCD"}
        verifs = {k: {"passes": True, "model_gender": "unclear"} for k in "ABCD"}
        for k in "ABCD":
            result["options"][k] = {
                "image_path":   str(paths[k]),
                "source":       "cached",
                "search_query": plan["options"][k]["search_query"],
            }
        result["all_collected"] = True
        result["set_consistency"] = self.verifier.check_set_consistency(verifs)
        result["homogeneity_AB"]  = self.checker.pair_homogeneity(
            paths["A"], paths["B"])
        result["homogeneity_CD"]  = self.checker.pair_homogeneity(
            paths["C"], paths["D"])
        result["structure_preserved"] = result["homogeneity_AB"]
        return result


# ============================================================
#  Backward-compat wrapper
# ============================================================

def collect_for_plan(plan, knn, checker, verifier, out_root, mode, use_google,
                     top_k=50, skip_incomplete=True,
                     rewrite=True, rewrite_provider="vllm_vlm",
                     max_agent_attempts=3, scenario_verify=True):
    agent = CollectionAgent(
        knn=knn, verifier=verifier, checker=checker,
        mode=mode, use_google=use_google, top_k=top_k,
        rewrite=rewrite, rewrite_provider=rewrite_provider,
        max_attempts=max_agent_attempts,
        scenario_verify=scenario_verify,
    )
    return agent.collect(plan, out_root)


def _cleanup_partial(out_dir, paths):
    for p in list(paths.values()):
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


# ============================================================
#  run / selftest / main
# ============================================================

def run(plan_path, output_path, image_root, backend, client_url, indice_name,
        limit, mode, use_google, top_k, vlm_urls=None, workers=1,
        skip_incomplete=True, rewrite=True, rewrite_provider="vllm_vlm",
        max_agent_attempts=3, scenario_verify=True):
    log_step(
        f"Image Collection (backend={backend}, verify={mode}, "
        f"workers={workers}, rewrite={'on' if rewrite else 'off'}, "
        f"agent_attempts={max_agent_attempts}, "
        f"scenario_verify={'on' if scenario_verify else 'off'})"
    )
    plans = load_jsonl(plan_path)
    print(f"  Loaded {len(plans)} plans")

    knn     = KnnRetriever(backend, client_url, indice_name)
    checker = HomogeneityChecker()

    vlm_urls  = vlm_urls or [None]
    verifiers = [
        AttributeVerifier(vlm_url=u, scenario_verify=scenario_verify)
        for u in vlm_urls
    ]

    if Path(output_path).exists():
        existing = load_jsonl(output_path)
        done     = {r["query_id"] for r in existing}
        results  = existing
        todo     = [p for p in plans if p["query_id"] not in done]
        print(f"  Resuming: {len(done)} already done")
    else:
        results, todo = [], plans
    if limit > 0:
        todo = todo[:limit]
    print(f"  To collect: {len(todo)} across {len(vlm_urls)} VLM endpoint(s)")

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock()

    def worker(idx_plan):
        idx, plan = idx_plan
        verifier  = verifiers[idx % len(verifiers)]
        return collect_for_plan(
            plan, knn, checker, verifier, image_root,
            mode, use_google, top_k, skip_incomplete,
            rewrite=rewrite, rewrite_provider=rewrite_provider,
            max_agent_attempts=max_agent_attempts,
            scenario_verify=scenario_verify,
        )

    n_workers = max(1, workers)
    if n_workers == 1:
        for i, plan in enumerate(todo):
            print(f"  [{i+1}/{len(todo)}] {plan['query_id']} "
                  f"({plan['active_axis']})")
            try:
                results.append(worker((i, plan)))
                if (i + 1) % 10 == 0:
                    save_jsonl(results, output_path)
            except Exception as e:
                print(f"    ERROR {plan['query_id']}: {e}")
            time.sleep(0.05)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs   = {ex.submit(worker, (i, p)): p
                      for i, p in enumerate(todo)}
            done_n = 0
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    r = fut.result()
                    with lock:
                        results.append(r)
                        done_n += 1
                        tag   = ("OK"   if r["all_collected"]
                                 else ("SKIP" if r.get("skipped") else "PARTIAL"))
                        n_acts = len(r.get("agent_actions", []))
                        print(f"  [{done_n}/{len(todo)}] {p['query_id']} "
                              f"({p['active_axis']}) -> {tag} "
                              f"[att={r.get('agent_attempts',1)}, "
                              f"acts={n_acts}]")
                        if done_n % 10 == 0:
                            save_jsonl(results, output_path)
                except Exception as e:
                    print(f"    ERROR {p['query_id']}: {e}")

    save_jsonl(results, output_path)
    n_done  = sum(1 for r in results if r.get("all_collected"))
    n_skip  = sum(1 for r in results if r.get("skipped"))
    n_acts  = sum(len(r.get("agent_actions", [])) for r in results)
    cases   = Counter(
        a["case"] for r in results for a in r.get("agent_actions", []))
    print(f"\n  Saved {len(results)}: complete={n_done} "
          f"({n_done/max(len(results),1):.0%}), skipped={n_skip}")
    print(f"  Agent actions total={n_acts}, by case: {dict(cases)}")
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
    print(f"=== KNN self-test (backend={backend}) ===")
    knn   = KnnRetriever(backend, client_url, indice_name)
    q     = augment("navy wool coat")
    cands = knn.search(q, top_k=top_k)
    print(f"  query: {q!r}  got {len(cands)} candidates")
    for i, c in enumerate(cands[:5]):
        print(f"   [{i}] {c.get('source')} | {c.get('title','')[:60]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend",        choices=["hosted", "local"], default="local")
    ap.add_argument("--client-url",     default="http://127.0.0.1:1234/knn-service")
    ap.add_argument("--indice-name",    default="pod_fashion")
    ap.add_argument("--plan_path",      type=Path,
                    default=OPTIONS_DIR / "option_plans.jsonl")
    ap.add_argument("--output",         type=Path,
                    default=IMAGES_DIR / "collection_log.jsonl")
    ap.add_argument("--image_root",     type=Path, default=IMAGES_DIR)
    ap.add_argument("--limit",          type=int,  default=0)
    ap.add_argument("--top_k",          type=int,  default=50)
    ap.add_argument("--verify",         choices=["strict", "lenient", "off"],
                    default="lenient")
    ap.add_argument("--google",         action="store_true")
    ap.add_argument("--vlm-urls",       default="")
    ap.add_argument("--workers",        type=int,  default=1)
    ap.add_argument("--no-skip-incomplete", action="store_true")
    ap.add_argument("--selftest",       action="store_true")
    ap.add_argument("--force",          action="store_true")
    ap.add_argument("--rewrite",        choices=["on", "off"], default="on")
    ap.add_argument("--rewrite-provider", default="vllm_vlm")
    ap.add_argument("--agent-attempts", type=int,  default=3)
    ap.add_argument("--scenario-verify", choices=["on", "off"], default="on",
                    help="inject scenario/TPO context into VLM verify prompt "
                         "for soft scenario_fit logging (default: on)")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.backend, args.client_url, args.indice_name, args.top_k)
        return
    if args.force and Path(args.output).exists():
        Path(args.output).unlink()
    vlm_urls = [u.strip() for u in args.vlm_urls.split(",") if u.strip()] or None
    run(
        args.plan_path, args.output, args.image_root,
        args.backend, args.client_url, args.indice_name,
        args.limit, args.verify, args.google, args.top_k,
        vlm_urls=vlm_urls, workers=args.workers,
        skip_incomplete=not args.no_skip_incomplete,
        rewrite=(args.rewrite == "on"),
        rewrite_provider=args.rewrite_provider,
        max_agent_attempts=args.agent_attempts,
        scenario_verify=(args.scenario_verify == "on"),
    )


if __name__ == "__main__":
    main()
