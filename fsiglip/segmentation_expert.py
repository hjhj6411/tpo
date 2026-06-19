#!/usr/bin/env python3
"""
segmentation_expert.py — garment mask provider (SAM 3 / Grounded-SAM / fallback)
--------------------------------------------------------------------------------
Medium-Term #1 from the multi-expert pipeline study: replace the 5x5 uniform
grid (which lets background / skin / hair tiles pollute color & pattern argmax)
with a garment-segmentation mask so verification scores ONLY garment pixels.

Design constraints honored:
  * frozen / zero-shot — no fine-tuning.
  * fail-open — if the segmentation backend is unavailable, the model fails to
    load, or it returns no/low-confidence mask, we fall back to a whole-image
    foreground heuristic (the same near-white/near-black drop the grid used),
    so the pipeline degrades to "v6-ish" behavior rather than crashing.
  * lazy import — heavy deps (torch, sam3, groundingdino) are imported only when
    a real backend is selected, so the module (and its tests) run with numpy+PIL
    only. The default backend is "heuristic".

Public API:
  seg = GarmentSegmenter(backend="sam3"|"grounded_sam"|"heuristic", ...)
  mask = seg.mask(pil_image, concept="tank top")   # -> bool ndarray (H,W) or None
  pixels = seg.garment_pixels(pil_image, concept)   # -> (M,3) uint8 RGB inside mask
  region = seg.region_pixels(pil_image, mask, "body"|"upper"|"lower")  # coarse parts

The returned mask convention: True = garment pixel to keep. None means
"no reliable mask" -> caller should treat every (non-background) pixel as valid
(fail-open). garment_pixels() already applies that fallback internally.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


# ════════════════════════════════════════════════════════════════════════
#  Foreground heuristic (fallback / default; CPU, no model)
# ════════════════════════════════════════════════════════════════════════
def foreground_mask(pil_img, white_thresh=0.95, black_thresh=0.05,
                    sat_floor=0.0):
    """Cheap product-shot foreground: drop near-white and near-black pixels.
    Returns bool (H,W). This is the same notion the old grid used (drop_white),
    generalized to also drop near-black studio backdrops. Not a real garment
    segmenter — it's the fail-open floor."""
    arr = np.asarray(pil_img.convert("RGB"), dtype="float32") / 255.0
    mx = arr.max(-1)
    mn = arr.min(-1)
    near_white = mn > white_thresh
    near_black = mx < black_thresh
    keep = ~(near_white | near_black)
    if sat_floor > 0:
        sat = np.where(mx > 0, (mx - mn) / np.clip(mx, 1e-6, None), 0.0)
        keep &= (sat >= sat_floor) | (mx < 0.5)   # keep dark garments too
    return keep


# ════════════════════════════════════════════════════════════════════════
#  GarmentSegmenter
# ════════════════════════════════════════════════════════════════════════
class GarmentSegmenter:
    """Unified garment mask provider with pluggable backends.

    backend:
      "heuristic"     : no model; foreground_mask only (default, always works).
      "sam3"          : Meta SAM 3 Promptable Concept Segmentation by text noun.
      "grounded_sam"  : Grounding DINO (text->box) + SAM (box->mask).
    Any backend that fails to load silently degrades to "heuristic" (fail-open),
    recording the reason in self.load_error.
    """

    def __init__(self, backend="heuristic", device="cuda",
                 min_mask_frac=0.03, sam3_checkpoint=None,
                 gdino_config=None, gdino_checkpoint=None, sam_checkpoint=None):
        self.backend = backend
        self.device = device
        self.min_mask_frac = min_mask_frac
        self.load_error = None
        self._model = None
        self._predictor = None
        if backend in ("sam3", "grounded_sam"):
            try:
                if backend == "sam3":
                    self._load_sam3(sam3_checkpoint)
                else:
                    self._load_grounded_sam(gdino_config, gdino_checkpoint,
                                            sam_checkpoint)
            except Exception as e:           # fail-open to heuristic
                self.load_error = f"{type(e).__name__}: {e}"
                self.backend = "heuristic"

    # ---- backend loaders (lazy, optional deps) --------------------------
    def _load_sam3(self, checkpoint):
        # SAM 3 (github.com/facebookresearch/sam3). API per the released repo:
        # a concept (text) -> instance masks. Imported lazily.
        import torch  # noqa
        from sam3.model_zoo import build_sam3_concept_model  # type: ignore
        self._model = build_sam3_concept_model(checkpoint).to(self.device).eval()

    def _load_grounded_sam(self, gdino_config, gdino_ckpt, sam_ckpt):
        import torch  # noqa
        from groundingdino.util.inference import load_model as _load_gdino  # type: ignore
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
        self._gdino = _load_gdino(gdino_config, gdino_ckpt)
        sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(self.device)
        self._predictor = SamPredictor(sam)

    # ---- mask -----------------------------------------------------------
    def mask(self, pil_img, concept="garment"):
        """Return bool (H,W) garment mask, or None if no reliable mask.
        None signals the caller to fail open (use all non-background pixels)."""
        if self.backend == "heuristic":
            return None
        try:
            if self.backend == "sam3":
                m = self._mask_sam3(pil_img, concept)
            else:
                m = self._mask_grounded_sam(pil_img, concept)
        except Exception:
            return None                      # runtime failure -> fail open
        if m is None:
            return None
        if m.mean() < self.min_mask_frac:    # too small -> unreliable
            return None
        return m

    def _mask_sam3(self, pil_img, concept):
        import torch
        with torch.no_grad():
            out = self._model.segment(np.asarray(pil_img.convert("RGB")),
                                      concept=concept)
        masks = out.get("masks")
        if masks is None or len(masks) == 0:
            return None
        # union of all instances of the concept (e.g. all "tank top" regions)
        return np.any(np.asarray(masks).astype(bool), axis=0)

    def _mask_grounded_sam(self, pil_img, concept):
        import torch
        from groundingdino.util.inference import predict as _gdino_predict  # type: ignore
        rgb = np.asarray(pil_img.convert("RGB"))
        boxes, _, _ = _gdino_predict(model=self._gdino, image=rgb,
                                     caption=concept, box_threshold=0.3,
                                     text_threshold=0.25)
        if boxes is None or len(boxes) == 0:
            return None
        self._predictor.set_image(rgb)
        H, W = rgb.shape[:2]
        union = np.zeros((H, W), dtype=bool)
        for b in boxes:
            cx, cy, bw, bh = b.tolist()
            box = np.array([(cx - bw / 2) * W, (cy - bh / 2) * H,
                            (cx + bw / 2) * W, (cy + bh / 2) * H])
            masks, scores, _ = self._predictor.predict(box=box, multimask_output=False)
            union |= masks[0].astype(bool)
        return union

    # ---- pixel extraction (with fail-open) ------------------------------
    def garment_pixels(self, pil_img, concept="garment", max_pixels=20000):
        """Return (M,3) uint8 RGB of garment pixels. Uses the segmentation mask
        when available; otherwise falls back to the foreground heuristic. The
        result is what the color/pattern experts score, so background/skin no
        longer leaks in when a real mask exists."""
        rgb = np.asarray(pil_img.convert("RGB"))
        m = self.mask(pil_img, concept)
        if m is None:
            m = foreground_mask(pil_img)
        px = rgb[m]
        if len(px) == 0:                     # mask emptied everything -> use all
            px = rgb.reshape(-1, 3)
        if len(px) > max_pixels:             # subsample for speed (deterministic)
            idx = np.linspace(0, len(px) - 1, max_pixels).astype(int)
            px = px[idx]
        return px

    def region_pixels(self, pil_img, mask=None, region="body"):
        """Coarse vertical region split of the garment mask for part-aware
        scoring (SCHP-lite): 'upper' = top 45%, 'lower' = bottom 45%, 'body' =
        full. Real collar/sleeve parsing is left to an SCHP backend; this is the
        dependency-free approximation the study lists as the SCHP stand-in."""
        rgb = np.asarray(pil_img.convert("RGB"))
        H, W = rgb.shape[:2]
        if mask is None:
            mask = foreground_mask(pil_img)
        band = np.zeros((H, W), dtype=bool)
        if region == "upper":
            band[: int(0.45 * H), :] = True
        elif region == "lower":
            band[int(0.55 * H):, :] = True
        else:
            band[:, :] = True
        sel = mask & band
        px = rgb[sel]
        return px if len(px) else rgb[mask] if mask.any() else rgb.reshape(-1, 3)
