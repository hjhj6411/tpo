#!/usr/bin/env python3

"""
IMG="/home1/hjhj6411/pod_bench/data/retrieval/qwenemb_product_patch_gallery_smoke/images/q00003__U001__cold_polar_expedition__color/D_neither__plaid_white_tank_top_studio_product_shot_isolated_clothing_item_/rank_001__000626209.jpg"

python retrieval/visualize_groundingdino_sam2_precise.py \
  --image "$IMG" \
  --text "skirt" \
  --output-root data/retrieval/groundingdino_sam2_precise_vis \
  --device cuda \
  --fp16 \
  --box-threshold 0.25 \
  --text-threshold 0.25 \
  --short-side-tiles 8 \
  --min-coverage 0.05
  
  """
from __future__ import annotations

import argparse
import json
import math
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def norm(x: Any) -> str:
    return str(x or "").replace("_", " ").strip().lower()


def font(size: int):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load_groundingdino(model_id: str, device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


def load_sam2(model_id: str, device: str):
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    if hasattr(predictor, "model"):
        predictor.model.to(device)
        predictor.model.eval()
    return predictor


def make_caption(garment: str, color: str = "", pattern: str = "", extra: str = "") -> tuple[str, list[str]]:
    g = norm(garment) or "clothing item"
    c = norm(color)
    p = norm(pattern)
    e = norm(extra)

    phrases = [
        f"entire {g}",
        f"full {g}",
        f"main {g}",
        f"{g} clothing item",
        f"{g} garment",
    ]

    if c:
        phrases.extend([
            f"{c} {g}",
            f"entire {c} {g}",
            f"main {c} {g}",
        ])

    if p and p != "solid":
        phrases.extend([
            f"{p} {g}",
            f"{p} clothing item",
            f"entire {p} {g}",
        ])

    phrases.extend([
        "main clothing item",
        "entire clothing garment",
    ])

    if e:
        phrases.insert(0, e)

    seen = set()
    clean = []
    for x in phrases:
        x = re.sub(r"\s+", " ", x.strip().lower())
        if x and x not in seen:
            clean.append(x)
            seen.add(x)

    # GroundingDINO는 phrase들을 period로 나누는 방식이 안정적입니다.
    return " . ".join(clean) + " .", clean


def expand_box(box: np.ndarray, w: int, h: int, ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    return np.array([
        max(0.0, x1 - bw * ratio),
        max(0.0, y1 - bh * ratio),
        min(float(w - 1), x2 + bw * ratio),
        min(float(h - 1), y2 + bh * ratio),
    ], dtype=np.float32)


def post_process_groundingdino(
    processor,
    outputs,
    input_ids,
    target_sizes,
    box_threshold: float,
    text_threshold: float,
):
    fn = processor.post_process_grounded_object_detection

    attempts = [
        lambda: fn(
            outputs,
            input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        ),
        lambda: fn(
            outputs,
            input_ids=input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        ),
        lambda: fn(
            outputs,
            input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        ),
        lambda: fn(
            outputs,
            input_ids=input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        ),
        lambda: fn(
            outputs,
            target_sizes=target_sizes,
            threshold=box_threshold,
            text_threshold=text_threshold,
        ),
    ]

    last_err = None
    for call in attempts:
        try:
            result = call()
            if isinstance(result, list):
                return result[0]
            return result
        except TypeError as e:
            last_err = e

    raise last_err

def detect_boxes(
    processor,
    model,
    image: Image.Image,
    caption: str,
    device: str,
    box_threshold: float,
    text_threshold: float,
    fp16: bool,
    box_expand: float,
    min_area_ratio: float,
    max_area_ratio: float,
    area_score_power: float,
):
    inputs = processor(images=image, text=caption, return_tensors="pt")
    inputs = inputs.to(device)

    ctx = (
        torch.autocast("cuda", dtype=torch.float16)
        if fp16 and str(device).startswith("cuda")
        else nullcontext()
    )

    with torch.no_grad(), ctx:
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)

    result = post_process_groundingdino(
        processor=processor,
        outputs=outputs,
        input_ids=inputs.input_ids,
        target_sizes=target_sizes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    boxes = result.get("boxes")
    scores = result.get("scores")
    labels = result.get("labels", [])

    if boxes is None or len(boxes) == 0:
        return []

    boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
    scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)

    w, h = image.size
    candidates = []

    for i, box in enumerate(boxes_np):
        x1, y1, x2, y2 = [float(v) for v in box]
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / max(1.0, w * h))

        if area_ratio < min_area_ratio:
            continue
        if area_ratio > max_area_ratio:
            continue

        score = float(scores_np[i]) if i < len(scores_np) else 0.0
        final_score = score * (area_ratio ** area_score_power)
        expanded = expand_box(box, w=w, h=h, ratio=box_expand)

        candidates.append({
            "idx": int(i),
            "box": expanded,
            "raw_box": np.asarray(box, dtype=np.float32),
            "score": score,
            "area_ratio": area_ratio,
            "final_score": final_score,
            "label": str(labels[i]) if i < len(labels) else "",
        })

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates


def sam2_predict_masks(predictor, image: Image.Image, box: np.ndarray):
    arr = np.asarray(image.convert("RGB"))

    with torch.inference_mode():
        predictor.set_image(arr)
        masks, scores, _ = predictor.predict(
            box=box.astype(np.float32),
            multimask_output=True,
        )

    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    else:
        masks = np.asarray(masks)

    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    else:
        scores = np.asarray(scores) if scores is not None else np.asarray([])

    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, :, :]

    if len(masks) == 0:
        return []

    out = []
    for i, m in enumerate(masks):
        score = float(scores[i]) if i < len(scores) else 0.0
        mask = m.astype(bool)
        area_ratio = float(mask.sum()) / float(mask.size)
        out.append({
            "idx": int(i),
            "mask": mask,
            "score": score,
            "area_ratio": area_ratio,
        })

    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def ceil_to_multiple(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


def patch_grid(w: int, h: int, short_side_tiles: int):
    s = max(1, int(math.ceil(min(w, h) / max(1, short_side_tiles))))
    pw = ceil_to_multiple(w, s)
    ph = ceil_to_multiple(h, s)
    return [(x, y, s) for y in range(0, ph, s) for x in range(0, pw, s)]


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--garment", required=True)
    ap.add_argument("--color", default="")
    ap.add_argument("--pattern", default="")
    ap.add_argument("--extra-caption", default="")
    ap.add_argument("--output-root", type=Path, default=Path("data/retrieval/groundingdino_sam2_patch_vis"))

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--sam2-model", default="facebook/sam2-hiera-tiny")

    ap.add_argument("--box-threshold", type=float, default=0.20)
    ap.add_argument("--text-threshold", type=float, default=0.20)
    ap.add_argument("--box-expand", type=float, default=0.03)
    ap.add_argument("--min-box-area-ratio", type=float, default=0.003)
    ap.add_argument("--max-box-area-ratio", type=float, default=0.98)
    ap.add_argument("--area-score-power", type=float, default=0.20)

    ap.add_argument("--short-side-tiles", type=int, default=8)
    ap.add_argument("--min-coverage", type=float, default=0.05)
    ap.add_argument("--alpha", type=int, default=95)
    args = ap.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(args.image)

    image = Image.open(args.image).convert("RGB")
    w, h = image.size

    caption, phrases = make_caption(
        garment=args.garment,
        color=args.color,
        pattern=args.pattern,
        extra=args.extra_caption,
    )

    out_dir = args.output_root / args.image.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print("image:", args.image)
    print("size:", image.size)
    print("caption:", caption)
    print("phrases:", phrases)

    print("loading GroundingDINO...")
    processor, gdino = load_groundingdino(args.grounding_model, args.device)

    print("detecting boxes...")
    candidates = detect_boxes(
        processor=processor,
        model=gdino,
        image=image,
        caption=caption,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        fp16=args.fp16,
        box_expand=args.box_expand,
        min_area_ratio=args.min_box_area_ratio,
        max_area_ratio=args.max_box_area_ratio,
        area_score_power=args.area_score_power,
    )

    boxes_json = []
    for c in candidates:
        boxes_json.append({
            "idx": c["idx"],
            "score": c["score"],
            "area_ratio": c["area_ratio"],
            "final_score": c["final_score"],
            "label": c["label"],
            "box_xyxy": [float(v) for v in c["box"]],
            "raw_box_xyxy": [float(v) for v in c["raw_box"]],
        })

    (out_dir / "boxes.json").write_text(
        json.dumps({
            "image": str(args.image),
            "caption": caption,
            "phrases": phrases,
            "boxes": boxes_json,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not candidates:
        print("NO GroundingDINO box")
        image.save(out_dir / "original.jpg", quality=95)
        print("out:", out_dir)
        return

    for c in candidates[:10]:
        print(
            f"box idx={c['idx']} score={c['score']:.4f} "
            f"area={c['area_ratio']:.4f} final={c['final_score']:.4f} "
            f"label={c['label']} xyxy={[round(float(v), 1) for v in c['box']]}"
        )

    best_box = candidates[0]["box"]

    print("loading SAM2...")
    sam2 = load_sam2(args.sam2_model, args.device)

    print("predicting SAM2 masks...")
    sam_masks = sam2_predict_masks(sam2, image, best_box)

    if not sam_masks:
        print("NO SAM2 mask")
        return

    best_mask = sam_masks[0]["mask"]
    print("selected SAM2 mask score:", sam_masks[0]["score"])
    print("selected SAM2 mask area:", sam_masks[0]["area_ratio"])

    save_mask(best_mask, out_dir / "mask.png")

    # Original
    image.save(out_dir / "original.jpg", quality=95)

    # Overlay
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    pix = layer.load()

    for yy in range(h):
        for xx in range(w):
            if best_mask[yy, xx]:
                pix[xx, yy] = (255, 0, 0, args.alpha)

    overlay = Image.alpha_composite(base, layer)
    draw = ImageDraw.Draw(overlay)

    # All candidate boxes: blue
    for c in candidates[:5]:
        x1, y1, x2, y2 = [int(round(v)) for v in c["box"]]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 120, 255, 255), width=2)
        draw.text(
            (x1 + 3, max(0, y1 - 18)),
            f"{c['score']:.2f}",
            fill=(0, 120, 255, 255),
            font=font(14),
        )

    # Selected box: green
    x1, y1, x2, y2 = [int(round(v)) for v in best_box]
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=4)
    draw.text(
        (x1 + 4, max(0, y1 - 26)),
        "GroundingDINO box + SAM2 mask",
        fill=(0, 255, 0, 255),
        font=font(18),
    )

    # Patch overlay: yellow
    kept = 0
    total = 0

    for i, (x, y, s) in enumerate(patch_grid(w, h, args.short_side_tiles)):
        x2p = min(x + s, w)
        y2p = min(y + s, h)
        if x2p <= x or y2p <= y:
            continue

        total += 1
        cov = float(best_mask[y:y2p, x:x2p].sum()) / float(s * s)

        if cov >= args.min_coverage:
            kept += 1
            draw.rectangle([x, y, x + s, y + s], outline=(255, 255, 0, 255), width=2)
            draw.text(
                (x + 3, y + 3),
                f"{cov:.2f}",
                fill=(255, 255, 0, 255),
                font=font(12),
            )

    overlay.convert("RGB").save(out_dir / "overlay.jpg", quality=95)

    # mask-only patch overlay
    mask_rgb = Image.new("RGB", (w, h), "white")
    mask_draw = ImageDraw.Draw(mask_rgb)
    mask_img = Image.fromarray((best_mask.astype(np.uint8) * 255), mode="L").convert("RGB")
    mask_rgb.paste(mask_img)

    mask_grid = mask_rgb.convert("RGBA")
    mask_draw = ImageDraw.Draw(mask_grid)
    for i, (x, y, s) in enumerate(patch_grid(w, h, args.short_side_tiles)):
        x2p = min(x + s, w)
        y2p = min(y + s, h)
        if x2p <= x or y2p <= y:
            continue
        cov = float(best_mask[y:y2p, x:x2p].sum()) / float(s * s)
        if cov >= args.min_coverage:
            mask_draw.rectangle([x, y, x + s, y + s], outline=(255, 255, 0, 255), width=2)
            mask_draw.text((x + 3, y + 3), f"{cov:.2f}", fill=(255, 255, 0, 255), font=font(12))

    mask_grid.convert("RGB").save(out_dir / "mask_patches.jpg", quality=95)

    summary = {
        "image": str(args.image),
        "size": [w, h],
        "caption": caption,
        "phrases": phrases,
        "selected_box": [float(v) for v in best_box],
        "selected_sam2_score": sam_masks[0]["score"],
        "selected_mask_area_ratio": sam_masks[0]["area_ratio"],
        "total_patches": total,
        "kept_patches": kept,
        "min_coverage": args.min_coverage,
        "outputs": {
            "original": str(out_dir / "original.jpg"),
            "overlay": str(out_dir / "overlay.jpg"),
            "mask": str(out_dir / "mask.png"),
            "mask_patches": str(out_dir / "mask_patches.jpg"),
            "boxes": str(out_dir / "boxes.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("patches:", total)
    print("kept_patches:", kept)
    print("out:", out_dir)
    print("overlay:", out_dir / "overlay.jpg")
    print("mask:", out_dir / "mask.png")
    print("mask_patches:", out_dir / "mask_patches.jpg")
    print("boxes:", out_dir / "boxes.json")


if __name__ == "__main__":
    main()
