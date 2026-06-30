#!/usr/bin/env python3
"""
python retrieval/visualize_sam3_mask_patch.py \
  --image "/home1/hjhj6411/pod_bench/data/retrieval/qwenemb_product_patch_gallery_smoke/images/q00003__U001__cold_polar_expedition__color/A_tpo_and_preference__plaid_beige_jacket_studio_product_shot_isolated_clothing_item_pl/rank_014__000065318.jpg" \
  --prompt "jacket" \
  --output-root data/retrieval/sam3_mask_patch_vis \
  --device cuda \
  --short-side-tiles 8 \
  --min-coverage 0.20
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def slug(text: Any, max_len: int = 80) -> str:
    s = str(text or "")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return (s or "unknown")[:max_len]


def font(size: int):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load_sam3(device: str):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(device=device, eval_mode=True)
    return Sam3Processor(model)


def to_numpy(x: Any) -> np.ndarray:
    if x is None:
        return np.asarray([])

    if torch.is_tensor(x):
        x = x.detach()

        # numpy does not support torch.bfloat16 directly.
        # SAM3 may return bfloat16 scores/boxes under autocast.
        if x.dtype in {torch.bfloat16, torch.float16}:
            x = x.float()

        return x.cpu().numpy()

    return np.asarray(x)


def normalize_masks(masks: Any) -> np.ndarray:
    arr = to_numpy(masks)

    if arr.size == 0:
        return np.zeros((0, 1, 1), dtype=bool)

    # Common cases:
    # [N, H, W]
    # [N, 1, H, W]
    # [1, N, H, W]
    if arr.ndim == 4:
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        elif arr.shape[0] == 1:
            arr = arr[0]

    if arr.ndim == 2:
        arr = arr[None, :, :]

    return arr.astype(bool)


def boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    boxes = []

    for mask in masks:
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            boxes.append([0, 0, 1, 1])
        else:
            boxes.append([
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ])

    return np.asarray(boxes, dtype=np.float32)


def normalize_boxes(boxes: Any, masks: np.ndarray) -> np.ndarray:
    if boxes is None:
        return boxes_from_masks(masks)

    arr = to_numpy(boxes)

    if arr.size == 0:
        return boxes_from_masks(masks)

    arr = arr.reshape(-1, 4).astype(np.float32)

    # If boxes are normalized 0~1, convert to pixel coords later outside.
    if len(arr) < len(masks):
        fallback = boxes_from_masks(masks)
        out = np.zeros((len(masks), 4), dtype=np.float32)
        for i in range(len(masks)):
            out[i] = arr[i] if i < len(arr) else fallback[i]
        return out

    return arr[: len(masks)]


def normalize_scores(scores: Any, n: int) -> np.ndarray:
    if scores is None:
        return np.ones((n,), dtype=np.float32)

    arr = to_numpy(scores).reshape(-1).astype(np.float32)

    if len(arr) < n:
        out = np.ones((n,), dtype=np.float32)
        out[: len(arr)] = arr
        return out

    return arr[:n]


def maybe_denormalize_box(box: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    w, h = image_size
    b = box.astype(np.float32).copy()

    # SAM3 usually returns pixel-space boxes, but this keeps the script robust.
    if float(np.nanmax(b)) <= 1.5:
        b[[0, 2]] *= w
        b[[1, 3]] *= h

    return b


def expand_box(box: np.ndarray, image_size: tuple[int, int], ratio: float) -> np.ndarray:
    w, h = image_size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    return np.asarray([
        max(0.0, x1 - bw * ratio),
        max(0.0, y1 - bh * ratio),
        min(float(w), x2 + bw * ratio),
        min(float(h), y2 + bh * ratio),
    ], dtype=np.float32)


def box_to_int(box: np.ndarray, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = image_size
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    return x1, y1, x2, y2


def run_sam3(processor, image: Image.Image, prompt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )

    with torch.inference_mode(), amp_ctx:
        state = processor.set_image(image)
        try:
            output = processor.set_text_prompt(state=state, prompt=prompt)
        except TypeError:
            output = processor.set_text_prompt(state, prompt)

    masks = normalize_masks(output.get("masks"))
    boxes = normalize_boxes(output.get("boxes"), masks)
    scores = normalize_scores(output.get("scores"), len(masks))

    boxes = np.asarray(
        [maybe_denormalize_box(b, image.size) for b in boxes],
        dtype=np.float32,
    )

    return masks, boxes, scores


def select_candidates(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    image_size: tuple[int, int],
    min_mask_area_ratio: float,
    max_mask_area_ratio: float,
    min_box_area_ratio: float,
    max_box_area_ratio: float,
) -> list[dict[str, Any]]:
    w, h = image_size
    out = []

    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        mask_area_ratio = float(mask.sum()) / float(mask.size)

        x1, y1, x2, y2 = [float(v) for v in box]
        box_area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / max(1.0, w * h))

        reject = None
        if mask_area_ratio < min_mask_area_ratio:
            reject = "mask_area_too_small"
        elif mask_area_ratio > max_mask_area_ratio:
            reject = "mask_area_too_large"
        elif box_area_ratio < min_box_area_ratio:
            reject = "box_area_too_small"
        elif box_area_ratio > max_box_area_ratio:
            reject = "box_area_too_large"

        print(
            f"raw {i}: score={float(score):.4f} "
            f"mask_area={mask_area_ratio:.6f} "
            f"box_area={box_area_ratio:.6f} "
            f"box={[round(float(v), 1) for v in box]} "
            f"reject={reject}"
        )

        if reject is not None:
            continue

        out.append({
            "index": int(i),
            "mask": mask,
            "box": box,
            "score": float(score),
            "mask_area_ratio": mask_area_ratio,
            "box_area_ratio": box_area_ratio,
        })

    # Do not boost large masks. Trust SAM3 concept score first.
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def overlay_mask_on_image(
    image: Image.Image,
    mask: np.ndarray,
    box: np.ndarray,
    score: float,
    alpha: int,
) -> Image.Image:
    image = image.convert("RGB")
    base = image.convert("RGBA")

    rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    rgba[mask] = np.array([255, 0, 0, alpha], dtype=np.uint8)
    layer = Image.fromarray(rgba, mode="RGBA")

    out = Image.alpha_composite(base, layer)
    draw = ImageDraw.Draw(out)

    x1, y1, x2, y2 = box_to_int(box, image.size)
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=4)
    draw.text(
        (x1 + 4, max(0, y1 - 26)),
        f"SAM3 score={score:.3f}",
        fill=(0, 255, 0, 255),
        font=font(18),
    )

    return out.convert("RGB")


def crop_image_and_mask(
    image: Image.Image,
    mask: np.ndarray,
    box: np.ndarray,
) -> tuple[Image.Image, np.ndarray, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box_to_int(box, image.size)
    crop = image.crop((x1, y1, x2, y2)).convert("RGB")
    crop_mask = mask[y1:y2, x1:x2].copy()
    return crop, crop_mask, (x1, y1, x2, y2)


def ceil_to_multiple(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


def patch_grid(w: int, h: int, short_side_tiles: int) -> list[tuple[int, int, int]]:
    side = max(1, int(math.ceil(min(w, h) / max(1, short_side_tiles))))
    padded_w = ceil_to_multiple(w, side)
    padded_h = ceil_to_multiple(h, side)

    return [
        (x, y, side)
        for y in range(0, padded_h, side)
        for x in range(0, padded_w, side)
    ]


def draw_crop_patch_overlay(
    crop: Image.Image,
    crop_mask: np.ndarray,
    short_side_tiles: int,
    min_coverage: float,
    alpha: int,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    crop = crop.convert("RGB")
    w, h = crop.size

    base = crop.convert("RGBA")

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[crop_mask] = np.array([255, 0, 0, alpha], dtype=np.uint8)
    layer = Image.fromarray(rgba, mode="RGBA")

    out = Image.alpha_composite(base, layer)
    draw = ImageDraw.Draw(out)

    kept = []

    for i, (x, y, s) in enumerate(patch_grid(w, h, short_side_tiles)):
        x2 = min(x + s, w)
        y2 = min(y + s, h)

        if x2 <= x or y2 <= y:
            continue

        cov = float(crop_mask[y:y2, x:x2].sum()) / float(s * s)
        if cov >= min_coverage:
            kept.append({
                "index": i,
                "x": x,
                "y": y,
                "size": s,
                "coverage": cov,
            })

            draw.rectangle([x, y, x + s, y + s], outline=(255, 255, 0, 255), width=2)
            draw.text(
                (x + 3, y + 3),
                f"{cov:.2f}",
                fill=(255, 255, 0, 255),
                font=font(12),
            )

    return out.convert("RGB"), kept


def draw_all_boxes(image: Image.Image, candidates: list[dict[str, Any]], out_path: Path) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for rank, c in enumerate(candidates, start=1):
        x1, y1, x2, y2 = box_to_int(c["box"], image.size)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 120, 255), width=3)
        draw.text(
            (x1 + 3, max(0, y1 - 22)),
            f"{rank} score={c['score']:.3f} mask={c['mask_area_ratio']:.2f}",
            fill=(0, 120, 255),
            font=font(15),
        )

    canvas.save(out_path, quality=95)


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--prompt", required=True, help='SAM3 text concept prompt, e.g. "tank top", "coat", "skirt"')
    ap.add_argument("--output-root", type=Path, default=Path("data/retrieval/sam3_mask_patch_vis"))

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--box-expand", type=float, default=0.02)

    ap.add_argument("--min-mask-area-ratio", type=float, default=0.001)
    ap.add_argument("--max-mask-area-ratio", type=float, default=0.95)
    ap.add_argument("--min-box-area-ratio", type=float, default=0.001)
    ap.add_argument("--max-box-area-ratio", type=float, default=0.95)

    ap.add_argument("--short-side-tiles", type=int, default=8)
    ap.add_argument("--min-coverage", type=float, default=0.20)
    ap.add_argument("--top-candidates", type=int, default=5)
    ap.add_argument("--alpha", type=int, default=95)

    args = ap.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(args.image)

    image = Image.open(args.image).convert("RGB")

    out_dir = args.output_root / args.image.stem / slug(args.prompt)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("image:", args.image)
    print("prompt:", args.prompt)
    print("output:", out_dir)

    print("loading SAM3...")
    processor = load_sam3(args.device)

    print("running SAM3 text prompt...")
    masks, boxes, scores = run_sam3(processor, image, args.prompt)

    candidates = select_candidates(
        masks=masks,
        boxes=boxes,
        scores=scores,
        image_size=image.size,
        min_mask_area_ratio=args.min_mask_area_ratio,
        max_mask_area_ratio=args.max_mask_area_ratio,
        min_box_area_ratio=args.min_box_area_ratio,
        max_box_area_ratio=args.max_box_area_ratio,
    )

    print("raw instances:", len(masks))
    print("valid candidates:", len(candidates))

    image.save(out_dir / "original.jpg", quality=95)

    boxes_json = []
    for rank, c in enumerate(candidates, start=1):
        row = {
            "rank": rank,
            "index": c["index"],
            "score": c["score"],
            "mask_area_ratio": c["mask_area_ratio"],
            "box_area_ratio": c["box_area_ratio"],
            "box_xyxy": [float(v) for v in c["box"]],
        }
        boxes_json.append(row)

        print(
            f"cand {rank}: idx={c['index']} "
            f"score={c['score']:.4f} "
            f"mask_area={c['mask_area_ratio']:.4f} "
            f"box_area={c['box_area_ratio']:.4f} "
            f"box={[round(float(v), 1) for v in c['box']]}"
        )

    (out_dir / "candidates.json").write_text(
        json.dumps(
            {
                "image": str(args.image),
                "prompt": args.prompt,
                "raw_instances": int(len(masks)),
                "valid_candidates": int(len(candidates)),
                "candidates": boxes_json,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not candidates:
        print("NO VALID SAM3 MASK")
        print("try a different prompt, e.g. 'skirt', 'coat', 'tank top', 'sleeveless top'")
        return

    draw_all_boxes(image, candidates[: args.top_candidates], out_dir / "all_candidate_boxes.jpg")

    # Save top candidate overlays.
    for rank, c in enumerate(candidates[: args.top_candidates], start=1):
        box = expand_box(c["box"], image.size, args.box_expand)
        mask = c["mask"]

        overlay = overlay_mask_on_image(
            image=image,
            mask=mask,
            box=box,
            score=c["score"],
            alpha=args.alpha,
        )
        overlay.save(out_dir / f"candidate_{rank:02d}_overlay.jpg", quality=95)
        save_mask(mask, out_dir / f"candidate_{rank:02d}_mask.png")

    # Selected = top score.
    selected = candidates[0]
    selected_box = expand_box(selected["box"], image.size, args.box_expand)
    selected_mask = selected["mask"]

    selected_overlay = overlay_mask_on_image(
        image=image,
        mask=selected_mask,
        box=selected_box,
        score=selected["score"],
        alpha=args.alpha,
    )
    selected_overlay.save(out_dir / "selected_overlay.jpg", quality=95)
    save_mask(selected_mask, out_dir / "selected_mask.png")

    crop, crop_mask, crop_xyxy = crop_image_and_mask(
        image=image,
        mask=selected_mask,
        box=selected_box,
    )
    crop.save(out_dir / "selected_crop.jpg", quality=95)
    save_mask(crop_mask, out_dir / "selected_crop_mask.png")

    crop_patch_overlay, kept_patches = draw_crop_patch_overlay(
        crop=crop,
        crop_mask=crop_mask,
        short_side_tiles=args.short_side_tiles,
        min_coverage=args.min_coverage,
        alpha=args.alpha,
    )
    crop_patch_overlay.save(out_dir / "selected_crop_patch_overlay.jpg", quality=95)

    (out_dir / "patches.json").write_text(
        json.dumps(
            {
                "crop_xyxy": list(crop_xyxy),
                "short_side_tiles": args.short_side_tiles,
                "min_coverage": args.min_coverage,
                "num_kept_patches": len(kept_patches),
                "kept_patches": kept_patches,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "image": str(args.image),
        "prompt": args.prompt,
        "selected": {
            "index": selected["index"],
            "score": selected["score"],
            "mask_area_ratio": selected["mask_area_ratio"],
            "box_area_ratio": selected["box_area_ratio"],
            "box_xyxy": [float(v) for v in selected_box],
            "crop_xyxy": list(crop_xyxy),
        },
        "patches": {
            "short_side_tiles": args.short_side_tiles,
            "min_coverage": args.min_coverage,
            "num_kept_patches": len(kept_patches),
        },
        "outputs": {
            "original": str(out_dir / "original.jpg"),
            "all_candidate_boxes": str(out_dir / "all_candidate_boxes.jpg"),
            "selected_overlay": str(out_dir / "selected_overlay.jpg"),
            "selected_mask": str(out_dir / "selected_mask.png"),
            "selected_crop": str(out_dir / "selected_crop.jpg"),
            "selected_crop_mask": str(out_dir / "selected_crop_mask.png"),
            "selected_crop_patch_overlay": str(out_dir / "selected_crop_patch_overlay.jpg"),
            "candidates": str(out_dir / "candidates.json"),
            "patches": str(out_dir / "patches.json"),
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("selected_overlay:", out_dir / "selected_overlay.jpg")
    print("selected_crop:", out_dir / "selected_crop.jpg")
    print("selected_crop_patch_overlay:", out_dir / "selected_crop_patch_overlay.jpg")
    print("selected_mask:", out_dir / "selected_mask.png")
    print("patches:", len(kept_patches))
    print("summary:", out_dir / "summary.json")


if __name__ == "__main__":
    main()
