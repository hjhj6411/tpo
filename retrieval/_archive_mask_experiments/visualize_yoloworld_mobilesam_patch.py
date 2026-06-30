#!/usr/bin/env python3
"""
IMG="/home1/hjhj6411/pod_bench/data/retrieval/qwenemb_product_patch_gallery_smoke/images/q00002__U001__cold_blizzard_outdoor__pattern/C_preference_only__striped_black_skirt_studio_product_shot_isolated_clothing_item_p/rank_009__000312257.jpg"

python retrieval/visualize_yoloworld_mobilesam_patch.py \
  --image "$IMG" \
  --query "skirt" \
  --output-root data/retrieval/yoloworld_mobilesam_patch_vis \
  --device cuda \
  --fp16 \
  --yolo-model yolov8s-world.pt \
  --yolo-conf 0.03 \
  --yolo-imgsz 640 \
  --short-side-tiles 8 \
  --min-coverage 0.05
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def load_yolo_world(model_name: str):
    from ultralytics import YOLOWorld
    return YOLOWorld(model_name)


def load_mobilesam(checkpoint: Path, device: str):
    from mobile_sam import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


def expand_box(box: np.ndarray, w: int, h: int, ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = [float(x) for x in box]
    bw = x2 - x1
    bh = y2 - y1
    return np.array([
        max(0, x1 - bw * ratio),
        max(0, y1 - bh * ratio),
        min(w - 1, x2 + bw * ratio),
        min(h - 1, y2 + bh * ratio),
    ], dtype=np.float32)


def detect_box(yolo, image_path: Path, query: str, args):
    classes = [x.strip() for x in query.split(",") if x.strip()]
    if hasattr(yolo, "set_classes"):
        yolo.set_classes(classes)

    device_arg = 0 if str(args.device).startswith("cuda") else "cpu"

    results = yolo.predict(
        source=str(image_path),
        conf=args.yolo_conf,
        iou=args.yolo_iou,
        imgsz=args.yolo_imgsz,
        device=device_arg,
        half=bool(args.fp16 and str(args.device).startswith("cuda")),
        verbose=False,
    )

    if not results:
        return None, []

    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None, []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    candidates = []
    for i, b in enumerate(xyxy):
        x1, y1, x2, y2 = b
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / max(1.0, w * h))
        if area_ratio < args.min_box_area_ratio:
            continue
        if area_ratio > args.max_box_area_ratio:
            continue
        score = float(conf[i]) * (area_ratio ** 0.25)
        candidates.append({
            "idx": i,
            "box": expand_box(b, w, h, args.box_expand),
            "conf": float(conf[i]),
            "area_ratio": float(area_ratio),
            "score": float(score),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return (candidates[0]["box"] if candidates else None), candidates


def predict_sam_mask(predictor, image: Image.Image, box: np.ndarray):
    arr = np.asarray(image.convert("RGB"))
    predictor.set_image(arr)
    masks, scores, _ = predictor.predict(
        box=box.astype(np.float32),
        multimask_output=False,
    )
    if masks is None or len(masks) == 0:
        return None, None
    return masks[0].astype(bool), float(scores[0]) if scores is not None and len(scores) else None


def ceil_to_multiple(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


def patch_grid(w: int, h: int, short_side_tiles: int):
    s = max(1, int(math.ceil(min(w, h) / max(1, short_side_tiles))))
    pw = ceil_to_multiple(w, s)
    ph = ceil_to_multiple(h, s)
    return [(x, y, s) for y in range(0, ph, s) for x in range(0, pw, s)]


def font(size: int):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--query", default="tank top, sleeveless top, camisole top")
    ap.add_argument("--output-root", type=Path, default=Path("data/retrieval/yoloworld_mobilesam_vis"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--yolo-model", default="yolov8s-world.pt")
    ap.add_argument("--yolo-conf", type=float, default=0.03)
    ap.add_argument("--yolo-iou", type=float, default=0.50)
    ap.add_argument("--yolo-imgsz", type=int, default=640)
    ap.add_argument("--box-expand", type=float, default=0.03)
    ap.add_argument("--min-box-area-ratio", type=float, default=0.005)
    ap.add_argument("--max-box-area-ratio", type=float, default=0.95)

    ap.add_argument("--mobilesam-checkpoint", type=Path, default=Path("weights/mobile_sam.pt"))
    ap.add_argument("--short-side-tiles", type=int, default=8)
    ap.add_argument("--min-coverage", type=float, default=0.05)
    ap.add_argument("--alpha", type=int, default=95)
    args = ap.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(args.image)
    if not args.mobilesam_checkpoint.exists():
        raise FileNotFoundError(args.mobilesam_checkpoint)

    out_dir = args.output_root / args.image.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    w, h = image.size

    print("loading YOLO-World...")
    yolo = load_yolo_world(args.yolo_model)

    print("detecting garment box...")
    box, candidates = detect_box(yolo, args.image, args.query, args)

    if box is None:
        print("NO YOLO BOX")
        image.save(out_dir / "original.jpg")
        return

    print("selected box:", [round(float(x), 2) for x in box])
    print("num candidates:", len(candidates))
    for c in candidates[:10]:
        print(
            f"box idx={c['idx']} conf={c['conf']:.4f} "
            f"area={c['area_ratio']:.4f} score={c['score']:.4f} "
            f"xyxy={[round(float(x), 1) for x in c['box']]}"
        )

    print("loading MobileSAM...")
    predictor = load_mobilesam(args.mobilesam_checkpoint, args.device)

    print("predicting SAM mask...")
    mask, sam_score = predict_sam_mask(predictor, image, box)
    if mask is None:
        print("NO SAM MASK")
        return

    print("sam_score:", sam_score)
    mask_area = float(mask.sum()) / float(mask.size)
    print("mask_area_ratio:", mask_area)

    # save mask
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img.save(out_dir / "mask.png")

    # overlay mask
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    px = layer.load()
    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                px[x, y] = (255, 0, 0, args.alpha)

    overlay = Image.alpha_composite(base, layer)

    draw = ImageDraw.Draw(overlay)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=4)
    draw.text((x1 + 4, max(0, y1 - 24)), "YOLO-World box + MobileSAM mask", fill=(0, 255, 0, 255), font=font(18))

    # patch overlay
    kept = 0
    all_patches = 0
    for i, (x, y, s) in enumerate(patch_grid(w, h, args.short_side_tiles)):
        x2p = min(x + s, w)
        y2p = min(y + s, h)
        if x2p <= x or y2p <= y:
            continue
        all_patches += 1
        cov = float(mask[y:y2p, x:x2p].sum()) / float(s * s)
        if cov >= args.min_coverage:
            kept += 1
            draw.rectangle([x, y, x + s, y + s], outline=(255, 255, 0, 255), width=2)
            draw.text((x + 3, y + 3), f"{cov:.2f}", fill=(255, 255, 0, 255), font=font(12))

    overlay = overlay.convert("RGB")
    overlay.save(out_dir / "overlay.jpg", quality=95)
    image.save(out_dir / "original.jpg", quality=95)

    print("patches:", all_patches)
    print("kept_patches:", kept)
    print("out:", out_dir)
    print("overlay:", out_dir / "overlay.jpg")
    print("mask:", out_dir / "mask.png")


if __name__ == "__main__":
    main()
