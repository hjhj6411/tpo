#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def font(size: int):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def clean_text(text: str) -> str:
    text = " ".join(str(text or "").lower().replace("_", " ").split())
    if not text.endswith("."):
        text += "."
    return text


def load_groundingdino(model_id: str, device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def load_sam2(model_id: str, device: str):
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    if hasattr(predictor, "model"):
        predictor.model.to(device)
        predictor.model.eval()
    return predictor


def post_process(processor, outputs, input_ids, image: Image.Image, threshold: float, text_threshold: float):
    target_sizes = [(image.height, image.width)]
    fn = processor.post_process_grounded_object_detection

    calls = [
        lambda: fn(outputs, input_ids, threshold=threshold, text_threshold=text_threshold, target_sizes=target_sizes),
        lambda: fn(outputs, input_ids=input_ids, threshold=threshold, text_threshold=text_threshold, target_sizes=target_sizes),
        lambda: fn(outputs, threshold=threshold, text_threshold=text_threshold, target_sizes=target_sizes),
        lambda: fn(outputs, input_ids, box_threshold=threshold, text_threshold=text_threshold, target_sizes=target_sizes),
        lambda: fn(outputs, input_ids=input_ids, box_threshold=threshold, text_threshold=text_threshold, target_sizes=target_sizes),
    ]

    last = None
    for c in calls:
        try:
            out = c()
            return out[0] if isinstance(out, list) else out
        except TypeError as e:
            last = e
    raise last


def detect_boxes(processor, model, image: Image.Image, text: str, device: str, args):
    prompt = clean_text(text)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    ctx = (
        torch.autocast("cuda", dtype=torch.float16)
        if args.fp16 and str(device).startswith("cuda")
        else nullcontext()
    )

    with torch.no_grad(), ctx:
        outputs = model(**inputs)

    result = post_process(
        processor=processor,
        outputs=outputs,
        input_ids=inputs.input_ids,
        image=image,
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    boxes = result.get("boxes")
    scores = result.get("scores")
    labels = result.get("text_labels") or result.get("labels") or []

    if boxes is None or len(boxes) == 0:
        return prompt, []

    boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
    scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)

    w, h = image.size
    out = []

    for i, box in enumerate(boxes_np):
        x1, y1, x2, y2 = [float(v) for v in box]
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / max(1.0, w * h))
        score = float(scores_np[i]) if i < len(scores_np) else 0.0
        label = str(labels[i]) if i < len(labels) else text

        if area_ratio < args.min_box_area_ratio:
            continue
        if area_ratio > args.max_box_area_ratio:
            continue

        out.append({
            "idx": int(i),
            "box": np.asarray([x1, y1, x2, y2], dtype=np.float32),
            "score": score,
            "area_ratio": area_ratio,
            "label": label,
        })

    # Correct selection rule:
    # Do NOT boost large boxes. Grounding score is the detector confidence.
    out.sort(key=lambda x: x["score"], reverse=True)
    return prompt, out


def expand_box(box: np.ndarray, image_size: tuple[int, int], ratio: float) -> np.ndarray:
    w, h = image_size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    return np.asarray([
        max(0.0, x1 - bw * ratio),
        max(0.0, y1 - bh * ratio),
        min(float(w - 1), x2 + bw * ratio),
        min(float(h - 1), y2 + bh * ratio),
    ], dtype=np.float32)


def sam2_masks(predictor, image: Image.Image, box: np.ndarray, args):
    arr = np.asarray(image.convert("RGB"))
    box = expand_box(box, image.size, args.box_expand)

    ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda")
        else nullcontext()
    )

    with torch.inference_mode(), ctx:
        predictor.set_image(arr)
        masks, scores, _ = predictor.predict(
            box=box.astype(np.float32),
            multimask_output=True,
        )

    masks_np = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)

    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]
    if masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]

    rows = []
    for i, m in enumerate(masks_np):
        mask = m.astype(bool)
        score = float(scores_np[i]) if i < len(scores_np) else 0.0
        area_ratio = float(mask.sum()) / float(mask.size)

        if area_ratio < args.min_mask_area_ratio:
            continue
        if area_ratio > args.max_mask_area_ratio:
            continue

        rows.append({
            "idx": int(i),
            "mask": mask,
            "score": score,
            "area_ratio": area_ratio,
            "box": box,
        })

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def ceil_to_multiple(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


def patch_grid(w: int, h: int, short_side_tiles: int):
    s = max(1, int(math.ceil(min(w, h) / max(1, short_side_tiles))))
    pw = ceil_to_multiple(w, s)
    ph = ceil_to_multiple(h, s)
    return [(x, y, s) for y in range(0, ph, s) for x in range(0, pw, s)]


def draw_boxes(image: Image.Image, boxes: list[dict[str, Any]], out_path: Path):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    f = font(16)

    for rank, b in enumerate(boxes, start=1):
        x1, y1, x2, y2 = [int(round(v)) for v in b["box"]]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 120, 255), width=3)
        draw.text(
            (x1 + 3, max(0, y1 - 22)),
            f"{rank} score={b['score']:.3f} area={b['area_ratio']:.2f} {b['label']}",
            fill=(0, 120, 255),
            font=f,
        )

    canvas.save(out_path, quality=95)


def overlay_mask_and_patches(
    image: Image.Image,
    box: np.ndarray,
    mask: np.ndarray,
    box_score: float,
    mask_score: float,
    out_path: Path,
    args,
):
    w, h = image.size
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    pix = layer.load()

    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                pix[x, y] = (255, 0, 0, args.alpha)

    overlay = Image.alpha_composite(base, layer)
    draw = ImageDraw.Draw(overlay)
    f = font(15)

    x1, y1, x2, y2 = [int(round(v)) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=4)
    draw.text(
        (x1 + 4, max(0, y1 - 24)),
        f"box={box_score:.3f} sam={mask_score:.3f}",
        fill=(0, 255, 0, 255),
        font=f,
    )

    kept = 0
    total = 0
    for i, (x, y, s) in enumerate(patch_grid(w, h, args.short_side_tiles)):
        x2p = min(x + s, w)
        y2p = min(y + s, h)
        if x2p <= x or y2p <= y:
            continue

        total += 1
        cov = float(mask[y:y2p, x:x2p].sum()) / float(s * s)
        if cov >= args.min_coverage:
            kept += 1
            draw.rectangle([x, y, x + s, y + s], outline=(255, 255, 0, 255), width=2)
            draw.text((x + 3, y + 3), f"{cov:.2f}", fill=(255, 255, 0, 255), font=font(12))

    overlay.convert("RGB").save(out_path, quality=95)
    return kept, total


def save_mask(mask: np.ndarray, path: Path):
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--text", required=True, help='Detection phrase, e.g. "tank top" or "coat". Use one target phrase.')
    ap.add_argument("--output-root", type=Path, default=Path("data/retrieval/groundingdino_sam2_precise_vis"))

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--sam2-model", default="facebook/sam2-hiera-tiny")

    ap.add_argument("--box-threshold", type=float, default=0.25)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--min-box-area-ratio", type=float, default=0.001)
    ap.add_argument("--max-box-area-ratio", type=float, default=0.95)
    ap.add_argument("--box-expand", type=float, default=0.02)

    ap.add_argument("--min-mask-area-ratio", type=float, default=0.001)
    ap.add_argument("--max-mask-area-ratio", type=float, default=0.95)
    ap.add_argument("--top-boxes", type=int, default=5)

    ap.add_argument("--short-side-tiles", type=int, default=8)
    ap.add_argument("--min-coverage", type=float, default=0.05)
    ap.add_argument("--alpha", type=int, default=95)
    args = ap.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(args.image)

    out_dir = args.output_root / args.image.stem / slug_like(args.text)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")

    print("image:", args.image)
    print("text:", clean_text(args.text))
    print("out:", out_dir)

    print("loading GroundingDINO...")
    processor, gdino = load_groundingdino(args.grounding_model, args.device)

    print("detecting boxes...")
    prompt, boxes = detect_boxes(
        processor=processor,
        model=gdino,
        image=image,
        text=args.text,
        device=args.device,
        args=args,
    )

    print("prompt:", prompt)
    print("num boxes:", len(boxes))

    image.save(out_dir / "original.jpg", quality=95)

    boxes_for_json = []
    for i, b in enumerate(boxes, start=1):
        row = {
            "rank": i,
            "score": b["score"],
            "area_ratio": b["area_ratio"],
            "label": b["label"],
            "box_xyxy": [float(v) for v in b["box"]],
        }
        boxes_for_json.append(row)
        print(
            f"box {i}: score={b['score']:.4f} area={b['area_ratio']:.4f} "
            f"label={b['label']} xyxy={[round(float(v), 1) for v in b['box']]}"
        )

    (out_dir / "boxes.json").write_text(
        json.dumps(
            {
                "image": str(args.image),
                "prompt": prompt,
                "boxes": boxes_for_json,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not boxes:
        print("NO BOX. Try lower --box-threshold 0.15 --text-threshold 0.15")
        return

    draw_boxes(image, boxes[: args.top_boxes], out_dir / "boxes.jpg")

    print("loading SAM2...")
    sam2 = load_sam2(args.sam2_model, args.device)

    selected = None

    for box_rank, b in enumerate(boxes[: args.top_boxes], start=1):
        print(f"sam2 for box {box_rank}/{min(args.top_boxes, len(boxes))}...")
        masks = sam2_masks(sam2, image, b["box"], args)

        for mask_rank, m in enumerate(masks, start=1):
            prefix = f"candidate_{box_rank:02d}_mask_{mask_rank:02d}"
            save_mask(m["mask"], out_dir / f"{prefix}_mask.png")
            kept, total = overlay_mask_and_patches(
                image=image,
                box=m["box"],
                mask=m["mask"],
                box_score=float(b["score"]),
                mask_score=float(m["score"]),
                out_path=out_dir / f"{prefix}_overlay.jpg",
                args=args,
            )

            print(
                f"  {prefix}: sam={m['score']:.4f} "
                f"mask_area={m['area_ratio']:.4f} patches={kept}/{total}"
            )

            if selected is None:
                selected = {
                    "box_rank": box_rank,
                    "mask_rank": mask_rank,
                    "box": m["box"],
                    "mask": m["mask"],
                    "box_score": float(b["score"]),
                    "mask_score": float(m["score"]),
                    "mask_area_ratio": float(m["area_ratio"]),
                    "kept_patches": kept,
                    "total_patches": total,
                }

    if selected is not None:
        save_mask(selected["mask"], out_dir / "selected_mask.png")
        overlay_mask_and_patches(
            image=image,
            box=selected["box"],
            mask=selected["mask"],
            box_score=selected["box_score"],
            mask_score=selected["mask_score"],
            out_path=out_dir / "selected_overlay.jpg",
            args=args,
        )

        summary = {
            "image": str(args.image),
            "prompt": prompt,
            "selected": {
                "box_rank": selected["box_rank"],
                "mask_rank": selected["mask_rank"],
                "box_score": selected["box_score"],
                "mask_score": selected["mask_score"],
                "mask_area_ratio": selected["mask_area_ratio"],
                "kept_patches": selected["kept_patches"],
                "total_patches": selected["total_patches"],
                "box_xyxy": [float(v) for v in selected["box"]],
            },
            "outputs": {
                "boxes": str(out_dir / "boxes.jpg"),
                "selected_overlay": str(out_dir / "selected_overlay.jpg"),
                "selected_mask": str(out_dir / "selected_mask.png"),
                "boxes_json": str(out_dir / "boxes.json"),
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print("selected_overlay:", out_dir / "selected_overlay.jpg")
        print("selected_mask:", out_dir / "selected_mask.png")

    print("boxes:", out_dir / "boxes.jpg")
    print("boxes_json:", out_dir / "boxes.json")


def slug_like(text: str) -> str:
    text = str(text or "").lower()
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return text.strip("_")[:80] or "query"


if __name__ == "__main__":
    main()
