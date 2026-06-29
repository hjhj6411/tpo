#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


def positions(length: int, side: int, stride: int) -> list[int]:
    if length <= side:
        return [0]
    xs = list(range(0, max(1, length - side + 1), stride))
    last = length - side
    if xs[-1] != last:
        xs.append(last)
    return xs


def square_grid(w: int, h: int, short_side_tiles: int, overlap: float) -> list[tuple[int, int, int]]:
    side = max(1, int(math.ceil(min(w, h) / max(1, short_side_tiles))))
    stride = max(1, int(round(side * (1.0 - max(0.0, min(overlap, 0.9))))))
    return [(x, y, side) for y in positions(h, side, stride) for x in positions(w, side, stride)]


def load_model(model_id: str, device: str, fp16: bool):
    model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
    model.to(device)
    model.eval()
    if fp16 and device.startswith("cuda"):
        model.half()
    return model


def predict_mask(model, image: Image.Image, device: str, input_size: int, fp16: bool) -> Image.Image:
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    if fp16 and device.startswith("cuda"):
        x = x.half()

    with torch.no_grad():
        pred = model(x)[-1].sigmoid().cpu()[0].squeeze()

    mask = transforms.ToPILImage()(pred)
    return mask.resize(image.size)


def mask_coverage(mask: Image.Image, x: int, y: int, s: int, mask_threshold: int) -> float:
    crop = mask.crop((x, y, x + s, y + s)).convert("L")
    hist = crop.histogram()
    fg = sum(hist[mask_threshold + 1:])
    return fg / max(1, s * s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("data/retrieval/birefnet_patch_mask"))
    p.add_argument("--model-id", default="ZhengPeng7/BiRefNet")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--input-size", type=int, default=1024)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--short-side-tiles", type=int, default=12)
    p.add_argument("--overlap", type=float, default=0.0)
    p.add_argument("--mask-threshold", type=int, default=32)
    p.add_argument("--min-coverage", type=float, default=0.03)
    p.add_argument("--alpha", type=int, default=90)
    args = p.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(args.image)

    image = Image.open(args.image).convert("RGB")
    w, h = image.size

    model = load_model(args.model_id, args.device, args.fp16)
    mask = predict_mask(model, image, args.device, args.input_size, args.fp16)

    out_dir = args.output_root / args.image.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    mask.save(out_dir / "mask.png")

    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

    rows = []
    for i, (x, y, s) in enumerate(square_grid(w, h, args.short_side_tiles, args.overlap)):
        coverage = mask_coverage(mask, x, y, s, args.mask_threshold)
        present = coverage >= args.min_coverage

        rows.append({
            "index": i,
            "x": x,
            "y": y,
            "size": s,
            "mask_coverage": coverage,
            "present": present,
            "mask_threshold": args.mask_threshold,
            "min_coverage": args.min_coverage,
        })

        if present:
            draw.rectangle(
                [x, y, x + s, y + s],
                fill=(255, 0, 0, args.alpha),
                outline=(255, 0, 0, 220),
                width=1,
            )

    overlay = Image.alpha_composite(base, layer).convert("RGB")
    overlay.save(out_dir / "overlay.png", quality=95)

    with (out_dir / "patch_scores.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    kept = sum(r["present"] for r in rows)
    print(f"patches={len(rows)} kept={kept}")
    print(out_dir / "mask.png")
    print(out_dir / "overlay.png")


if __name__ == "__main__":
    main()
