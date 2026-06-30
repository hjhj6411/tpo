#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


OPTION_LABELS = ["A", "B", "C", "D"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def fnum(x: Any) -> float:
    try:
        return float(x or 0.0)
    except Exception:
        return 0.0


def font(size: int):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def thumb_image(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def make_board(
    rows: list[dict[str, Any]],
    out_path: Path,
    title: str,
    show_k: int,
    cols: int,
    thumb: int,
) -> None:
    rows = sorted(
        rows,
        key=lambda r: (fnum(r.get("combined_score")), fnum(r.get("similarity"))),
        reverse=True,
    )[:show_k]

    title_font = font(24)
    rank_font = font(18)
    text_font = font(13)

    card_w = thumb + 24
    card_h = thumb + 82
    header_h = 70
    gap = 12

    n = len(rows)
    grid_rows = max(1, math.ceil(n / cols))

    W = cols * card_w + (cols + 1) * gap
    H = header_h + grid_rows * card_h + (grid_rows + 1) * gap

    board = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(board)

    draw.text((gap, 16), title, fill=(0, 0, 0), font=title_font)

    for idx, r in enumerate(rows, start=1):
        grid_i = idx - 1
        row = grid_i // cols
        col = grid_i % cols

        x0 = gap + col * (card_w + gap)
        y0 = header_h + gap + row * (card_h + gap)

        draw.rounded_rectangle(
            [x0, y0, x0 + card_w, y0 + card_h],
            radius=10,
            fill=(255, 255, 255),
            outline=(210, 210, 210),
            width=1,
        )

        local = r.get("local_path")
        img_x = x0 + 12
        img_y = y0 + 10

        if local and Path(local).exists():
            try:
                im = thumb_image(Path(local), thumb)
                board.paste(im, (img_x, img_y))
            except Exception:
                draw.rectangle([img_x, img_y, img_x + thumb, img_y + thumb], fill=(230, 230, 230))
                draw.text((img_x + 8, img_y + 8), "image error", fill=(160, 0, 0), font=text_font)
        else:
            draw.rectangle([img_x, img_y, img_x + thumb, img_y + thumb], fill=(230, 230, 230))
            draw.text((img_x + 8, img_y + 8), "no image", fill=(160, 0, 0), font=text_font)

        # rank badge
        badge = f"{idx:03d}"
        draw.rounded_rectangle(
            [img_x + 6, img_y + 6, img_x + 58, img_y + 34],
            radius=6,
            fill=(0, 0, 0),
        )
        draw.text((img_x + 12, img_y + 8), badge, fill=(255, 255, 255), font=rank_font)

        comb = fnum(r.get("combined_score"))
        pat = fnum(r.get("pattern_score"))
        col_s = fnum(r.get("color_score"))
        gar = fnum(r.get("garment_score"))
        orig = int(r.get("original_rank") or 0)

        tx = x0 + 12
        ty = img_y + thumb + 8
        draw.text((tx, ty), f"orig {orig:03d} | comb {comb:.3f}", fill=(0, 0, 0), font=text_font)
        draw.text((tx, ty + 18), f"pat {pat:.3f} | col {col_s:.3f} | gar {gar:.3f}", fill=(0, 0, 0), font=text_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(out_path, quality=95)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--show-k", type=int, default=100)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--thumb", type=int, default=180)
    ap.add_argument("--query-id", default="")
    ap.add_argument("--option", default="")
    args = ap.parse_args()

    records = load_jsonl(args.scored)
    if args.query_id:
        records = [r for r in records if str(r.get("query_id")) == args.query_id]
    if args.option:
        records = [r for r in records if str(r.get("option_label")).upper() == args.option.upper()]

    root = args.scored.parent
    out_dir = args.out_dir or (root / "rank_boards")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((str(r.get("query_id")), str(r.get("option_label"))), []).append(r)

    made = []
    for (query_id, label), rows in sorted(groups.items()):
        if not rows:
            continue
        head = rows[0]
        option_text = str(head.get("option_text") or "")
        target = f"c={head.get('target_color')} p={head.get('target_pattern')} g={head.get('target_garment')}"
        title = f"{query_id} | {label} | {option_text} | {target}"

        safe_q = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in query_id)[:80]
        safe_o = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
        out = out_dir / f"{safe_q}__{safe_o}_rank_board.jpg"

        make_board(
            rows=rows,
            out_path=out,
            title=title,
            show_k=args.show_k,
            cols=args.cols,
            thumb=args.thumb,
        )
        made.append(out)

    print("rank boards:")
    for p in made:
        print(p)


if __name__ == "__main__":
    main()
