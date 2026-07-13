#!/usr/bin/env python3
"""
Query the running FashionSigLIP KNN server for a few shirt prompts, download
the returned images, and write contact-sheet PNGs plus a small HTML gallery.

Expected server:
  python fsiglip/serve_fsiglip_knn.py --port 1235 --gpu 0

Example:
  python fsiglip/show_shirt_query_topk.py --top-k 12
"""

import argparse
import html
import io
import json
from pathlib import Path

import requests


DEFAULT_QUERIES = ["black fleece", "fleece"]


def query_knn(session, url, query, top_k, index_name, timeout):
    payload = {
        "text": query,
        "modality": "image",
        "num_images": top_k,
        "indice_name": index_name,
    }
    resp = session.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"KNN response must be a list, got {type(data).__name__}")
    return [row for row in data if isinstance(row, dict)]


def write_jsonl(path, results_by_query):
    with path.open("w", encoding="utf-8") as f:
        for query, rows in results_by_query.items():
            for rank, row in enumerate(rows, start=1):
                out = {"query": query, "rank": rank, **row}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")


def safe_name(text):
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def download_images(session, outdir, results_by_query, timeout):
    from PIL import Image

    image_dir = outdir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for query, rows in results_by_query.items():
        qdir = image_dir / safe_name(query)
        qdir.mkdir(parents=True, exist_ok=True)
        for rank, row in enumerate(rows, start=1):
            url = row.get("image_url") or row.get("url") or ""
            if not url:
                row["local_image_path"] = None
                row["download_error"] = "missing_url"
                continue
            try:
                resp = session.get(url, timeout=timeout)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                path = qdir / f"rank_{rank:03d}.jpg"
                img.save(path, quality=92)
                row["local_image_path"] = str(path)
            except Exception as e:
                row["local_image_path"] = None
                row["download_error"] = str(e)


def make_contact_sheet(path, query, rows, thumb_size=(220, 260), cols=4):
    from PIL import Image, ImageDraw, ImageFont

    loaded = []
    for rank, row in enumerate(rows, start=1):
        local = row.get("local_image_path")
        if not local:
            continue
        try:
            img = Image.open(local).convert("RGB")
        except Exception:
            continue
        loaded.append((rank, row, img))
    if not loaded:
        return False

    font = ImageFont.load_default()
    label_h = 48
    title_h = 34
    rows_n = (len(loaded) + cols - 1) // cols
    w = cols * thumb_size[0]
    h = title_h + rows_n * (thumb_size[1] + label_h)
    sheet = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), query, fill="black", font=font)

    for idx, (rank, row, img) in enumerate(loaded):
        x = (idx % cols) * thumb_size[0]
        y = title_h + (idx // cols) * (thumb_size[1] + label_h)
        img.thumbnail((thumb_size[0] - 12, thumb_size[1] - 12))
        ix = x + (thumb_size[0] - img.width) // 2
        iy = y + (thumb_size[1] - img.height) // 2
        sheet.paste(img, (ix, iy))
        sim = row.get("similarity")
        sim_text = f"{sim:.4f}" if isinstance(sim, (int, float)) else ""
        caption = str(row.get("caption") or "")[:70]
        draw.text((x + 6, y + thumb_size[1] + 4), f"#{rank} sim={sim_text}", fill="black", font=font)
        draw.text((x + 6, y + thumb_size[1] + 20), caption, fill="black", font=font)
    sheet.save(path)
    return True


def write_html(path, results_by_query):
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>FashionSigLIP shirt query top-k</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f7;color:#222}",
        "h1{font-size:22px;margin:0 0 20px}",
        "h2{font-size:18px;margin:28px 0 12px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px}",
        ".card{background:white;border:1px solid #ddd;border-radius:6px;padding:10px}",
        ".card img{width:100%;height:220px;object-fit:contain;background:#fafafa}",
        ".meta{font-size:12px;line-height:1.35;margin-top:8px;word-break:break-word}",
        ".rank{font-weight:bold}",
        "a{color:#0645ad}",
        "</style></head><body>",
        "<h1>FashionSigLIP shirt query top-k</h1>",
    ]
    for query, rows in results_by_query.items():
        parts.append(f"<h2>{html.escape(query)}</h2>")
        parts.append("<div class='grid'>")
        for rank, row in enumerate(rows, start=1):
            image_url = row.get("local_image_path") or row.get("image_url") or row.get("url") or ""
            caption = row.get("caption") or ""
            sim = row.get("similarity")
            sim_text = f"{sim:.4f}" if isinstance(sim, (int, float)) else ""
            parts.extend([
                "<div class='card'>",
                f"<a href='{html.escape(image_url)}' target='_blank'>",
                f"<img src='{html.escape(image_url)}' alt='rank {rank}'>",
                "</a>",
                "<div class='meta'>",
                f"<div><span class='rank'>#{rank}</span> similarity={html.escape(sim_text)}</div>",
                f"<div>{html.escape(caption)}</div>",
                f"<div><a href='{html.escape(image_url)}' target='_blank'>image url</a></div>",
                "</div></div>",
            ])
        parts.append("</div>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-url", default="http://127.0.0.1:1235/knn-service")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--index-name", default="pod_fashion")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--outdir", type=Path, default=Path("data/retrieval/shirt_query_topk"))
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    parser.add_argument("--contact-cols", type=int, default=4)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    results_by_query = {}
    with requests.Session() as session:
        for query in args.queries:
            rows = query_knn(session, args.client_url, query, args.top_k,
                             args.index_name, args.timeout)
            results_by_query[query] = rows
            print(f"{query}: {len(rows)} results")
        download_images(session, args.outdir, results_by_query, args.timeout)

    jsonl_path = args.outdir / "results.jsonl"
    html_path = args.outdir / "gallery.html"
    write_jsonl(jsonl_path, results_by_query)
    write_html(html_path, results_by_query)
    for query, rows in results_by_query.items():
        sheet_path = args.outdir / f"{safe_name(query)}_contact_sheet.png"
        if make_contact_sheet(sheet_path, query, rows, cols=args.contact_cols):
            print(f"Wrote {sheet_path}")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
