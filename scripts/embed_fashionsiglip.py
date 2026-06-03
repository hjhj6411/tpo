#!/usr/bin/env python3
"""
FashionSigLIP embedding extractor.
Output format is compatible with clip-retrieval index.
"""
import open_clip, torch, numpy as np, json, time
from pathlib import Path
import pandas as pd
import webdataset as wds
from tqdm import tqdm

# ── 설정 ─────────────────────────────────────────────────────
TAR_PATTERN    = "./data/clip_corpus/images/{00000..00065}.tar"
OUTPUT_DIR     = Path("./data/fashionsiglip_emb")
BATCH_SIZE     = 512
PARTITION_SIZE = 100_000
DEVICE         = "cuda:0"
TOTAL_IMAGES   = 651_489   # 진행바 기준
NUM_WORKERS    = 4         # ← 병목 핵심: 데이터 로딩 병렬화
PREFETCH       = 4         # ← 배치 prefetch 수
# ─────────────────────────────────────────────────────────────

(OUTPUT_DIR / "img_emb").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "metadata").mkdir(parents=True, exist_ok=True)

print("Loading Marqo/marqo-fashionSigLIP ...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:Marqo/marqo-fashionSigLIP'
)
model.eval().to(DEVICE)

def flush_batch(imgs):
    t0 = time.perf_counter()
    with torch.no_grad():
        t = torch.stack(imgs).to(DEVICE)
        embs = model.encode_image(t, normalize=True).cpu().float().numpy()
    return embs, time.perf_counter() - t0

def save_partition(emb_chunks, meta_rows, idx):
    arr = np.concatenate(emb_chunks, axis=0)
    np.save(OUTPUT_DIR / "img_emb" / f"img_emb_{idx}.npy", arr)
    pd.DataFrame(meta_rows).to_parquet(
        OUTPUT_DIR / "metadata" / f"metadata_{idx}.parquet",
        index=False
    )
    tqdm.write(f"  [partition {idx}] saved  shape={arr.shape}")
    return [], []
    
def process_sample(sample):
    img, meta, key = sample
    return preprocess(img.convert("RGB")), meta, key

dataset = (
    wds.WebDataset(TAR_PATTERN, shardshuffle=False)
    .decode("pil")
    .to_tuple("jpg", "json", "__key__")
    .map(process_sample)  # ← 튜플 통째로 처리
    .batched(BATCH_SIZE, partial=True)
)
loader = wds.WebLoader(dataset, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH)

part_embs, part_meta = [], []
partition_idx = 0
total = 0

# 시간 추적
t_load_total = 0.0
t_infer_total = 0.0
t_start = time.time()

pbar = tqdm(loader, total=TOTAL_IMAGES // BATCH_SIZE, unit="batch",
            desc="Embedding", dynamic_ncols=True)

for imgs, metas, keys in pbar:
    t_load = time.perf_counter()

    try:
        batch_imgs = list(imgs)
        batch_meta = [{
            "url":     m.get("url", k),
            "caption": m.get("caption", m.get("title", "")),
        } for m, k in zip(metas, keys)]
    except Exception:
        continue

    t_load_total += time.perf_counter() - t_load

    embs, t_infer = flush_batch(batch_imgs)
    t_infer_total += t_infer

    part_embs.append(embs)
    part_meta.extend(batch_meta)
    total += len(batch_imgs)

    if len(part_meta) >= PARTITION_SIZE:
        part_embs, part_meta = save_partition(part_embs, part_meta, partition_idx)
        partition_idx += 1

    elapsed = time.time() - t_start
    speed = total / elapsed if elapsed > 0 else 0
    eta = (TOTAL_IMAGES - total) / speed / 60 if speed > 0 else 0

    pbar.set_postfix({
        "imgs": f"{total:,}",
        "img/s": f"{speed:.0f}",
        "ETA": f"{eta:.1f}m",
        "load": f"{t_load_total:.0f}s",   # ← 이게 크면 I/O 병목
        "infer": f"{t_infer_total:.0f}s", # ← 이게 크면 GPU 병목
    })

# 나머지
if part_embs:
    save_partition(part_embs, part_meta, partition_idx)

elapsed = time.time() - t_start
print(f"\nDone. total={total:,}  partitions={partition_idx + 1}")
print(f"Total: {elapsed/60:.1f}min | Load: {t_load_total:.0f}s | Infer: {t_infer_total:.0f}s")