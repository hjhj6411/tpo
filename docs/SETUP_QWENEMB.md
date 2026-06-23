# SETUP_QWENEMB.md — Qwen3-VL-Embedding image pipeline (extract → index → serve → collect → evaluate)

This is the **QwenEmb** retrieval-backend experiment parallel to `SETUP_FSIGLIP.md`.
It reuses the **already-downloaded** img2dataset image bytes in `data/clip_corpus/`
(read-only), re-embeds them with a Qwen VL embedding model, serves a self-contained
FAISS KNN endpoint, runs the same DPV collector, and compares collection quality
against FashionSigLIP.

The purpose is not to replace FashionSigLIP blindly. The purpose is to test whether
Qwen3-VL-Embedding-8B gives better retrieval for POD-Bench fashion options under
the same downstream collection/evaluation protocol.

Nothing here writes into `data/fashionsiglip_corpus/`. All QwenEmb artifacts live
under `QwenEmb/` by default.

```bash
clip_corpus/       (READ-ONLY: existing img2dataset .tar shards + metadata parquet)
QwenEmb/           (NEW: scripts + corpus/)
QwenEmb/corpus/    (NEW, written here: shards/, index.faiss, ids.parquet, meta.json)
```

---

## 0. Layout, hardware, conda envs

Server paths:

```bash
export CLIP_CORPUS=/home1/hjhj6411/pod_bench/data/clip_corpus        # READ-ONLY source
export QWENEMB_ROOT=/home1/hjhj6411/pod_bench/QwenEmb                # scripts + output root
export QWENEMB_CORPUS=/home1/hjhj6411/pod_bench/QwenEmb/corpus       # NEW output
export QWENEMB_MODEL=Qwen/Qwen3-VL-Embedding-8B
```

Hardware target: 4× NVIDIA RTX A6000 48GB.

Two conda envs, never shared unless you know the dependency pins are compatible:

- **`qwenemb`** — Qwen embedding extraction + FAISS + KNN server.
- **`pod`** — option generation / DPV collection / evaluation scripts.

```bash
conda create -n qwenemb python=3.10 -y && conda activate qwenemb
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install sentence-transformers transformers accelerate pillow pandas pyarrow flask numpy tqdm
pip install faiss-gpu-cu12
python -c "import torch, sentence_transformers, transformers, faiss, flask; print('qwenemb env OK')"
```

If the Hugging Face model id differs in your cache or Qwen releases a new path, keep
the scripts unchanged and only override:

```bash
export QWENEMB_MODEL=<actual multimodal Qwen VL embedding model id>
```

Scripts added under `QwenEmb/`:

- `qwenemb_encoder.py` — small adapter around the model's embedding API.
- `extract_qwenemb.py` — tar shard image embedding extraction.
- `build_faiss_qwenemb.py` — FAISS index + metadata join.
- `serve_qwenemb_knn.py` — clip-retrieval-compatible KNN HTTP service.

---

## 1. Pre-cache the model ONCE

Avoid concurrent download races before launching multiple extraction workers:

```bash
conda activate qwenemb
python - <<'PY'
import os
from QwenEmb.qwenemb_encoder import QwenEmbConfig, QwenEmbEncoder
enc = QwenEmbEncoder(QwenEmbConfig(model_id=os.environ.get("QWENEMB_MODEL", "Qwen/Qwen3-VL-Embedding-8B"), device="cuda", dtype="bfloat16"))
print('cached / loaded:', enc.cfg.model_id)
PY
```

If this fails with `Could not obtain image embeddings` or `Could not obtain text embeddings`,
the model's remote-code API uses a method name not covered by `QwenEmb/qwenemb_encoder.py`.
Patch only that adapter, then keep the rest of the pipeline unchanged.

---

## 2. Smoke test one tiny extraction

Before embedding all shards, run a tiny extraction to verify the model API and output dimension:

```bash
conda activate qwenemb
CUDA_VISIBLE_DEVICES=0 python QwenEmb/extract_qwenemb.py \
  --model-id "$QWENEMB_MODEL" \
  --clip-corpus "$CLIP_CORPUS" --out "$QWENEMB_CORPUS" \
  --gpu 0 --num-gpus 1 --batch 4 --limit-shards 1 --limit-images 16 --force

ls -lh "$QWENEMB_CORPUS"/shards
```

Expected: one `.npy` and one `.parquet` under `QwenEmb/corpus/shards/`.

Clean the smoke-test shard before the full extraction if you used very small limits:

```bash
rm -f "$QWENEMB_CORPUS"/shards/*.npy "$QWENEMB_CORPUS"/shards/*.parquet
```

---

## 3. Extract embeddings from existing tars

Encodes `$CLIP_CORPUS/images/*.tar` with QwenEmb, L2-normalizes embeddings, and writes
per-shard `shards/<stem>.npy` + `.parquet` into `$QWENEMB_CORPUS`.

Extraction is per-shard resumable. Re-running skips completed shards unless `--force` is set.

```bash
conda activate qwenemb
for G in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$G python QwenEmb/extract_qwenemb.py \
    --model-id "$QWENEMB_MODEL" \
    --gpu $G --num-gpus 4 --batch 32 \
    --clip-corpus "$CLIP_CORPUS" --out "$QWENEMB_CORPUS" &
done
wait
```

Monitor in another shell:

```bash
watch -n 30 'ls /home1/hjhj6411/pod_bench/QwenEmb/corpus/shards/*.npy 2>/dev/null | wc -l'
```

Expected total is the true count of downloaded jpgs across the existing tars. For the
current FashionSigLIP doc this was about 651,489 images, but trust the script output
rather than a hard-coded number.

---

## 4. Build the FAISS index

Concatenates all QwenEmb shard vectors into one `IndexFlatIP` index. Since extraction
already L2-normalizes vectors, inner product is cosine similarity.

```bash
conda activate qwenemb
python QwenEmb/build_faiss_qwenemb.py \
  --clip-corpus "$CLIP_CORPUS" \
  --corpus "$QWENEMB_CORPUS" \
  --index-name pod_qwenemb
```

Outputs:

```bash
$QWENEMB_CORPUS/index.faiss
$QWENEMB_CORPUS/ids.parquet
$QWENEMB_CORPUS/meta.json
```

`with_url` should be close to `vectors`. If it is much smaller, the metadata parquet
join is wrong; inspect `--metadata-glob` in `build_faiss_qwenemb.py`.

---

## 5. Serve the QwenEmb KNN endpoint

The server exposes the same HTTP contract as the collector expects:

```text
POST /knn-service
request:  {text, modality, num_images, indice_name}
response: [{url, image_url, caption, title, key, similarity, source}, ...]
```

Run it on `:1236`, separate from FashionSigLIP's `:1235`:

```bash
conda activate qwenemb
CUDA_VISIBLE_DEVICES=0 python QwenEmb/serve_qwenemb_knn.py \
  --model-id "$QWENEMB_MODEL" \
  --corpus "$QWENEMB_CORPUS" \
  --port 1236 --index-name pod_qwenemb
```

Health + retrieval sanity:

```bash
curl -s http://127.0.0.1:1236/health
curl -s -X POST http://127.0.0.1:1236/knn-service \
  -H 'Content-Type: application/json' \
  -d '{"text":"beige checkered shirt","modality":"image","num_images":5,"indice_name":"pod_qwenemb"}'
```

Judge retrieval by rank and caption/image inspection, not absolute similarity values.
QwenEmb and FashionSigLIP cosine ranges may not be directly comparable.

---

## 6. Serve the verification VLM(s)

Use the same DPV verifier as the FashionSigLIP pipeline so the comparison isolates the
retrieval backend.

```bash
conda activate pod
for G in 0 1 2 3; do P=$((8002+G)); CUDA_VISIBLE_DEVICES=$G \
  vllm serve Qwen/Qwen3-VL-4B-Instruct --port $P \
    --max-model-len 32768 --limit-mm-per-prompt '{"image": 1}' &
done
```

Confirm the served id before collecting:

```bash
curl -s http://127.0.0.1:8002/v1/models | python -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])"
```

---

## 7. Collect with the same DPV collector

Point the existing collector at the QwenEmb KNN endpoint. Keep image/log paths distinct:

```bash
conda activate pod
python src/collect_images_siglip_dpv.py \
  --plan_path data/options/option_plans.jsonl \
  --client-url http://127.0.0.1:1236/knn-service \
  --vlm-urls "http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1,http://127.0.0.1:8005/v1" \
  --vlm-model Qwen/Qwen3-VL-4B-Instruct \
  --gates prominence,demographic,pattern --gender-lock majority \
  --image_root data/images_qwenemb_dpv --output data/images_qwenemb_dpv/collection_log.jsonl \
  --workers 4 --top_k 12 --limit 30 --force
```

For a full run, remove `--limit 30`.

Ablations:

```bash
--gates ""                       # pure QwenEmb top-1
--gates prominence               # framing gate only
--pattern-dual                   # stricter whole-pattern check
--gender-lock anchor             # lock gender from option A
```

---

## 8. Evaluate against FashionSigLIP

Use the same eval stack and compare logs by method name:

```bash
conda activate pod
python src/eval_extract.py \
  --logs fsiglip:data/images_dpv/collection_log.jsonl,qwenemb:data/images_qwenemb_dpv/collection_log.jsonl \
  --vlm-urls http://127.0.0.1:8002/v1 \
  --vlm-model OpenGVLab/InternVL3-78B-Instruct \
  --collector-model Qwen/Qwen3-VL-4B-Instruct \
  --out data/eval/qwenemb_vs_fsiglip_extractions.jsonl --workers 4 --force

python src/eval_score.py \
  --extractions data/eval/qwenemb_vs_fsiglip_extractions.jsonl \
  --plans data/options/option_plans.jsonl \
  --logs fsiglip:data/images_dpv/collection_log.jsonl,qwenemb:data/images_qwenemb_dpv/collection_log.jsonl \
  --out data/eval/qwenemb_vs_fsiglip_scores.json \
  --per-option data/eval/qwenemb_vs_fsiglip_per_option_scored.jsonl
```

Key columns to compare:

- axis accuracy by method
- pattern_full / pattern coverage accuracy
- option-level failures by pattern value, especially `argyle`, `checkered`, `polka_dot`
- residual value skew after collection

---

## Resumability & troubleshooting

- **Extraction** is per-shard idempotent: completed `.npy + .parquet` shard pairs are skipped.
- **Index build** is cheap relative to extraction; re-run after any shard changes.
- **Server dimension mismatch** means `--dim` or model id differs between extraction and serving.
- **No URLs after build** means metadata join failed; check the parquet columns under `$CLIP_CORPUS`.
- **QwenEmb adapter cannot encode** means the model's remote-code API differs from the common
  `encode_*` / `get_*_features` methods. Patch only `QwenEmb/qwenemb_encoder.py`.
- **Do not compare raw cosine values** across FashionSigLIP and QwenEmb. Compare downstream
  collection/evaluation metrics.
- **Do not write into `data/fashionsiglip_corpus/`**. QwenEmb outputs stay in `QwenEmb/corpus/`.
