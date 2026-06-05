# SETUP_FSIGLIP.md — FashionSigLIP image pipeline (extract → index → serve → collect → evaluate)

This is the **FashionSigLIP** replacement for the old `SETUP_CLIP_ENV.md` (ViT-L/14
clip-retrieval) path. It reuses the **already-downloaded** img2dataset image bytes in
`clip_corpus/` (read-only) and re-embeds them with **Marqo-FashionSigLIP**, then serves a
self-contained FAISS KNN endpoint, runs the **Decomposed Perceptual Verification (DPV)**
collector, and scores collection quality.

Nothing here touches retrieval quality after this doc — retrieval is **frozen** on
FashionSigLIP. The old ViT clip-retrieval backend (`:1234`, `clip` env) is no longer used.

```
clip_corpus/            (READ-ONLY: 66 img2dataset .tar shards + metadata parquet)  ── source of image bytes
fashionsiglip_corpus/   (NEW, written here: shards/, index.faiss, ids.parquet, meta.json)
```

---

## 0. Layout, hardware, conda envs

Server paths (do not change):
```bash
export CLIP_CORPUS=/home1/hjhj6411/pod_bench/data/clip_corpus            # READ-ONLY source
export FS_CORPUS=/home1/hjhj6411/pod_bench/data/fashionsiglip_corpus     # NEW output
```
Hardware: 4× NVIDIA RTX A6000 48GB. Model cache already has `models--Marqo--marqo-fashionSigLIP`.

Two conda envs, never shared (numpy/pyarrow pins conflict):

- **`fsiglip`** — embedding + FAISS + KNN server (open_clip, faiss, flask). No repo deps.
- **`pod`** — generation/eval + the collectors/eval scripts (needs `requests`, `Pillow`).

```bash
# fsiglip env (extraction / index / serving)
conda create -n fsiglip python=3.10 -y && conda activate fsiglip
pip install open_clip_torch transformers pillow pandas pyarrow flask numpy tqdm webdataset
pip install faiss-gpu-cu12            # GPU FAISS for build + serve
python -c "import open_clip, faiss, flask, transformers; print('fsiglip env OK')"
```
`pod` env already exists (profile/query/option/eval). The collectors and eval scripts only
need `requests` + `Pillow`, both present in `pod`.

Scripts referenced (place in repo `src/` unless noted):
- `extract_fsiglip.py`, `build_faiss_fsiglip.py`, `serve_fsiglip_knn.py`  (fsiglip env)
- `collect_images_siglip_dpv.py`  (pod env — the current collector)
- `eval_extract.py`, `eval_score.py`, `eval_gold.py`, `eval_canon.py`, `CODEBOOK.md`  (pod env)

---

## 1. Pre-cache the model ONCE (avoid concurrent-download races)

Before launching 4 parallel extraction processes, materialize the HF weights once:
```bash
conda activate fsiglip
python -c "import open_clip; open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP'); open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionSigLIP'); print('cached')"
```

---

## 2. Extract embeddings from the existing tars (`extract_fsiglip.py`)

Encodes the 66 webdataset tars in `$CLIP_CORPUS/images/*.tar` with FashionSigLIP,
L2-normalized (768-dim), writing per-shard `shards/<stem>.npy` + `.parquet` into
`$FS_CORPUS`. Reads `clip_corpus` read-only; **per-shard resumable** (re-run skips done shards).
Uses a torch DataLoader so decode/preprocess runs in workers and the GPU isn't starved.

Multi-GPU by shard split — one process per GPU:
```bash
conda activate fsiglip
for G in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$G python src/extract_fsiglip.py \
      --gpu $G --num-gpus 4 --batch 256 --workers 8 \
      --clip-corpus "$CLIP_CORPUS" --out "$FS_CORPUS" &
done
wait
```
Monitor (separate shell): `watch -n 30 'ls $FS_CORPUS/shards/*.npy 2>/dev/null | wc -l'` (should climb to 66).
Expected total ≈ **651,489** vectors (the true count of downloaded jpgs across the 66 tars).
If a process dies, just re-run the same loop — completed shards are skipped.

---

## 3. Build the FAISS index (`build_faiss_fsiglip.py`)

Concatenates the shard `.npy` into one `IndexFlatIP` (cosine = inner product on normalized
vectors) and joins shard `key` → `url`/`caption` from the img2dataset metadata parquet.
Paths are hardcoded to `$CLIP_CORPUS` / `$FS_CORPUS`; DIM=768.
```bash
conda activate fsiglip
python src/build_faiss_fsiglip.py
# -> $FS_CORPUS/index.faiss, ids.parquet (key,url,caption), meta.json
# expect: "DONE  vectors=651489  with_url=651489"
```
`with_url` should equal `vectors` (100% url join). If not, the metadata parquet glob is wrong.

---

## 4. Serve the KNN endpoint (`serve_fsiglip_knn.py`)

Flask service on `:1235`, endpoint `POST /knn-service`, matching the collector's HTTP contract:
request `{text, modality, num_images, indice_name}` → `[{url, image_url, caption, similarity}, ...]`.
Query text is embedded with the SAME FashionSigLIP model and searched against the index.
```bash
conda activate fsiglip
python fsiglip/serve_fsiglip_knn.py --port 1235 --gpu 0      # leave running (tmux/screen)
```
Health + retrieval sanity:
```bash
curl -s http://127.0.0.1:1235/health                      # {"ntotal": 651489}
curl -s -X POST http://127.0.0.1:1235/knn-service \
  -H 'Content-Type: application/json' \
  -d '{"text":"beige plaid shirt","modality":"image","num_images":5,"indice_name":"pod_fashion"}'
```
Or the built-in self-test (validates the whole client path):
```bash
conda activate pod
python src/collect_images_siglip_dpv.py --selftest \
  --client-url http://127.0.0.1:1235/knn-service --top_k 5
```
Note: SigLIP cosine sits in a narrow ~0.11–0.16 band — judge **rank + caption**, not the
absolute similarity value.

---

## 5. Serve the verification VLM(s)

The DPV collector's three gates are HTTP calls to an OpenAI-compatible vLLM endpoint.
Default collector model is the lean **Qwen3-VL-4B** (one focused question per call). Run one
instance per free GPU for round-robin throughput, e.g.:
```bash
conda activate pod
for G in 0 1 2 3; do P=$((8002+G)); CUDA_VISIBLE_DEVICES=$G \
  vllm serve Qwen/Qwen3-VL-4B-Instruct --port $P \
    --max-model-len 32768 --limit-mm-per-prompt '{"image": 1}' & done
```
(Or a single larger model with `--tensor-parallel-size 4` on one port.)
**Confirm the served id BEFORE collecting** — `--vlm-model` must match exactly, else every
gate fail-opens to pass and verification is silently disabled:
```bash
curl -s http://127.0.0.1:8002/v1/models | python -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])"
```

---

## 6. Collect with Decomposed Perceptual Verification (`collect_images_siglip_dpv.py`)

TOP-1 over the frozen FashionSigLIP retrieval, with three deterministic, single-purpose
gates routed by instance metadata (no agent, no rewrite, no best-of-k):
1. **prominence** (always, first, short-circuits) — garment worn AND the main subject.
2. **demographic** (always, label) — `{gender, age}`; reject child; set-level gender lock.
3. **pattern-coverage** (routed: only `active_axis==pattern` AND option's target pattern
   non-solid) — COVERS_WHOLE on a deterministic torso crop. Skipped otherwise.
All gates are **fail-open** (VLM failure → pass), so a flaky VLM degrades to plain top-1.

```bash
conda activate pod
python src/collect_images_siglip_dpv.py \
  --plan_path data/options/option_plans.jsonl \
  --client-url http://127.0.0.1:1235/knn-service \
  --vlm-urls "http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1,http://127.0.0.1:8005/v1" \
  --vlm-model Qwen/Qwen3-VL-4B-Instruct \
  --gates prominence,demographic,pattern --gender-lock majority \
  --image_root data/images_dpv --output data/images_dpv/collection_log.jsonl \
  --workers 4 --top_k 12 --limit 30 --force
```
Outputs: `data/images_dpv/<query_id>/{A,B,C,D}.jpg` + `collection_log.jsonl` (one record per
query_id, with per-option `rank_used`/`model_gender`/`model_age`/`pattern_coverage` and
set-level `set_gender`/`gender_consistent`). Resumable: re-run to continue; `--force` restarts.

Ablations (compare via the eval in §7):
```bash
--gates ""                       # pure top-1 (no gates)
--gates prominence               # framing gate only
--pattern-dual                   # require BOTH upper- and lower-torso crops COVERS_WHOLE
--no-pattern-crop                # ask the pattern question on the full image (weaker)
--gender-lock anchor             # lock gender from option A instead of majority vote
```

---

## 7. Evaluate collection quality (independent judge + human anchor)

Three-layer evaluation (see `CODEBOOK.md` for labeling rules). The evaluator VLM **must
differ** from the collector (anti-circularity): collector = Qwen3-VL-4B, evaluator =
a larger / different-family model (e.g. `OpenGVLab/InternVL3-78B-Instruct` or
`Qwen/Qwen2.5-VL-72B-Instruct`). Serve that model on an OpenAI-compatible port first.

```bash
conda activate pod
# (1) Layer 1 — blind structured extraction over collected images (multi-method)
python src/eval_extract.py \
  --logs off:data/images_siglip_top1/collection_log.jsonl,dpv:data/images_dpv/collection_log.jsonl \
  --vlm-urls http://127.0.0.1:8002/v1 \
  --vlm-model OpenGVLab/InternVL3-78B-Instruct \
  --collector-model Qwen/Qwen3-VL-4B-Instruct \
  --out data/eval/extractions.jsonl --workers 4 --force

# (2) Layers 1+2 — rule-based scoring: method × axis accuracy + pattern_full + confusion
python src/eval_score.py \
  --extractions data/eval/extractions.jsonl --plans data/options/option_plans.jsonl \
  --logs off:data/images_siglip_top1/collection_log.jsonl,dpv:data/images_dpv/collection_log.jsonl \
  --out data/eval/scores.json --per-option data/eval/per_option_scored.jsonl

# (3) Layer 3 — single-annotator gold set (blind sheet → κ)
python src/eval_gold.py make --extractions data/eval/extractions.jsonl \
  --plans data/options/option_plans.jsonl --n 150 --oversample-pattern \
  --out-csv data/eval/goldsheet.csv --key data/eval/goldsheet_key.json
#   fill goldsheet.csv per CODEBOOK.md (do NOT open the key); re-label a subset after 2 weeks = pass2
python src/eval_gold.py kappa --filled data/eval/goldsheet_filled.csv \
  --key data/eval/goldsheet_key.json --extractions data/eval/extractions.jsonl \
  --filled2 data/eval/goldsheet_pass2.csv --out data/eval/kappa.json
```
The `eval_score` table (method × axis, with the headline **pattern_full** column) is the
numeric justification for the gates; `kappa.json` (human-vs-VLM + intra-annotator) licenses
trusting the auto metric on all N.

---

## Resumability & troubleshooting

- **Extraction** is per-shard idempotent; **collection** and **eval_extract** are per-record
  idempotent (skip done; `--force` restarts). FAISS build is cheap — just re-run.
- **`/health` ntotal ≠ 651489** → a shard `.npy` is missing or empty; re-run §2 for that GPU.
- **`with_url` < `vectors`** in §3 → metadata parquet path/glob wrong in `build_faiss_fsiglip.py`.
- **Every gate passes / no rejects** → `--vlm-model` ≠ served id (gates fail-open). Re-check §5.
- **KNN self-test prints 0 candidates** → server down or wrong `--indice-name` (must be `pod_fashion`).
- **Never modify `clip_corpus/`** — it is the read-only source of image bytes for both the old
  ViT path and this one.
