# SETUP_CLIP_ENV.md — clip-retrieval in an isolated conda env

The pod analysis env and the image-retrieval stack pin **incompatible numpy /
pyarrow** versions. Installing clip-retrieval into the pod env breaks the numpy
your option-planning / eval code depends on. So image retrieval runs in its own
`clip` conda env; the two never share a Python.

What lives where:
- **pod env**  → profile/query/option generation, validation, text-only eval
  (already done; do NOT pip-install clip-retrieval here).
- **clip env** → img2dataset, clip-retrieval inference/index/back, and
  `vit/collect_images_clip_retrieval.py`.

The collector itself only needs `requests` + `Pillow` (+ optional `scikit-image`
for the SSIM gate). The VLM attribute check is an HTTP call to your vLLM server
on :8002, so no torch is needed inside the collector process.

────────────────────────────────────────────────────────────
## 0. Create the isolated env
────────────────────────────────────────────────────────────
```bash
conda create -n clip python=3.10 -y
conda activate clip
pip install clip-retrieval img2dataset autofaiss
pip install requests pillow scikit-image
```
Sanity:
```bash
python -c "import clip_retrieval, img2dataset, autofaiss; print('clip env OK')"
```

────────────────────────────────────────────────────────────
## A. (DEPRECATED) hosted LAION backend — DOES NOT WORK
────────────────────────────────────────────────────────────
The public LAION knn-service (knn.laion.ai) was taken OFFLINE on 2023-12-19
after CSAM was found in LAION-5B, and has not been restored. `--backend hosted`
will time out. There is no public hosted fallback. Use the local backend below.
(The CSAM history is also why LAION-5B is not an appropriate image source for a
fashion benchmark in the first place — your Amazon corpus is the right call.)

────────────────────────────────────────────────────────────
## B. LOCAL clip-retrieval back (Amazon corpus) — THE ONLY PATH
────────────────────────────────────────────────────────────
Standard rom1504/clip-retrieval pipeline. Run all of this in the `clip` env.
`$WORK` is a scratch dir with space for images + embeddings + index.

### B1. URL list → from your Amazon metadata
You already generated this CSV (columns: `url,caption`) at
`~/pod_bench/corpus/amazon_fashion_urls.csv` (~116 MB). No need to rebuild it.
If you ever do rebuild: `python amazon_to_clip_corpus.py --out ~/pod_bench/corpus/amazon_fashion_urls.csv`

```bash
conda activate clip
export WORK=/home1/hjhj6411/pod_bench/data/clip_corpus
export URLCSV=~/pod_bench/corpus/amazon_fashion_urls.csv
head -2 $URLCSV          # confirm header is: url,caption
wc -l $URLCSV            # number of products
```

### B2. Download images into webdataset shards (img2dataset)

IMPORTANT — the download can stall after the first round of shards if the
process pool dead-locks (seen as: the first `processes_count` shards appear
almost instantly, then no further `.tar` files are written even though RAM,
disk and bandwidth are all fine). This is a multiprocessing-distributor /
oversubscription issue, NOT a resource issue. Avoid it with:
  - a moderate `--processes_count` (8, not 16) so workers don't oversubscribe
    a shared node,
  - explicit `--timeout 10 --retries 1` so a slow/hanging URL batch fails fast
    instead of blocking a worker forever,
  - `--incremental incremental` so re-runs resume from completed shards,
  - DO NOT pipe stderr to /dev/null — you need to see where it stalls.

Clean start (first run):
```bash
rm -rf $WORK/images          # only if restarting from scratch
img2dataset \
  --url_list $URLCSV \
  --input_format csv \
  --url_col url --caption_col caption \
  --output_folder $WORK/images \
  --output_format webdataset \
  --processes_count 8 --thread_count 64 \
  --image_size 384 --resize_mode keep_ratio \
  --timeout 10 --retries 1 \
  --incremental incremental \
  --enable_wandb False
```

If it stalls anyway, kill and resume (completed shards are skipped):
```bash
pkill -f img2dataset ; sleep 3
# re-run the SAME command; --incremental picks up where it left off
```

Sanity while it runs (separate shell, with $WORK exported there too):
```bash
echo "WORK=[$WORK]"                    # must NOT be empty
watch -n 30 'ls $WORK/images/*.tar 2>/dev/null | wc -l'   # should climb
```
Note: each img2dataset run creates one `.tar` + one `.parquet` per shard. The
`.parquet` holds per-URL status — use it for the success-rate check below.

### B3. CLIP-embed the shards (clip-retrieval inference)
Use the SAME clip model end-to-end. ViT-L/14 matches LAION indices and the
hosted service; if you prefer fashion-tuned retrieval you can swap the model,
but keep inference and `back` on the SAME model string.

```bash
export WORK=/home1/hjhj6411/pod_bench/data/clip_corpus
clip-retrieval inference \
  --input_dataset "$WORK/images/{00000..00065}.tar" \
  --output_folder $WORK/embeddings \
  --input_format webdataset \
  --enable_metadata True \
  --clip_model "ViT-L/14"
```

### B4. Build the FAISS index (autofaiss via clip-retrieval index)
```bash
clip-retrieval index \
  --embeddings_folder $WORK/embeddings \
  --index_folder $WORK/index
```

### B5. Write indices_paths.json BEFORE starting back
`clip-retrieval back` needs a config that points indice_name → index + metadata.
Create `$WORK/indices_paths.json`:
```json
{
  "pod_fashion": {
    "indice_folder": "/home1/hjhj6411/pod_bench/data/clip_corpus/index",
    "provide_safety_model": false,
    "enable_faiss_memory_mapping": true,
    "columns_to_return": ["url", "caption"],
    "metadata_folder": "/home1/hjhj6411/pod_bench/data/clip_corpus/embeddings/metadata"
  }
}
```
(Confirm the exact `index` / `metadata` subfolder names autofaiss produced and
fix the paths to match.)

### B6. Start the KNN backend
```bash
clip-retrieval back \
  --port 1234 \
  --indices-paths $WORK/indices_paths.json \
  --enable_faiss_memory_mapping True \
  --clip_model "ViT-L/14"
```
Leave this running (tmux/screen). Note the flag is `--enable_faiss_memory_mapping`
(not "enableaiss"), and `--clip_model` MUST equal the one used in B3.

Health check:
```bash
curl -s -X POST http://127.0.0.1:1234/knn-service \
  -H 'Content-Type: application/json' \
  -d '{"text":"navy wool coat","modality":"image","num_images":3,"num_result_ids":3,"indice_name":"pod_fashion"}'
```

Or use the built-in self-test (validates the whole client path, not just curl):
```bash
python vit/collect_images_clip_retrieval.py --backend local \
  --client-url http://127.0.0.1:1234/knn-service --indice-name pod_fashion --selftest
```
It fires one KNN query for "navy wool coat" and prints up to 5 candidate URLs.
If it prints "0 candidates", `back` isn't reachable or --indice-name is wrong.

### B7. Collect against the local backend
In a second `clip`-env shell (vLLM VLM up on :8002):
```bash
conda activate clip
# pilot
python vit/collect_images_clip_retrieval.py --backend local \
  --client-url http://127.0.0.1:1234/knn-service --indice-name pod_fashion \
  --limit 30 --verify lenient --top_k 20
# full run (resumable; re-run to continue, skips done query_ids)
python vit/collect_images_clip_retrieval.py --backend local \
  --client-url http://127.0.0.1:1234/knn-service --indice-name pod_fashion \
  --verify lenient --top_k 20
```

────────────────────────────────────────────────────────────
## Outputs & resumability
────────────────────────────────────────────────────────────
- Images:  `data/images/<query_id>/{A,B,C,D}.jpg`
- Log:     `data/images/collection_log.jsonl` (one record per query_id)
- Re-running resumes: query_ids already in the log are skipped, and existing
  per-option jpgs are reused. Delete the log (or pass `--force`) to restart.
- `complete=` is the share of instances with all 4 images. Expect some loss;
  with 1,851 plans a 70–85% complete rate yields ~1,300–1,570 usable instances,
  still ample. The downstream report should headline the counterbalanced subset.

────────────────────────────────────────────────────────────
## Verify modes (quality vs yield trade-off)
────────────────────────────────────────────────────────────
- `off`     : plumbing smoke test only (accept first downloadable image).
- `lenient` : VLM must match garment AND color (recommended default; pattern
              detection on product shots is noisy).
- `strict`  : garment AND color AND pattern. Use once you confirm the VLM reads
              pattern reliably on your corpus, else yield drops sharply.

The homogeneity gate runs only within {A,B} and {C,D} (same garment, differ on
the active axis); A↔C are different garments by design and are never compared.
The SSIM gate is informational when scikit-image is absent (won't fail an
instance); install scikit-image in the clip env to enable it.
