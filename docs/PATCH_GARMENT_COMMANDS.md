# Exact commands for the patch garment experiment

## Start QwenEmb server

```bash
cd ~/pod_bench
conda activate qwenemb

export QWENEMB_MODEL=Qwen/Qwen3-VL-Embedding-8B
export QWENEMB_CORPUS=/home1/hjhj6411/pod_bench/QwenEmb/corpus

CUDA_VISIBLE_DEVICES=0 python QwenEmb/serve_qwenemb_knn.py \
  --model-id "$QWENEMB_MODEL" \
  --corpus "$QWENEMB_CORPUS" \
  --host 127.0.0.1 \
  --port 1236 \
  --index-name pod_qwenemb
```

```bash
curl -s http://127.0.0.1:1236/health | python -m json.tool
```

## Visualize an image already downloaded by retrieval

```bash
cd ~/pod_bench
conda activate pod

git fetch origin
git checkout exp/patch-garment-viz

python retrieval/visualize_qwenemb_patch_garment.py \
  --image /absolute/path/to/image.jpg \
  --client-url http://127.0.0.1:1236/knn-service \
  --output-root data/retrieval/patch_garment_viz_one \
  --short-side-tiles 10 \
  --overlap 0.25 \
  --threshold 0.0 \
  --force
```

## Visualize images from a retrieval jsonl

```bash
python retrieval/visualize_qwenemb_patch_garment.py \
  --results-jsonl data/retrieval/qwenemb_square_gallery_smoke_allpatch/scored_results.jsonl \
  --max-images 12 \
  --client-url http://127.0.0.1:1236/knn-service \
  --output-root data/retrieval/patch_garment_viz_from_results \
  --short-side-tiles 10 \
  --overlap 0.25 \
  --threshold 0.0 \
  --force
```

## Run QwenEmb square-patch collection

Use `collect_topk_qwenemb_square_gallery_v2.py` on this branch. It monkey-patches the collector to use adaptive square patches.

```bash
python retrieval/collect_topk_qwenemb_square_gallery_v2.py \
  --plan-path data/options/option_plans.jsonl \
  --client-url http://127.0.0.1:1236/knn-service \
  --index-name pod_qwenemb \
  --output-root data/retrieval/qwenemb_square_gallery_smoke_allpatch \
  --retrieval-k 12 \
  --score-top-n 8 \
  --show-k 8 \
  --limit 1 \
  --workers 4 \
  --score-workers 1 \
  --tile-grid 10 \
  --pattern-max-tiles 0 \
  --force
```

Important: `--pattern-max-tiles 0` means all square patches in the square tiler. Do not use `12` for the actual local-pattern experiment unless you are only debugging speed.
