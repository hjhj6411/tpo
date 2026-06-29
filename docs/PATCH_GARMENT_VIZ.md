# Patch garment visualization workflow

This experiment is for checking whether QwenEmb can act as a lightweight patch-level foreground/background judge before introducing a heavier segmentation model.

The intended order is:

1. Use adaptive square patches.
2. Ask QwenEmb whether each patch contains visible clothing/fabric or non-clothing background.
3. Inspect an overlay image and patch gallery manually.
4. Only if this looks unreliable, move to a real segmentation/mask backend.

## 1. Start QwenEmb server

```bash
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

Health check:

```bash
curl -s http://127.0.0.1:1236/health | python -m json.tool
```

The server must expose `/score-image-files`.

## 2. Visualize one local image

```bash
conda activate pod

python retrieval/visualize_qwenemb_patch_garment.py \
  --image /path/to/image.jpg \
  --client-url http://127.0.0.1:1236/knn-service \
  --output-root data/retrieval/patch_garment_viz_one \
  --short-side-tiles 10 \
  --overlap 0.25 \
  --threshold 0.0 \
  --force
```

Open:

```bash
python -m http.server 8899 --directory data/retrieval/patch_garment_viz_one
```

Then open `http://127.0.0.1:8899/index.html` through your SSH tunnel or copy the output directory.

## 3. Visualize candidates from an existing retrieval result

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

## 4. Patch size tuning

Use one of these styles:

```bash
# Adaptive: shorter image side has 10 square patches
--short-side-tiles 10

# Finer patches
--short-side-tiles 14

# Fixed pixel side, useful when all corpus images have similar resolution
--patch-size 96
```

Recommended smoke settings:

```bash
--short-side-tiles 10 --overlap 0.25 --threshold 0.0
```

Recommended stricter settings after checking overlays:

```bash
--short-side-tiles 12 --overlap 0.25 --threshold 0.02
```

## 5. Interpreting outputs

Each image output directory contains:

```text
original.*
overlay.png
patch_scores.jsonl
summary.json
patches/*.jpg
```

`overlay.png` highlights clothing-positive patches in red. `index.html` shows original image, overlay, and all patch scores.

If red areas mostly follow clothing/fabric regions, this can replace a segmentation backend for the first ablation. If red areas often fire on background, skin, logos, packages, or scene objects, then proceed to a proper segmentation/mask backend.
