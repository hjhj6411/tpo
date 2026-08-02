# ViT-L/14 retrieval

This directory gathers the existing ViT-L/14 work that was previously spread
across `src/`, `scripts/`, and `docs/`. The old paths remain in place for
backwards compatibility; use the files here for new ViT runs.

## Existing assets

The completed corpus is kept outside this code directory because it is about
50 GB and is shared read-only by other retrieval backbones:

```text
data/clip_corpus/
├── images/                  # downloaded WebDataset shards
├── embeddings/             # ViT-L/14 embeddings + metadata
├── index/                   # FAISS image/text indices
└── indices_paths.json       # pod_fashion backend configuration
```

Do not rebuild or overwrite this corpus for an ordinary benchmark run.

## Files

- `serve_vit_l14.sh`: start the existing ViT-L/14 clip-retrieval backend.
- `run_coverage_v7.sh`: run the latest coverage collector.
- `collect_images_clip_retrieval.py`: full agent/VLM collector documented in
  `SETUP.md`.
- `collect_images_vit_top1.py`: minimal top-1 + gender/dedup baseline.
- `collect_images_vit_coverage_v6.py`: pattern-coverage variant.
- `collect_images_vit_coverage_v7.py`: latest pattern + color coverage variant.
- `amazon_to_clip_corpus.py`: rebuild the Amazon URL CSV only when necessary.
- `tests/test_v7_collector_mock.py`: server-free v7 logic smoke test.

## Environment and backend

Use the existing isolated `clip` environment. Do not install clip-retrieval in
the main POD environment.

```bash
conda activate clip
bash vit/serve_vit_l14.sh
```

In another shell:

```bash
conda activate clip
python vit/collect_images_clip_retrieval.py \
  --backend local \
  --client-url http://127.0.0.1:1234/knn-service \
  --indice-name pod_fashion \
  --selftest
```

## Latest coverage collector

The v7 collector uses ViT-L/14 for retrieval on port 1234. Its pattern/color
coverage reranking still calls the existing FashionSigLIP coverage endpoints
on port 1235; set those URLs empty to disable the corresponding coverage stage.

Pilot:

```bash
conda activate clip
LIMIT=10 WORKERS=2 bash vit/run_coverage_v7.sh
```

Full resumable run:

```bash
conda activate clip
bash vit/run_coverage_v7.sh
```

Outputs default to `data/images_vit_cov_c/`. Existing query IDs in
`collection_log.jsonl` are skipped. Set `FORCE=1` only when intentionally
starting a new result set.

## Important scope distinction

These collectors replace FashionSigLIP **retrieval** with ViT-L/14. They are
not a drop-in replacement for the newer SAM3 patch-ranking script's
`/score-image-files` API. That pipeline needs a ViT local-file scoring endpoint
before both retrieval and patch scoring can be fully ViT-based.

See `SETUP.md` for corpus construction, indexing, backend configuration, and
resumability details.
