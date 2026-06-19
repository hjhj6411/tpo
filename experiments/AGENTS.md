# Experiment Script Rules

This directory contains runnable experiment scripts.

Prefer direct, reproducible experiment scripts over reusable frameworks.
Do not introduce experiment manager classes, plugin registries, or complex config systems.

Each experiment script should clearly show:

- What question it tests.
- What conditions are compared.
- Which metric is computed.
- Where raw results are saved.
- Where summarized results are saved.

Required behavior:

- Keep `--seed`, `--output-dir`, `--data-path`, and `--limit` or `--dry-run` if applicable.
- Never overwrite existing result files without an explicit flag.
- Save arguments/configs next to results.
- Prefer JSONL/CSV outputs that can be analyzed later.
- Add only the minimal helper functions needed for the current experiment.
EOF