# AGENTS.md

# Research Ponytail Mode for Slurm GPU Experiments

You are working on a research codebase used through VSCode Remote-SSH on a GPU server.
The code is often executed through Slurm jobs, not on the local machine.

## Core principle

Make the smallest correct change that preserves reproducibility, experiment traceability, and rerun safety.
Be minimal, but never careless.

## Do not over-engineer

- Do not add new dependencies unless clearly necessary.
- Do not create unnecessary classes, factories, registries, plugins, or framework-like abstractions.
- Do not split files unless the current file is genuinely hard to maintain.
- Do not rewrite working code just to make it look cleaner.
- Do not introduce speculative config options that are not used by the current experiment.
- Do not create a new training framework when a simple script or small function change is enough.

## Research requirements that must NOT be removed

- Preserve seed control.
- Preserve or add clear experiment condition names.
- Preserve output paths, result file saving, logs, metrics, and config dumps.
- Never silently overwrite previous experiment results.
- Keep dataset paths, checkpoint paths, model names, split names, and output directories configurable.
- Keep code understandable for later paper writing, ablation analysis, and result reproduction.
- Prefer explicit experiment names over clever abstractions.

## Slurm / GPU server rules

- Do not assume local execution.
- Do not run full training jobs unless explicitly asked.
- Do not submit `sbatch` jobs unless explicitly asked.
- For validation, prefer cheap checks first:
  - `python -m py_compile <file>`
  - `python <script>.py --help`
  - small `--dry-run` or `--limit 2` runs if available
  - lightweight unit/smoke tests if present
- If GPU execution is necessary, suggest the exact `srun` or `sbatch` command instead of launching a long job automatically.
- Do not create or modify conda environments unless explicitly requested.
- Do not change CUDA, PyTorch, Slurm, or environment setup files unless the task is specifically about environment problems.

## Preferred implementation style

- Use Python standard library and already-installed project dependencies first.
- Prefer simple functions over classes.
- Prefer `argparse`, `pathlib`, `json`, `csv`, and existing project utilities.
- Prefer one clear experiment script over many tiny files.
- Add shared utilities only when at least two scripts actually use them.
- Keep diffs small and localized.
- When adding non-trivial logic, include one minimal runnable check or dry-run path.

## When modifying existing code

- First inspect existing helpers before adding new ones.
- Preserve existing naming and directory conventions.
- Make the smallest diff that solves the requested issue.
- Do not refactor unrelated code.
- Do not change experiment semantics unless explicitly requested.
- Explain what was intentionally not changed.

## Output expectation

After editing code, report:

1. What changed.
2. Which files changed.
3. How to run a cheap validation.
4. How to run the real Slurm/GPU experiment if applicable.
5. What was intentionally kept simple.