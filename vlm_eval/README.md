# Query + ABCD image-and-text VLM evaluation

This folder contains VLM experiments over the human-validated image library.
Every condition contains four shuffled options A--D, with one image and one
text description per option. Depending on the condition, the context is a
query, a profile, or both.

The `none` condition contains no user profile. The `narrative` and `key-value`
conditions additionally contain the corresponding profile representation from
`data_wacv_scenario_v4/profiles/profiles.jsonl`.

## Accuracy definition

Without a profile, original options A and B are both situation-compatible.
Therefore the query-only experiment's primary `accuracy` is **Scenario
Accuracy**:

```text
correct = predicted original option is A or B
random baseline = 50%
```

Original-A-only accuracy is reported as **TPO Accuracy** for the combined target;
it is not the primary query-only accuracy because the query alone cannot
distinguish the preferred A from the non-preferred B.

With a profile, original A is the only option satisfying both the query and the
profile. Query+profile therefore uses **TPO Accuracy** (original A) as its
primary metric (random baseline 25%). It also reports Scenario Accuracy (A/B)
and Profile Accuracy (A/C).

For profile-only conditions, original A and C both satisfy the profile.
**Profile Accuracy** (A/C) is therefore the primary metric (random baseline
50%); Scenario Accuracy (A/B) and TPO Accuracy (A) are diagnostics because the
query is omitted.

The evaluator always records all three values in every overall and per-axis
summary:

```text
Profile Accuracy  = original A or C
Scenario Accuracy = original A or B
TPO Accuracy      = original A
```

New result rows use `profile_correct`, `scenario_correct`, and
`joint_tpo_correct`. The older `tpo_correct` row field is retained as an A/B
legacy alias for resume compatibility; use `summary.json` or the three explicit
fields for analysis.

Physical and dress-code tracks are written to separate result and summary
files. Every track summary also reports separate color, garment, and pattern
active-axis accuracies. No pooled track accuracy is produced.

## Input selection

The evaluator reads `annotation/attribute_library.json` directly. A plan is
evaluated only if all four options have complete
`color|garment_category|pattern` keys and each exact cell has a current
`available` image. With the completed 2026-07-31 annotation snapshot this is
1,012 plans: 318 physical and 694 dress-code.

The evaluator never scans all files under `annotation/images_final/`, because
that directory contains superseded choices. It resolves only paths referenced
by the current attribute library.

## Cheap validation

This performs selection, decodes the selected images, and prints one shuffled
prompt mapping without writing results or calling a model:

```bash
conda activate pod

python -m vlm_eval.eval_query_abcd \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --profile-format narrative \
  --limit 2 \
  --dry-run
```

## Real VLM experiment

Start an OpenAI-compatible VLM server in an allocated GPU session. For
example, adjust GPU count and model settings to the Slurm allocation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --host 127.0.0.1 \
  --port 8002 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 4}' \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
```

Then run the evaluator from a separate shell:

```bash
conda activate pod

python -m vlm_eval.eval_query_abcd \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --base-urls http://127.0.0.1:8002/v1 \
  --concurrency 8
```

The command above is the query-only baseline (`--profile-format none` is the
default). Run the two profile conditions separately against the same server:

```bash
# Query + narrative profile + four (image, text) options
python -m vlm_eval.eval_query_abcd \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --base-urls http://127.0.0.1:8002/v1 \
  --profile-format narrative \
  --concurrency 8

# Query + key-value profile + four (image, text) options
python -m vlm_eval.eval_query_abcd \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --base-urls http://127.0.0.1:8002/v1 \
  --profile-format key-value \
  --concurrency 8
```

The key-value representation follows `text_exp` exactly:

```text
likes.garment: ...
dislikes.garment: ...
likes.color: ...
dislikes.color: ...
likes.pattern: ...
dislikes.pattern: ...
```

Profile-only uses the same two representations while removing the query and
all situation-specific instructions:

```bash
# Narrative profile only
python -m vlm_eval.eval_query_abcd \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --base-urls http://127.0.0.1:8002/v1 \
  --profile-format narrative \
  --profile-only \
  --concurrency 8

# Key-value profile only
python -m vlm_eval.eval_query_abcd \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --base-urls http://127.0.0.1:8002/v1 \
  --profile-format key-value \
  --profile-only \
  --concurrency 8
```

Multiple compatible endpoints can be round-robined:

```bash
--base-urls http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1
```

Do not launch the server or full evaluation on a login node. Allocate the GPUs
with the cluster's normal `srun` or `sbatch` workflow first.

## Outputs and resume behavior

Default condition names are derived from the profile format and seed:

```text
vlm_eval/results/query_abcd_image_text_seed42/<model>/
vlm_eval/results/query_profile_narrative_abcd_image_text_seed42/<model>/
vlm_eval/results/query_profile_key_value_abcd_image_text_seed42/<model>/
vlm_eval/results/profile_narrative_abcd_image_text_seed42/<model>/
vlm_eval/results/profile_key_value_abcd_image_text_seed42/<model>/
```

Each condition has the same output structure:

```text
<condition>/<model>/
├── run_config.json
├── summary.json
├── physical/
│   ├── results.jsonl
│   └── summary.json
└── dress_code/
    ├── results.jsonl
    └── summary.json
```

`results.jsonl` is append-only and written one response at a time. Re-running
the same command resumes unfinished/API-error plans. `run_config.json` records
the seed, prompt, model, input hashes, image size, and selection counts. If any
condition changes, the evaluator refuses to mix it into the old directory;
use a new `--experiment-name`.

Useful partial checks:

```bash
# One track only
python -m vlm_eval.eval_query_abcd --model MODEL --track physical

# Small real smoke run; use a distinct name so it cannot mix with the full run
python -m vlm_eval.eval_query_abcd \
  --model MODEL \
  --limit 4 \
  --experiment-name query_abcd_image_text_smoke_seed42
```
