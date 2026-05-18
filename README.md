# POD-Bench: Preference Origin Disentanglement Benchmark

A vision-essential benchmark for diagnosing whether VLM-based personalization
methods can disentangle a user's **intrinsic preference** from **situational TPO
(Time / Place / Occasion)** context.

## Core Idea

Each instance is a 4-option multiple-choice item where all options are images
of fashion items in the same garment category. The four options are constructed
along two independent axes:

```
                    matches preference?
                       YES        NO
TPO match?  YES        A          B          (A is correct)
            NO         C          D
```

- **A** = TPO + preference both satisfied (correct)
- **B** = TPO only — situationally appropriate but violates user taste
- **C** = preference only — matches taste but situationally wrong
- **D** = neither

The diagnostic power lies in the *confusion matrix*:
- B-bias → TPO over-weighting (PCogAlign-style failure)
- C-bias → preference over-weighting (Whose Boat? / SynthesizeMe / FSPO-style
  failure, i.e., "mechanical FAE")

## Design (Locked)

| Aspect | Choice | Rationale |
|---|---|---|
| Domain | Fashion only | Phase 1 simplicity |
| Profile language | English | International venue target |
| Profile structure | likes/dislikes keywords + narrative | MMPB-style, supports ablation |
| Attribute axes | Closed-set, objectively decidable | Unambiguous rule-based labeling |
| Image source | Amazon Reviews 2023 + Google fallback | Free, real catalog |
| Labeling | Rule-based + 3-judge LLM ensemble | No human annotation needed |
| Phase 1 size | 50 users × 20 queries | Pilot → expand on validation |

## Attribute Taxonomy (Closed Vocabulary)

The taxonomy is grounded in DeepFashion (Liu et al. CVPR 2016), Fashionpedia
(Jia et al. ECCV 2020), and Berlin & Kay (1969) basic color terms:

| Axis | Values |
|---|---|
| color | black, white, gray, navy, blue, red, pink, orange, yellow, green, brown, beige, purple |
| material | cotton, linen, wool, silk, cashmere, denim, leather, suede, polyester, nylon, knit, fleece |
| pattern | solid, striped, checkered, plaid, floral, polka_dot, graphic_print, camouflage, animal_print |
| garment_category | t_shirt, shirt, blouse, sweater, hoodie, jacket, coat, trench_coat, blazer, dress, skirt, pants, jeans, shorts, suit |
| fit | slim, regular, loose, oversized |
| sleeve_length | sleeveless, short_sleeve, three_quarter_sleeve, long_sleeve |
| neckline | crew_neck, v_neck, scoop_neck, turtleneck, collared, off_shoulder, hooded |
| formality_level | very_casual, casual, smart_casual, business_casual, formal |

**Excluded from likes/dislikes**: style labels like "minimalist", "classic",
"bohemian" — they are subjective and ambiguous. They may appear in the
narrative profile for fluency only.

## Profile Representation (MMPB-style)

Each profile has three representations to support ablation:

```json
{
  "user_id": "U001",
  "structured_attributes": {       // hidden ground-truth
    "color":    {"likes": ["navy", "beige"], "dislikes": ["orange"]},
    "material": {"likes": ["cotton", "linen"], "dislikes": ["polyester"]},
    "fit":      {"likes": ["regular"], "dislikes": ["oversized"]}
  },
  "likes_keywords":    ["color:navy", "color:beige", "material:cotton",
                        "material:linen", "fit:regular"],
  "dislikes_keywords": ["color:orange", "material:polyester", "fit:oversized"],
  "narrative_profile": "This early-30s researcher gravitates toward navy and
                        beige tones, and prefers natural-fiber pieces in cotton
                        or linen. They favor a regular fit and tend to avoid
                        polyester, bright orange tones, and oversized cuts."
}
```

Three exposure variants for ablation (in final benchmark):
- `keyword_only`: only the keyword lists
- `narrative_only`: only the narrative
- `combined`: narrative + keyword recap

## Query Types

| Type | Ratio | Example |
|---|---|---|
| explicit_tpo | 30% | "What should I wear for a rainy outdoor event this fall?" |
| implicit_tpo | 25% | "What should I wear to my friend's wedding?" |
| visual_tpo | 30% | "What should I wear to a place like this?" (+ TPO image) |
| neutral | 15% | "Pick an item that fits my personal style." |

## Pipeline

```
1. profile_generator  → 50 narrative+keyword profiles (English)
2. query_generator    → 20 queries/user across 4 TPO types
3. option_planner     → 4-option attribute specs per query
4. image_collector    → Amazon + Google image fetch + CLIP homogeneity
5. label_verifier     → rule + 3-judge ensemble + Krippendorff α / Cohen κ
6. quality_audit      → vision-essentiality controls + final assembly
7. evaluator          → VLM eval with position-shuffle + confusion matrix
```

## Cost Policy

- **Paid**: GPT-5-mini only (profile/query/option generation, 1 of 3 judges)
- **Free**:
  - Local vLLM: Qwen2.5-7B, Llama-3.1-8B (2 of 3 judges), Qwen2.5-VL-7B
    (captioner + VLM evaluator)
  - HuggingFace CLIP (homogeneity check)
  - Amazon Reviews 2023 (already downloaded)
  - Google Custom Search (free 100/day)

## GPT-5-mini API Notes

This codebase handles three specific quirks of GPT-5 reasoning models:

1. **Parameter name**: uses `max_completion_tokens`, not `max_tokens`.
2. **Reasoning budget**: reasoning tokens (300-500 typical) count toward the
   `max_completion_tokens` cap. Default is set to 4096, long-output mode 8192.
3. **No temperature**: GPT-5 family rejects custom temperature values
   (must use default 1). Our code does NOT pass `temperature` at all.
4. **Response parsing**: uses the Chat Completions endpoint
   (`/v1/chat/completions`) with `.choices[0].message.content` extraction
   that gracefully handles refusals, list-form content, and `finish_reason=length`
   (empty response when reasoning consumed the entire budget).

## Quick Start

```bash
unzip pod_bench.zip && cd pod_bench
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY=sk-...

# Start local vLLM (see docs/SETUP.md for full setup)
# Then run the pipeline:
bash scripts/run_pipeline.sh

# Evaluate
python -m src.evaluator --model vlm_evaluator --profile_variant combined
python -m src.evaluator --model vlm_evaluator --profile_variant keyword_only
python -m src.evaluator --model vlm_evaluator --profile_variant narrative_only
```

Each pipeline step supports `--limit N` for partial runs and is idempotent
(resumable on re-run).
