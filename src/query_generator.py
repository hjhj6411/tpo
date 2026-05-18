"""
Query Generator (English, Fashion)
====================================

Generates N queries per user, distributed across 4 query types:
  - explicit_tpo (30%): TPO directly stated in text
  - implicit_tpo (25%): TPO inferable from named situation
  - visual_tpo  (30%): vague text + separate TPO context image (image collected later)
  - neutral     (15%): pure preference, no TPO

Queries are deliberately INDEPENDENT of the user's preferences — they specify
situations only. The user's enduring taste is supplied via their profile.

All queries in ENGLISH.
"""

import argparse
import random
from pathlib import Path

from .utils import (
    call_gpt5_mini, parse_json_list_response,
    save_jsonl, load_jsonl, log_step,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    QUERIES_DIR, PROFILES_DIR, FASHION_TPO,
    QUERY_TYPE_RATIO, PHASE1_CONFIG, GPT5_MINI,
)


# ─────────────────────────────────────────────
# TPO scenario sampler
# ─────────────────────────────────────────────

def sample_tpo_scenario(seed: int) -> dict:
    """Sample a coherent TPO scenario from the fashion TPO axes."""
    rng = random.Random(seed)
    scenario = {}
    for axis, sub_axes in FASHION_TPO.items():
        scenario[axis] = {}
        sub_keys = list(sub_axes.keys())
        n_to_sample = rng.randint(1, len(sub_keys))
        for sk in rng.sample(sub_keys, n_to_sample):
            scenario[axis][sk] = rng.choice(sub_axes[sk])
    return scenario


def tpo_to_hint(scenario: dict) -> str:
    """Render TPO scenario as a hint string for the LLM."""
    parts = []
    for axis, sub in scenario.items():
        for k, v in sub.items():
            parts.append(f"{axis}.{k}={v}")
    return "; ".join(parts)


# ─────────────────────────────────────────────
# Prompts (English)
# ─────────────────────────────────────────────

QUERY_SYSTEM = """\
You write natural English queries that users would ask a fashion
recommendation assistant. Output ONLY a JSON list of strings, no other text.
"""


def build_explicit_prompt(scenario: dict, n: int) -> str:
    return f"""Generate {n} natural English fashion queries for the following TPO scenario.
The TPO information must appear EXPLICITLY in the query text (mention season,
weather, place, or occasion words directly).

TPO scenario hint: {tpo_to_hint(scenario)}

Requirements:
- Each query: 1-2 sentences, natural English
- Ask for a recommendation (e.g., "What should I wear...?", "Pick an outfit for...")
- Do NOT mention specific colors, materials, or styles (those come from the user's profile)

Good examples:
  "What should I wear for a rainy outdoor event this fall?"
  "I'm flying out for a winter trip — pick a coat for me."

Bad examples:
  "Recommend a navy raincoat."     (color specified — forbidden)
  "Pick something for me."          (no TPO)

Output JSON list: ["query1", "query2", ...]
"""


def build_implicit_prompt(scenario: dict, n: int) -> str:
    return f"""Generate {n} natural English fashion queries where the TPO is conveyed
IMPLICITLY through the name of a situation (rather than spelling out
weather/time/formality directly).

TPO scenario hint (for your understanding): {tpo_to_hint(scenario)}

Good examples:
  "What should I wear to a friend's wedding?"     (formality inferred from 'wedding')
  "Pick an outfit for my weekend hiking trip."    (outdoor/active inferred from 'hiking')

Bad examples:
  "What should I wear to a formal indoor wedding?"  (too explicit)
  "Pick an outfit."                                  (no situation)

Output JSON list: ["query1", "query2", ...]
"""


def build_visual_prompt(n: int) -> str:
    return f"""Generate {n} VAGUE English fashion queries that point to an
external image as the source of TPO information. The query text alone
should NOT reveal the situation.

Good examples:
  "What should I wear to a place like this?"
  "Pick an outfit that matches this vibe."
  "What's appropriate for the setting shown here?"

Bad examples:
  "What should I wear to a wedding?"   (situation revealed in text)
  "Pick an outfit."                     (too generic, no reference to image)

Output JSON list: ["query1", "query2", ...]
"""


def build_neutral_prompt(n: int) -> str:
    return f"""Generate {n} English fashion queries with NO TPO context — purely
about the user's taste, no situation, no season, no weather, no occasion.

Good examples:
  "Pick something I would like."
  "Which of these matches my taste best?"
  "Recommend an item that fits my personal style."

Bad examples:
  "Pick an office outfit."    (situational)
  "Anything for tonight?"      (time)

Output JSON list: ["query1", "query2", ...]
"""


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

def generate_queries_for_user(profile: dict, n_per_user: int) -> list[dict]:
    user_id = profile["user_id"]
    base_seed = abs(hash(user_id)) % 100000

    # allocate type counts
    type_counts = {}
    remaining = n_per_user
    for qtype, ratio in QUERY_TYPE_RATIO.items():
        c = int(n_per_user * ratio)
        type_counts[qtype] = c
        remaining -= c
    type_counts["explicit_tpo"] += remaining

    all_queries = []

    for qtype, count in type_counts.items():
        if count == 0:
            continue

        batch = min(count, 4)
        generated = 0
        while generated < count:
            this_batch = min(batch, count - generated)

            if qtype == "explicit_tpo":
                scen = sample_tpo_scenario(base_seed + generated)
                prompt = build_explicit_prompt(scen, this_batch)
                tpo_field = scen
            elif qtype == "implicit_tpo":
                scen = sample_tpo_scenario(base_seed + generated + 100)
                prompt = build_implicit_prompt(scen, this_batch)
                tpo_field = scen
            elif qtype == "visual_tpo":
                scen = sample_tpo_scenario(base_seed + generated + 200)
                prompt = build_visual_prompt(this_batch)
                tpo_field = scen
            else:  # neutral
                prompt = build_neutral_prompt(this_batch)
                tpo_field = None

            try:
                response = call_gpt5_mini(
                    prompt=prompt, system=QUERY_SYSTEM,
                    max_completion_tokens=GPT5_MINI["max_completion_tokens_default"],
                )
                queries = parse_json_list_response(response)
                if not queries:
                    print(f"      Parse fail for {qtype}")
                    generated += this_batch
                    continue

                for q in queries[:this_batch]:
                    all_queries.append({
                        "query_id": f"{user_id}_Q{len(all_queries)+1:03d}",
                        "user_id": user_id,
                        "domain": "fashion",
                        "query_type": qtype,
                        "query_text": q,
                        "tpo_scenario": tpo_field,
                    })
                generated += len(queries[:this_batch])
            except Exception as e:
                print(f"      Error in {qtype}: {e}")
                generated += this_batch

    return all_queries[:n_per_user]


def run_pipeline(profile_path: Path, output_path: Path,
                 n_per_user: int, force: bool = False):
    log_step(f"Query Generator — {n_per_user} queries/user (English)")

    profiles = load_jsonl(profile_path)
    print(f"  Loaded {len(profiles)} profiles")

    if output_path.exists() and not force:
        existing = load_jsonl(output_path)
        done_users = {q["user_id"] for q in existing}
        all_queries = existing
        todo = [p for p in profiles if p["user_id"] not in done_users]
        print(f"  Resuming: {len(done_users)} users already have queries")
    else:
        all_queries = []
        todo = profiles

    for i, prof in enumerate(todo):
        print(f"\n  [{i+1}/{len(todo)}] {prof['user_id']}")
        try:
            qs = generate_queries_for_user(prof, n_per_user)
            all_queries.extend(qs)
            print(f"    {len(qs)} queries generated")
            if (i + 1) % 5 == 0:
                save_jsonl(all_queries, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")

    save_jsonl(all_queries, output_path)
    print(f"\n  ✓ Saved {len(all_queries)} queries to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--output", type=Path,
                        default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--n_per_user", type=int,
                        default=PHASE1_CONFIG["n_queries_per_user"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.profile_path, args.output, args.n_per_user, args.force)


if __name__ == "__main__":
    main()
