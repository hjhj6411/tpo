"""
common/prompts.py — message builders for every experiment.

Objective : single source of truth for prompt construction so that experiments
            differ in EXACTLY one variable. Builders:
  * build_mcq_messages   : 4-choice MCQ. Knobs: display order (permute.py),
                           input condition, profile mode, block layout.
  * build_bc_messages    : 2-choice forced trade-off (canonical B vs C),
                           display side seeded 50:50, labels relabeled A/B.
  * build_dialogue_messages : profile -> multi-turn dialogue; each entry is a
                           user turn (sentence + image); per-entry implicit
                           with prob p; assistant turns fixed "Got it." so the
                           model cannot harvest attribute words from itself.
Input     : item dict (common/data.py schema), order (common/permute.py).
Output    : OpenAI-format messages. Images are markers
            {"type":"image_url","image_url":{"url":"__IMG__<src>"}} resolved
            to base64 by common/clients.py at send time (keeps prompts cheap
            to build/serialize and cache keys small).
Conditions: both | scenario_only | profile_only | none
Layouts   : text_first (context -> options) | img_first (options -> context).
Metrics   : n/a (infrastructure).
Run       : imported. Sample dump: every runner saves sample_prompt*.json.
"""
import copy

from .data import CANON

IMG_MARK = "__IMG__"

INSTR_HEADER = ("You are a fashion assistant. Choose the single outfit option "
                "that best satisfies BOTH the user's situation and the user's "
                "preferences, based on everything provided.")
INSTR_FOOTER = "Respond with only the letter of the best option."
INSTR_FOOTER_BC = "Respond with only the letter of the better option."
DIALOGUE_ACK = "Got it."


def _txt(s):
    return {"type": "text", "text": s}


def _img(src):
    return {"type": "image_url", "image_url": {"url": f"{IMG_MARK}{src}"}}


def context_blocks(item, condition="both", profile_mode="text"):
    """Scenario / profile blocks per input condition (e3 variable)."""
    blocks = []
    if condition in ("both", "scenario_only"):
        blocks.append(_txt(f"Situation: {item['scenario_text']}"))
    if condition in ("both", "profile_only"):
        if profile_mode in ("text", "both") and item.get("profile_text"):
            blocks.append(_txt(f"User preferences: {item['profile_text']}"))
        if profile_mode in ("image", "both") and item.get("profile_images"):
            blocks.append(_txt("Items the user loves wearing:"))
            blocks += [_img(u) for u in item["profile_images"]]
    return blocks


def option_blocks(item, order, letters):
    blocks = []
    for pos, canon_idx in enumerate(order):
        blocks.append(_txt(f"Option {letters[pos]}:"))
        blocks.append(_img(item["options"][CANON[canon_idx]]["image"]))
    return blocks


def build_mcq_messages(item, order, condition="both", profile_mode="text",
                       layout="text_first", letters=("A", "B", "C", "D")):
    ctx = context_blocks(item, condition, profile_mode)
    opts = option_blocks(item, order, letters)
    first, second = (ctx, opts) if layout == "text_first" else (opts, ctx)
    content = [_txt(INSTR_HEADER)] + first + second + [_txt(INSTR_FOOTER)]
    return [{"role": "user", "content": content}]


def build_bc_messages(item, b_first, condition="both", profile_mode="text",
                      layout="text_first"):
    """Canonical B (tpo_only) vs C (pref_only); display labels are A/B so the
    letter itself carries no semantics. b_first decides which side B sits on."""
    pair = ["B", "C"] if b_first else ["C", "B"]
    letters = ("A", "B")
    ctx = context_blocks(item, condition, profile_mode)
    opts = []
    for pos, canon in enumerate(pair):
        opts.append(_txt(f"Option {letters[pos]}:"))
        opts.append(_img(item["options"][canon]["image"]))
    first, second = (ctx, opts) if layout == "text_first" else (opts, ctx)
    content = [_txt(INSTR_HEADER)] + first + second + [_txt(INSTR_FOOTER_BC)]
    return [{"role": "user", "content": content}], pair


def build_dialogue_messages(item, p_implicit, rng, order,
                            layout="text_first", letters=("A", "B", "C", "D")):
    """Preference enters ONLY through dialogue turns (final MCQ turn carries
    scenario + options, no profile block). Returns (messages, realized_frac)."""
    msgs = []
    n_impl = 0
    entries = item["profile_entries"]
    for e in entries:
        implicit = rng.random() < p_implicit
        n_impl += int(implicit)
        sent = e["implicit"] if implicit else e["explicit"]
        msgs.append({"role": "user", "content": [_txt(sent), _img(e["image"])]})
        msgs.append({"role": "assistant", "content": [_txt(DIALOGUE_ACK)]})
    final = build_mcq_messages(item, order, condition="scenario_only",
                               layout=layout, letters=letters)[0]
    msgs.append(final)
    return msgs, (n_impl / max(len(entries), 1))


def serialize_for_dump(messages):
    """Human-readable copy for sample_prompt.json (markers kept visible)."""
    return copy.deepcopy(messages)
