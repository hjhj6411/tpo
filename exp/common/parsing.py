"""
common/parsing.py — answer extraction from model text.

Objective : uniform parser for guided AND free-form outputs so closed models
            (no guided decoding) are scored by the same rules as open ones.
            Parse FAILURES are surfaced (method=None) and reported per model —
            never silently dropped (thinking/free-form fairness control).
Input     : raw completion text + allowed display letters.
Output    : (letter|None, method) where method in
            {exact, prefix, answer_kw, standalone, None}.
Metrics   : parse failure rate (aggregated by runners).
Run       : python -m common.parsing  (self-demo)
"""
import re


def extract_choice(text, letters):
    if not text:
        return None, None
    t = text.strip()
    ls = "".join(letters)
    if t.upper() in letters:
        return t.upper(), "exact"
    m = re.match(rf"^\s*\(?([{ls}])\)?[\s.):,]", t.upper() + " ")
    if m:
        return m.group(1), "prefix"
    m = re.search(rf"answer\s*(?:is|:)?\s*\(?([{ls}])\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper(), "answer_kw"
    hits = re.findall(rf"(?<![A-Za-z])([{ls}])(?![A-Za-z])", t.upper())
    if hits:
        return hits[-1], "standalone"
    return None, None


if __name__ == "__main__":
    for s in ["B", " (c) ", "The answer is D.", "I'd go with option A here",
              "blah"]:
        print(repr(s), "->", extract_choice(s, ("A", "B", "C", "D")))
