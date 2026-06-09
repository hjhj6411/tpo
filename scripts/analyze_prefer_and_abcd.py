#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def has_prefer_signal(opt: dict) -> bool:
    candidates = [
        opt,
        opt.get('attributes', {}) if isinstance(opt.get('attributes'), dict) else {},
        opt.get('metadata', {}) if isinstance(opt.get('metadata'), dict) else {},
    ]
    keys = {
        'prefer', 'preferred', 'preference', 'is_preferred', 'preference_hit',
        'preference_match', 'profile_match', 'fits_preference', 'matches_preference'
    }
    for d in candidates:
        for k, v in d.items():
            kl = str(k).lower()
            if kl in keys or 'prefer' in kl or 'profile' in kl:
                if isinstance(v, bool):
                    if v:
                        return True
                elif v not in (None, '', 0, '0', False, 'false', 'False', [], {}):
                    return True
    return False


def summarize_option_prefer(plans):
    option_counter = Counter()
    per_slot = defaultdict(Counter)
    detected_keys = Counter()
    total_plans = 0

    for plan in plans:
        opts = plan.get('options', {})
        if not isinstance(opts, dict):
            continue
        total_plans += 1
        for slot, opt in opts.items():
            hit = False
            if isinstance(opt, dict):
                candidates = [opt]
                if isinstance(opt.get('attributes'), dict):
                    candidates.append(opt['attributes'])
                if isinstance(opt.get('metadata'), dict):
                    candidates.append(opt['metadata'])
                for d in candidates:
                    for k, v in d.items():
                        kl = str(k).lower()
                        if 'prefer' in kl or 'profile' in kl:
                            detected_keys[kl] += 1
                hit = has_prefer_signal(opt)
            option_counter['total_options'] += 1
            option_counter['prefer_positive_options'] += int(hit)
            per_slot[slot]['total'] += 1
            per_slot[slot]['prefer_positive'] += int(hit)

    return total_plans, option_counter, per_slot, detected_keys


def summarize_predictions(results):
    display = Counter()
    original = Counter()
    invalid = 0
    total = 0
    by_axis = defaultdict(Counter)
    by_mode = defaultdict(Counter)

    for r in results:
        total += 1
        pred = r.get('predicted')
        pred_orig = r.get('predicted_original')
        axis = r.get('active_axis', 'unknown')
        mode = r.get('profile_mode', 'unknown')

        if pred in {'A', 'B', 'C', 'D'}:
            display[pred] += 1
            by_axis[axis][pred] += 1
            by_mode[mode][pred] += 1
        else:
            invalid += 1
            by_axis[axis]['INVALID'] += 1
            by_mode[mode]['INVALID'] += 1

        if pred_orig in {'A', 'B', 'C', 'D'}:
            original[pred_orig] += 1

    return total, invalid, display, original, by_axis, by_mode


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def print_counter(title, counter, total):
    print(f'\n=== {title} ===')
    for key in ['A', 'B', 'C', 'D']:
        val = counter.get(key, 0)
        print(f'{key}: {val:6d}  ({pct(val, total):5.1f}%)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plans', type=Path, required=True)
    ap.add_argument('--results', type=Path, required=True)
    args = ap.parse_args()

    plans = load_jsonl(args.plans)
    results = load_jsonl(args.results)

    total_plans, option_counter, per_slot, detected_keys = summarize_option_prefer(plans)
    total_results, invalid, display, original, by_axis, by_mode = summarize_predictions(results)

    print(f'Plans loaded:   {total_plans}')
    print(f'Results loaded: {total_results}')

    print('\n=== Preference-like signal in option plans ===')
    print(f"Positive options: {option_counter['prefer_positive_options']}/{option_counter['total_options']} "
          f"({pct(option_counter['prefer_positive_options'], option_counter['total_options']):.1f}%)")
    print('By original option slot:')
    for slot in ['A', 'B', 'C', 'D']:
        t = per_slot[slot]['total']
        h = per_slot[slot]['prefer_positive']
        print(f'  {slot}: {h}/{t} ({pct(h,t):.1f}%)')

    print('\nDetected preference-like keys (top 20):')
    if detected_keys:
        for k, v in detected_keys.most_common(20):
            print(f'  {k}: {v}')
    else:
        print('  None found')

    valid_display = sum(display.values())
    valid_original = sum(original.values())

    print_counter('Predicted DISPLAY labels', display, total_results)
    print(f'INVALID/None: {invalid:6d}  ({pct(invalid, total_results):5.1f}%)')

    print_counter('Predicted ORIGINAL option labels', original, total_results)
    print(f'Valid original-mapped predictions: {valid_original}/{total_results} ({pct(valid_original, total_results):.1f}%)')

    print('\n=== Predicted DISPLAY labels by active_axis ===')
    for axis in sorted(by_axis):
        cnt = by_axis[axis]
        n = sum(cnt.values())
        print(f'[{axis}] n={n}')
        for key in ['A', 'B', 'C', 'D', 'INVALID']:
            val = cnt.get(key, 0)
            print(f'  {key}: {val:6d}  ({pct(val, n):5.1f}%)')

    print('\n=== Predicted DISPLAY labels by profile_mode ===')
    for mode in sorted(by_mode):
        cnt = by_mode[mode]
        n = sum(cnt.values())
        print(f'[{mode}] n={n}')
        for key in ['A', 'B', 'C', 'D', 'INVALID']:
            val = cnt.get(key, 0)
            print(f'  {key}: {val:6d}  ({pct(val, n):5.1f}%)')


if __name__ == '__main__':
    main()
