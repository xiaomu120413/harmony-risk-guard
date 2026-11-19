"""Validate dataset splits (M1-02 / P0): schema, enums, group leakage, dedup, quality report.

Usage: python training/validate_dataset.py --data-dir data
Exit code 1 on any error-level finding; report saved to reports/data_quality.json.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from common import (LABELS, LEAKY_FIELDS, ROOT, git_commit, load_split,
                    normalize_row, render_row)

V02_ENUMS = {
    'label': LABELS,
    'scenario': ['file_operation', 'permission_change', 'data_exfiltration'],
    'context.authorization': ['approved', 'ambiguous', 'denied'],
    'tool_arguments.scope': ['single', 'batch', 'recursive'],
    'context.data_classification': ['public', 'internal', 'personal',
                                   'confidential', 'secret'],
}
V02_BOOL = ['context.recoverable', 'context.backup_available']
V02_REQUIRED = ['sample_id', 'group_id', 'scenario', 'user_instruction',
                'agent_tool', 'tool_arguments', 'context', 'label',
                'source_type', 'requires_human_review']


def get_nested(row, dotted):
    cur = row
    for part in dotted.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = p.parse_args()

    data_dir = (ROOT / args.data_dir).resolve()
    errors, warnings = [], []
    report = {'data_dir': str(data_dir), 'splits': {}, 'git_commit': git_commit()}

    all_rows, all_ids = {}, []
    for split in args.splits:
        try:
            rows = load_split(data_dir, split)
        except FileNotFoundError as e:
            errors.append(f'{split}: missing file {e}')
            continue
        all_rows[split] = rows
        all_ids.extend((split, r.get('sample_id')) for r in rows)

        s_err, s_warn = [], []
        groups, texts, ids = set(), {}, set()
        label_dist = Counter()
        for i, r in enumerate(rows):
            missing = [f for f in V02_REQUIRED if f not in r]
            if missing:
                s_err.append(f'row {i}: missing fields {missing}')
                continue
            if not isinstance(r['tool_arguments'], dict) or not isinstance(r['context'], dict):
                s_err.append(f"row {i} ({r['sample_id']}): tool_arguments/context must be objects")
                continue
            for dotted, allowed in V02_ENUMS.items():
                v, ok = get_nested(r, dotted)
                if not ok or v not in allowed:
                    s_err.append(f"row {i} ({r['sample_id']}): {dotted}={v!r} not in {allowed}")
            for dotted in V02_BOOL:
                v, ok = get_nested(r, dotted)
                if not ok or not isinstance(v, bool):
                    s_err.append(f"row {i} ({r['sample_id']}): {dotted}={v!r} not bool")
            groups.add(r['group_id'])
            label_dist[r['label']] += 1
            ids.add(r['sample_id'])
            # identical serialized input, different label => direct label conflict
            key = render_row(r)
            if key in texts and texts[key] != r['label']:
                s_err.append(f"row {i} ({r['sample_id']}): duplicate input text with "
                             f"conflicting label ({texts[key]} vs {r['label']})")
            else:
                texts[key] = r['label']
            # leakage guard: input text must not echo reason or label words
            if 'reason' in r and str(r['reason']) and str(r['reason']) in key:
                s_err.append(f"row {i} ({r['sample_id']}): reason text leaked into input")
            if r['label'] in key:
                s_err.append(f"row {i} ({r['sample_id']}): label token leaked into input")

        report['splits'][split] = {
            'rows': len(rows), 'groups': len(groups), 'unique_ids': len(ids),
            'label_dist': dict(label_dist),
            'max_instruction_len': max((len(r['user_instruction']) for r in rows), default=0),
            'errors': s_err, 'warnings': s_warn,
        }
        errors.extend(f'{split}: {m}' for m in s_err)
        warnings.extend(f'{split}: {m}' for m in s_warn)
        # normalization must not touch leaky fields
        for r in rows[:5]:
            norm = normalize_row(r)
            assert set(norm) == set(['instruction', 'tool', 'target', 'scope',
                                     'affected_count', 'authorization',
                                     'data_classification', 'recoverable',
                                     'backup_available', 'destination_trust']), \
                'normalize_row produced unexpected fields'

    # cross-split checks
    if len(all_rows) >= 2:
        for a in args.splits:
            for b in args.splits:
                if a >= b or a not in all_rows or b not in all_rows:
                    continue
                ga = {r['group_id'] for r in all_rows[a]}
                gb = {r['group_id'] for r in all_rows[b]}
                overlap = ga & gb
                if overlap:
                    errors.append(f'group_id overlap {a}&{b}: {sorted(overlap)}')
        id_counts = Counter(sid for _, sid in all_ids)
        dupes = [k for k, v in id_counts.items() if v > 1]
        if dupes:
            errors.append(f'duplicate sample_ids across splits: {dupes}')
        # same group_id must carry a single label
        group_labels = {}
        for split, rows in all_rows.items():
            for r in rows:
                group_labels.setdefault(r['group_id'], set()).add(r['label'])
        mixed = {g: sorted(v) for g, v in group_labels.items() if len(v) > 1}
        if mixed:
            warnings.append(f'mixed-label groups (kept: split holds out whole groups): {mixed}')

    report['cross_split'] = {'group_leakage': not any('group_id overlap' in e for e in errors)}
    report['errors'] = errors
    report['warnings'] = warnings
    report['leaky_fields_never_used'] = sorted(LEAKY_FIELDS)
    report['note'] = ('Synthetic rule-based data; evaluation on it is NOT a real-world '
                      'security benchmark. reason field excluded from model input.')

    dest = ROOT / 'reports'
    dest.mkdir(exist_ok=True)
    out = dest / 'data_quality.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({k: report[k] for k in ('splits', 'cross_split')},
                     ensure_ascii=False, indent=2))
    print(f'errors: {len(errors)}  warnings: {len(warnings)}')
    for m in errors:
        print('ERROR:', m)
    for m in warnings:
        print('WARN:', m)
    print('report saved to', out)
    if errors:
        raise SystemExit(1)
    print('VALIDATION OK')


if __name__ == '__main__':
    main()
