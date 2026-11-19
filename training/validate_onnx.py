"""ONNX Runtime parity validation against PyTorch (M1-08) + golden case export.

Usage: python training/validate_onnx.py --model-dir models/finetuned_e1 --samples 30
Deterministically stratified sample (seed=42, all three labels covered) from the
test split. Writes reports/onnx_parity.json and tests/golden_cases.jsonl (token
IDs + both logits) for the later HarmonyOS tokenizer/runtime parity spike (M2).
Pass: 100% label agreement and max abs logit diff <= 1e-2 (fp32 CPU backends).
"""
import argparse
import json
import platform
import random

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import LABELS, ROOT, git_commit, load_split, render_row


def stratified_rows(rows, n, seed):
    rng = random.Random(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(r['label'], []).append(r)
    per_class = max(n // len(by_label), 1)
    picked = []
    for label in sorted(by_label):
        pool = sorted(by_label[label], key=lambda r: r['sample_id'])
        picked.extend(rng.sample(pool, min(per_class, len(pool))))
    # fill up to n if any class was short
    if len(picked) < n:
        rest = [r for r in rows if r not in picked]
        picked.extend(rng.sample(rest, min(n - len(picked), len(rest))))
    return sorted(picked, key=lambda r: r['sample_id'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', default='models/risk_guard.onnx')
    p.add_argument('--model-dir', default='models/finetuned_last2',
                   help='checkpoint matching --onnx (see reports/ for selection)')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tolerance', type=float, default=1e-2)
    args = p.parse_args()

    onnx_path = ROOT / args.onnx
    model_dir = ROOT / args.model_dir
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    rows = stratified_rows(load_split('data', args.split), args.samples, args.seed)
    golden, diffs, agreement = [], [], 0
    with torch.inference_mode():
        for r in rows:
            text = render_row(r)
            enc = tok(text, max_length=args.seq_len, padding='max_length',
                      truncation=True, return_tensors='pt')
            pt = model(input_ids=enc['input_ids'],
                       attention_mask=enc['attention_mask']).logits[0].tolist()
            ort_out = session.run(['logits'], {k: enc[k].numpy()
                                               for k in ('input_ids', 'attention_mask')})[0][0].tolist()
            diff = float(np.max(np.abs(np.array(pt) - np.array(ort_out))))
            diffs.append(diff)
            agree = int(np.argmax(pt) == np.argmax(ort_out))
            agreement += agree
            golden.append({
                'sample_id': r['sample_id'], 'text': text,
                'input_ids': enc['input_ids'][0].tolist(),
                'attention_mask': enc['attention_mask'][0].tolist(),
                'gold_label': r['label'],
                'pt_label': LABELS[int(np.argmax(pt))],
                'ort_label': LABELS[int(np.argmax(ort_out))],
                'pt_logits': [round(v, 6) for v in pt],
                'ort_logits': [round(v, 6) for v in ort_out],
                'max_abs_diff': round(diff, 8),
            })

    report = {
        'onnx_path': str(onnx_path), 'model_dir': str(model_dir),
        'split': args.split, 'samples': len(rows),
        'label_agreement': agreement / len(rows),
        'max_abs_logit_diff': max(diffs),
        'mean_abs_logit_diff': round(float(np.mean(diffs)), 8),
        'tolerance': args.tolerance,
        'label_distribution': {l: sum(1 for g in golden if g['gold_label'] == l)
                               for l in LABELS},
        'env': {'onnxruntime': ort.__version__, 'torch': torch.__version__,
                'python': platform.python_version(), 'git_commit': git_commit()},
        'note': ('Numerical tolerance recorded for CPUExecutionProvider fp32; '
                 'other backends must record their own measured tolerance.'),
    }
    dest = ROOT / 'reports/onnx_parity.json'
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    golden_path = ROOT / 'tests/golden_cases.jsonl'
    golden_path.parent.mkdir(exist_ok=True)
    with open(golden_path, 'w', encoding='utf-8') as f:
        for g in golden:
            f.write(json.dumps(g, ensure_ascii=False) + '\n')

    print(json.dumps({k: report[k] for k in
                      ('samples', 'label_distribution', 'label_agreement',
                       'max_abs_logit_diff', 'mean_abs_logit_diff')},
                     ensure_ascii=False, indent=2))
    print('saved', dest, 'and', golden_path)
    if agreement < len(rows) or max(diffs) > args.tolerance:
        raise SystemExit(f'ONNX parity FAILED (agreement={agreement}/{len(rows)}, '
                         f'max_diff={max(diffs):.3e} > {args.tolerance})')
    print('ONNX PARITY OK')


if __name__ == '__main__':
    main()
