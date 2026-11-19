"""Evaluate a finetuned checkpoint on a held-out split (M1-06).

Usage: python training/evaluate.py --model-dir models/finetuned_e1 --split test
Reports macro-F1, per-class P/R/F1, confusion matrix, BLOCK recall/misses,
inference latency P50/P95, peak RSS. Saves reports/evaluation_<tag>_<split>.json.
"""
import argparse
import json
import platform
import time

import psutil
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import LABELS, ROOT, git_commit, load_split, render_row


def percentile(sorted_ms, q):
    if not sorted_ms:
        return None
    idx = min(int(len(sorted_ms) * q / 100), len(sorted_ms) - 1)
    return round(sorted_ms[idx], 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--tag', default=None, help='report filename tag (default: model-dir name)')
    args = p.parse_args()

    tag = args.tag or args.model_dir.rstrip('/\\').replace('\\', '/').split('/')[-1]
    model_dir = ROOT / args.model_dir
    process = psutil.Process()

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

    rows = load_split(args.data_dir, args.split)
    pred, gold, times = [], [], []
    with torch.inference_mode():
        for r in rows:
            x = tok(render_row(r), padding='max_length', truncation=True,
                    max_length=args.seq_len, return_tensors='pt')
            t = time.perf_counter()
            out = model(input_ids=x['input_ids'], attention_mask=x['attention_mask']).logits
            times.append((time.perf_counter() - t) * 1000)
            pred.append(int(out.argmax(-1)))
            gold.append(LABELS.index(r['label']))

    matrix = confusion_matrix(gold, pred, labels=list(range(3)))
    sorted_ms = sorted(times)
    report = {
        'model_dir': str(model_dir),
        'split': args.split,
        'count': len(rows),
        'labels': LABELS,
        'macro_f1': float(f1_score(gold, pred, average='macro', zero_division=0)),
        'classification_report': classification_report(
            gold, pred, labels=list(range(3)), target_names=LABELS,
            output_dict=True, zero_division=0),
        'confusion_matrix': matrix.tolist(),
        'block_recall': float(matrix[2, 2] / matrix[2].sum()) if matrix[2].sum() else None,
        'block_missed': int(matrix[2, :2].sum()),
        'latency_ms': {
            'mean': round(sum(times) / len(times), 2),
            'p50': percentile(sorted_ms, 50),
            'p95': percentile(sorted_ms, 95),
        },
        'peak_rss_mb': round(process.memory_info().peak_wset / 1e6, 1),
        'env': {'torch': torch.__version__, 'python': platform.python_version(),
                'git_commit': git_commit()},
        'note': ('Synthetic-only evaluation on held-out scenario groups; '
                 'NOT a real-world security benchmark. Test split was never used '
                 'for model selection.'),
    }
    dest = ROOT / f'reports/evaluation_{tag}_{args.split}.json'
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in
                      ('model_dir', 'split', 'count', 'macro_f1', 'block_recall',
                       'block_missed', 'confusion_matrix', 'latency_ms', 'peak_rss_mb')},
                     ensure_ascii=False, indent=2))
    print('saved', dest)


if __name__ == '__main__':
    main()
