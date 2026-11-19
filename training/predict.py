"""Single-shot risk prediction demo (CLI mirrors the task book section 8 target).

Usage:
  python training/predict.py --instruction "整理下载目录" --tool file.delete --target "用户文档"
  python training/predict.py --instruction "永久删除用户重要资料" --tool file.delete \
      --target 用户文档 --scope recursive --authorization denied --recoverable false
  python training/predict.py --instruction "..." --onnx models/risk_guard.onnx
Output: JSON with decision label, per-class scores, and the mandatory warning.
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import LABELS, ROOT, render


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes', '是'):
        return True
    if v.lower() in ('false', '0', 'no', '否'):
        return False
    raise argparse.ArgumentTypeError(f'bool expected, got {v!r}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--instruction', required=True)
    p.add_argument('--tool', required=True)
    p.add_argument('--target', default='user_document')
    p.add_argument('--scope', default='single', choices=['single', 'batch', 'recursive'])
    p.add_argument('--affected-count', type=int, default=1)
    p.add_argument('--authorization', default='ambiguous',
                   choices=['approved', 'ambiguous', 'denied'])
    p.add_argument('--classification', default='internal',
                   choices=['public', 'internal', 'personal', 'confidential', 'secret'])
    p.add_argument('--recoverable', type=str2bool, default=True)
    p.add_argument('--backup-available', type=str2bool, default=True)
    p.add_argument('--destination-trust', default='trusted')
    p.add_argument('--model-dir', default='models/finetuned_last2',
                   help='current winner checkpoint (see reports/ for selection)')
    p.add_argument('--onnx', default=None, help='run via ONNX Runtime instead of PyTorch')
    p.add_argument('--seq-len', type=int, default=128)
    args = p.parse_args()

    norm = {
        'instruction': args.instruction, 'tool': args.tool, 'target': args.target,
        'scope': args.scope, 'affected_count': args.affected_count,
        'authorization': args.authorization,
        'data_classification': args.classification,
        'recoverable': args.recoverable,
        'backup_available': args.backup_available,
        'destination_trust': args.destination_trust,
    }
    text = render(norm)
    model_dir = ROOT / args.model_dir
    tok = AutoTokenizer.from_pretrained(model_dir)
    enc = tok(text, max_length=args.seq_len, padding='max_length',
              truncation=True, return_tensors='pt')

    if args.onnx:
        import onnxruntime as ort
        session = ort.InferenceSession(str(ROOT / args.onnx),
                                       providers=['CPUExecutionProvider'])
        logits = session.run(['logits'], {k: enc[k].numpy()
                                          for k in ('input_ids', 'attention_mask')})[0][0]
        backend = f'onnxruntime:{ort.__version__}'
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        with torch.inference_mode():
            logits = model(input_ids=enc['input_ids'],
                           attention_mask=enc['attention_mask']).logits[0].numpy()
        backend = f'torch:{torch.__version__}'

    prob = np.exp(logits - logits.max())
    prob = (prob / prob.sum()).tolist()
    result = {
        'decision': LABELS[int(np.argmax(prob))],
        'scores': {label: round(float(prob[i]), 5) for i, label in enumerate(LABELS)},
        'backend': backend,
        'input': norm,
        'serialized': text,
        'warning': ('Experimental model on synthetic data; do not authorize destructive '
                    'actions based on this prediction. Independent policy checks must '
                    'run regardless and fail closed when uncertain.'),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
