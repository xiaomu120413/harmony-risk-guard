"""Export the finetuned MiniRBT classifier to fixed-shape ONNX (M1-07).

Usage: python training/export_onnx.py --model-dir models/finetuned_e1 --seq-len 128
Produces models/risk_guard.onnx ([1, seq] int64 inputs) + reports/onnx_export.json,
and runs a 3-sample ORT smoke check. Full parity is validate_onnx.py's job.
"""
import argparse
import json
import platform

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import ROOT, git_commit, load_split, render_row


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask).logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', default='models/finetuned_last2',
                   help='current winner checkpoint (see reports/ for selection)')
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    model_dir = ROOT / args.model_dir
    dest = ROOT / (args.out or 'models/risk_guard.onnx')
    dest.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    sample = tok('读取普通会议纪要', max_length=args.seq_len, padding='max_length',
                 truncation=True, return_tensors='pt')
    assert sample['input_ids'].shape == (1, args.seq_len)

    with torch.inference_mode():
        torch.onnx.export(Wrapper(model), (sample['input_ids'], sample['attention_mask']),
                          str(dest), input_names=['input_ids', 'attention_mask'],
                          output_names=['logits'], opset_version=args.opset, dynamo=False)
    # torch.onnx.export leaves the module in training mode (verified by
    # scripts/probe_export_diff.py: training=True after export, dropout then
    # skews in-process parity by ~2-4 logits). Re-assert eval before comparing.
    model.eval()
    onnx.checker.check_model(str(dest))

    # ORT smoke: 3 fixed rows must agree with PyTorch labels
    session = ort.InferenceSession(str(dest), providers=['CPUExecutionProvider'])
    rows = load_split('data', 'test')[:3]
    agree, diffs = 0, []
    with torch.inference_mode():
        for row in rows:
            enc = tok(render_row(row), max_length=args.seq_len,
                      padding='max_length', truncation=True, return_tensors='pt')
            pt = model(input_ids=enc['input_ids'],
                       attention_mask=enc['attention_mask']).logits.numpy()
            ort_out = session.run(['logits'], {k: enc[k].numpy()
                                               for k in ('input_ids', 'attention_mask')})[0]
            diffs.append(float(np.max(np.abs(pt - ort_out))))
            agree += int(np.argmax(pt) == np.argmax(ort_out))

    session_inputs = [{'name': i.name, 'shape': i.shape, 'type': i.type}
                      for i in session.get_inputs()]
    session_outputs = [{'name': o.name, 'shape': o.shape, 'type': o.type}
                       for o in session.get_outputs()]
    report = {
        'source_model_dir': str(model_dir),
        'onnx_path': str(dest),
        'onnx_bytes': dest.stat().st_size,
        'opset': args.opset,
        'inputs': session_inputs,
        'outputs': session_outputs,
        'fixed_shape': [1, args.seq_len],
        'smoke_check': {'samples': len(rows), 'label_agreement': agree / len(rows),
                        'max_abs_logit_diff': max(diffs)},
        'env': {'onnx': onnx.__version__, 'onnxruntime': ort.__version__,
                'torch': torch.__version__, 'python': platform.python_version(),
                'git_commit': git_commit()},
        'note': 'int64 token-id inputs; MindSpore Lite compatibility is NOT established by this export.',
    }
    out = ROOT / 'reports/onnx_export.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in
                      ('onnx_path', 'onnx_bytes', 'inputs', 'outputs', 'smoke_check')},
                     ensure_ascii=False, indent=2))
    if agree < len(rows):
        raise SystemExit('ORT smoke check failed: label disagreement')
    print('saved', dest)


if __name__ == '__main__':
    main()
