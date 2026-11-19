"""Probe the export-smoke discrepancy: in-process post-export PyTorch forward
differs from ORT by ~4.6, but a fresh process agrees to 1e-5.

Steps:
  A: fresh model logits on 3 fixed rows
  B: same model instance logits AFTER torch.onnx.export (in-process)
  C: another fresh model loaded AFTER export, same rows
  D: ORT session on the exported file, same rows
Prints pairwise max diffs to isolate what export mutates.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import ROOT, load_split, render_row

MODEL_DIR = ROOT / 'models/finetuned_last2'
ONNX_PATH = ROOT / 'models/risk_guard_onnx_probe.onnx'


def rows_logits(model, tok, rows):
    out = []
    with torch.inference_mode():
        for r in rows:
            enc = tok(render_row(r), max_length=128, padding='max_length',
                      truncation=True, return_tensors='pt')
            out.append(model(input_ids=enc['input_ids'],
                             attention_mask=enc['attention_mask']).logits[0].numpy())
    return np.stack(out)


def maxdiff(x, y):
    return float(np.max(np.abs(x - y)))


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask).logits


def main():
    rows = load_split(ROOT / 'data', 'test')[:3]
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    model_a = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    A = rows_logits(model_a, tok, rows)

    sample = tok('读取普通会议纪要', max_length=128, padding='max_length',
                 truncation=True, return_tensors='pt')
    with torch.inference_mode():
        torch.onnx.export(Wrapper(model_a), (sample['input_ids'], sample['attention_mask']),
                          str(ONNX_PATH), input_names=['input_ids', 'attention_mask'],
                          output_names=['logits'], opset_version=17, dynamo=False)

    B = rows_logits(model_a, tok, rows)                     # same instance, post-export
    model_c = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    C = rows_logits(model_c, tok, rows)                     # fresh instance, post-export
    session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    D = []
    with torch.inference_mode():
        for r in rows:
            enc = tok(render_row(r), max_length=128, padding='max_length',
                      truncation=True, return_tensors='pt')
            D.append(session.run(['logits'], {k: enc[k].numpy()
                                              for k in ('input_ids', 'attention_mask')})[0][0])
    D = np.stack(D)

    print('A vs B (same instance pre/post export):', maxdiff(A, B))
    print('A vs C (fresh instance post export):  ', maxdiff(A, C))
    print('A vs D (fresh model vs ORT):         ', maxdiff(A, D))
    print('B vs D (post-export instance vs ORT): ', maxdiff(B, D))
    print('training flags:', model_a.training, model_c.training)
    ONNX_PATH.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
