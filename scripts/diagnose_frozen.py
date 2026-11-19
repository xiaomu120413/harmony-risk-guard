"""Diagnose why head-only training is stuck: inspect frozen-encoder feature quality.

Compares CLS-token features vs (random-init) pooled features by fitting sklearn
LogisticRegression on each: if CLS features are separable but pooled are not,
the randomly-initialized frozen pooler is the bottleneck.
Usage: .venv python scripts/diagnose_frozen.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import LABELS, ROOT, load_split, render_row

torch.manual_seed(42)


def cls_features(model, tok, rows):
    out = []
    with torch.inference_mode():
        for r in rows:
            enc = tok(render_row(r), max_length=128, padding='max_length',
                      truncation=True, return_tensors='pt')
            res = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'],
                        output_hidden_states=True)
            out.append(res.hidden_states[-1][0, 0].numpy())
    return np.stack(out)


def pooled_features(model, tok, rows):
    out = []
    with torch.inference_mode():
        for r in rows:
            enc = tok(render_row(r), max_length=128, padding='max_length',
                      truncation=True, return_tensors='pt')
            res = model.bert(input_ids=enc['input_ids'],
                            attention_mask=enc['attention_mask'])
            out.append(res.pooler_output[0].numpy())
    return np.stack(out)


def probe(name, X_train, y_train, X_val, y_val):
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    print(f'{name}: train_F1={f1_score(y_train, clf.predict(X_train), average="macro"):.3f} '
          f'val_F1={f1_score(y_val, pred, average="macro"):.3f}')


def main():
    tok = AutoTokenizer.from_pretrained('hfl/minirbt-h256')
    model = AutoModelForSequenceClassification.from_pretrained('hfl/minirbt-h256',
                                                               num_labels=3).eval()
    train = load_split(ROOT / 'data', 'train')
    val = load_split(ROOT / 'data', 'val')
    y_train = [LABELS.index(r['label']) for r in train]
    y_val = [LABELS.index(r['label']) for r in val]

    cls_train = cls_features(model, tok, train)
    cls_val = cls_features(model, tok, val)
    probe('CLS-token features  ', cls_train, y_train, cls_val, y_val)

    pool_train = pooled_features(model, tok, train)
    pool_val = pooled_features(model, tok, val)
    probe('random frozen pooler', pool_train, y_train, pool_val, y_val)


if __name__ == '__main__':
    main()
