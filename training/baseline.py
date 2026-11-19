"""E0 baseline: TF-IDF (word+char n-grams) + LogisticRegression on the same serialized input.

Usage: python training/baseline.py --data-dir data
Saves models/baseline_e0.joblib and reports/baseline_e0.json.
"""
import argparse
import json
import platform
import time

import joblib
import psutil
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from common import (LABELS, ROOT, git_commit, load_split, render_row)


def build_pipeline(seed):
    features = FeatureUnion([
        ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)),
        ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)),
    ])
    clf = LogisticRegression(max_iter=2000, class_weight='balanced',
                             random_state=seed)
    return Pipeline([('features', features), ('clf', clf)])


def evaluate_split(pipe, rows):
    texts = [render_row(r) for r in rows]
    gold = [LABELS.index(r['label']) for r in rows]
    pred = pipe.predict(texts).tolist()
    matrix = confusion_matrix(gold, pred, labels=list(range(3)))
    return {
        'count': len(rows),
        'macro_f1': float(f1_score(gold, pred, average='macro', zero_division=0)),
        'classification_report': classification_report(
            gold, pred, labels=list(range(3)), target_names=LABELS,
            output_dict=True, zero_division=0),
        'confusion_matrix': matrix.tolist(),
        'block_recall': float(matrix[2, 2] / matrix[2].sum()) if matrix[2].sum() else None,
        'block_missed': int(matrix[2, :2].sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    process = psutil.Process()
    start = time.perf_counter()
    train = load_split(args.data_dir, 'train')
    val = load_split(args.data_dir, 'val')
    test = load_split(args.data_dir, 'test')

    pipe = build_pipeline(args.seed)
    train_texts = [render_row(r) for r in train]
    train_gold = [LABELS.index(r['label']) for r in train]
    pipe.fit(train_texts, train_gold)

    out_model = ROOT / 'models/baseline_e0.joblib'
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': pipe, 'labels': LABELS}, out_model)

    feature_count = int(pipe.named_steps['features'].transform(train_texts[:1]).shape[1])
    report = {
        'experiment': 'E0_tfidf_logreg',
        'config': {'max_iter': 2000, 'class_weight': 'balanced',
                   'ngrams': 'word 1-2 + char_wb 2-4', 'seed': args.seed},
        'train': {'rows': len(train)},
        'val': evaluate_split(pipe, val),
        'test': evaluate_split(pipe, test),
        'feature_count': feature_count,
        'elapsed_seconds': round(time.perf_counter() - start, 2),
        'peak_rss_mb': round(process.memory_info().peak_wset / 1e6, 1),
        'versions': {'sklearn': sklearn.__version__, 'python': platform.python_version()},
        'git_commit': git_commit(),
        'note': ('TF-IDF sees the same serialized input as the neural models. '
                 'Groups are held out across splits, so performance on val/test '
                 'reflects generalization to unseen scenario groups, which is '
                 'expectedly harder than keyword matching. Synthetic data only.'),
    }
    out = ROOT / 'reports/baseline_e0.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('val', 'test')}, ensure_ascii=False, indent=2))
    print('saved', out_model, 'and', out)


if __name__ == '__main__':
    main()
