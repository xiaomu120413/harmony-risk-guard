"""Train MiniRBT risk classifier (E1 head-only / E2 unfreeze last layers) on CPU.

Usage examples:
  python training/train.py --device cpu --mode head  --epochs 20 --threads 8
  python training/train.py --device cpu --mode last2 --epochs 20 --threads 8
  python training/train.py --mode head --smoke          # 20-sample pipeline check
Saves best-val-macro-F1 checkpoint to --out-dir and reports/training_<tag>.json.
"""
import argparse
import json
import platform
import time

import psutil
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (LABELS, MODEL_ID, ROOT, RiskDataset, git_commit, load_split,
                    seed_all)

MODE_DEFAULT_LR = {'head': 2e-3, 'last1': 2e-5, 'last2': 2e-5}
HEAD_DEFAULT_LR = 2e-3  # pooler+classifier head lr for last1/last2 modes


def class_weights(labels_list, mode):
    if mode != 'balanced':
        return None
    counts = torch.bincount(torch.tensor(labels_list, dtype=torch.long),
                            minlength=len(LABELS)).float()
    w = len(labels_list) / (len(LABELS) * counts.clamp(min=1))
    return w


def run_eval(model, loader, loss_weight):
    model.eval()
    pred, gold, total = [], [], 0.0
    with torch.inference_mode():
        for batch in loader:
            logits = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask']).logits
            loss = F.cross_entropy(logits, batch['labels'], weight=loss_weight)
            total += float(loss)
            pred.extend(logits.argmax(-1).tolist())
            gold.extend(batch['labels'].tolist())
    return f1_score(gold, pred, average='macro', zero_division=0), pred, gold


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=MODEL_ID)
    p.add_argument('--mode', choices=['head', 'last1', 'last2'], default='head')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=None,
                   help='default: 2e-3 head / 2e-5 unfrozen layers')
    p.add_argument('--lr-head', type=float, default=HEAD_DEFAULT_LR,
                   help='pooler+classifier lr when encoder layers are unfrozen (last1/last2)')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--class-weights', choices=['balanced', 'none'], default='balanced')
    p.add_argument('--early-stop-patience', type=int, default=2)
    p.add_argument('--device', default='cpu', choices=['cpu'])
    p.add_argument('--smoke', action='store_true',
                   help='20-sample/1-epoch pipeline check; writes to models/smoke only')
    args = p.parse_args()

    if args.lr is None:
        args.lr = MODE_DEFAULT_LR[args.mode]
    if args.smoke:
        args.epochs = 1
    out_dir = ROOT / (args.out_dir or ('models/smoke' if args.smoke
                                       else f'models/finetuned_{args.mode}'))
    tag = 'smoke' if args.smoke else f'{args.mode}'

    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    process = psutil.Process()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3,
        id2label={i: x for i, x in enumerate(LABELS)},
        label2id={x: i for i, x in enumerate(LABELS)})

    # Freeze the pretrained encoder (embeddings + transformer layers). The pooler
    # is NOT in the hfl/minirbt-h256 checkpoint (HF warns it is newly initialized),
    # so pooler+classifier form the trainable head in every mode; last1/last2
    # additionally unfreeze the final N encoder layers at a lower lr.
    base = model.base_model
    base.embeddings.requires_grad_(False)
    layers = getattr(getattr(base, 'encoder', None), 'layer', None)
    if layers is None:
        raise RuntimeError('Cannot locate encoder layers for this architecture')
    for layer in layers:
        layer.requires_grad_(False)
    unfrozen = 0
    if args.mode != 'head':
        n = int(args.mode.removeprefix('last'))
        for layer in layers[-n:]:
            layer.requires_grad_(True)
            unfrozen += sum(p.numel() for p in layer.parameters())

    train_rows = load_split(args.data_dir, 'train')
    val_rows = load_split(args.data_dir, 'val')
    train = RiskDataset(train_rows, tokenizer, args.max_length,
                        limit=20 if args.smoke else None)
    val = RiskDataset(val_rows, tokenizer, args.max_length,
                      limit=8 if args.smoke else None)

    weight = class_weights([LABELS.index(r['label']) for r in train_rows],
                           args.class_weights)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                         generator=torch.Generator().manual_seed(args.seed))
    valid = DataLoader(val, batch_size=args.batch_size)

    def is_head(name):
        return name.startswith('bert.pooler') or name.startswith('classifier')

    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and is_head(n)]
    layer_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not is_head(n)]
    groups = [{'params': head_params,
               'lr': args.lr if args.mode == 'head' else args.lr_head}]
    if layer_params:
        groups.append({'params': layer_params, 'lr': args.lr})
    optimizer = torch.optim.AdamW(groups)

    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_epoch, history = -1.0, 0, []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total_loss, steps = 0.0, 0
        t0 = time.perf_counter()
        for batch in loader:
            optimizer.zero_grad()
            logits = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask']).logits
            loss = F.cross_entropy(logits, batch['labels'], weight=weight)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
        score, _, _ = run_eval(model, valid, weight)
        record = {'epoch': epoch + 1, 'train_loss': total_loss / max(steps, 1),
                  'val_macro_f1': score,
                  'epoch_seconds': round(time.perf_counter() - t0, 2)}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if score > best:
            best, best_epoch = score, epoch + 1
            model.save_pretrained(out_dir, safe_serialization=True)
            tokenizer.save_pretrained(out_dir)
        elif epoch + 1 - best_epoch >= args.early_stop_patience:
            print(f'early stop at epoch {epoch + 1} (best {best:.4f} @ {best_epoch})')
            break

    report = {
        'experiment': f'E1_{args.mode}' if args.mode == 'head' else f'E2_{args.mode}',
        'smoke': args.smoke,
        'config': {**vars(args), 'lr': args.lr},
        'env': {'torch': torch.__version__, 'python': platform.python_version(),
                'cpu': platform.processor(), 'threads': args.threads,
                'git_commit': git_commit()},
        'train_rows': len(train), 'val_rows': len(val),
        'history': history,
        'best_val_macro_f1': best,
        'best_epoch': best_epoch,
        'unfrozen_layer_params': unfrozen,
        'trainable_parameters': sum(q.numel() for q in model.parameters() if q.requires_grad),
        'total_parameters': sum(q.numel() for q in model.parameters()),
        'elapsed_seconds': round(time.perf_counter() - start, 2),
        'peak_rss_mb': round(process.memory_info().peak_wset / 1e6, 1),
        'out_dir': str(out_dir),
    }
    dest = ROOT / 'reports'
    dest.mkdir(exist_ok=True)
    (dest / f'training_{tag}.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved best checkpoint:', out_dir, f'(best val macro-F1 {best:.4f} @ epoch {best_epoch})')


if __name__ == '__main__':
    main()
