"""Shared utilities: unified v0.1/v0.2 schema, leak-safe serialization, dataset."""
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['CONTINUE', 'ASK', 'BLOCK']  # index order matters: BLOCK==2 in metric reports
MODEL_ID = 'hfl/minirbt-h256'

# Fields that must NEVER be serialized into model input (label/answer leakage).
LEAKY_FIELDS = {'label', 'reason', 'risk_factors', 'source_type',
                'requires_human_review', 'sample_id', 'group_id'}

# Canonical normalized fields produced by normalize_row(); render() touches ONLY these.
CANONICAL_FIELDS = ['instruction', 'tool', 'target', 'scope', 'affected_count',
                    'authorization', 'data_classification', 'recoverable',
                    'backup_available', 'destination_trust']


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(data_dir, name):
    path = Path(data_dir) / f'{name}.jsonl'
    if not path.exists():
        raise FileNotFoundError(path)
    return load_jsonl(path)


def git_commit():
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                             capture_output=True, text=True, timeout=5, check=False)
        return out.stdout.strip() or None
    except Exception:
        return None


def is_v02(row):
    return 'user_instruction' in row and 'agent_tool' in row


def normalize_row(row):
    """Map a v0.2 (nested) or v0.1 (flat, legacy) row to canonical fields.

    v0.1 mapping is lossy (no classification/backup/trust info) and only kept
    so the legacy synthetic data remains runnable; v0.2 is the real input.
    """
    if is_v02(row):
        args = row.get('tool_arguments') or {}
        ctx = row.get('context') or {}
        norm = {
            'instruction': str(row['user_instruction']),
            'tool': str(row['agent_tool']),
            'target': str(args.get('target_type', 'unknown')),
            'scope': str(args.get('scope', 'unknown')),
            'affected_count': args.get('affected_count', 1),
            'authorization': str(ctx.get('authorization', 'unknown')),
            'data_classification': str(ctx.get('data_classification', 'unknown')),
            'recoverable': bool(ctx.get('recoverable', False)),
            'backup_available': bool(ctx.get('backup_available', False)),
            'destination_trust': str(ctx.get('destination_trust', 'unknown')),
        }
    else:
        norm = {
            'instruction': str(row['instruction']),
            'tool': str(row['tool']),
            'target': str(row['target']),
            'scope': str(row['scope']),
            'affected_count': 1,
            'authorization': 'approved' if row.get('authorized') else 'denied',
            'data_classification': 'unknown',
            'recoverable': bool(row.get('recoverable', False)),
            'backup_available': False,
            'destination_trust': 'unknown',
        }
    return norm


def _zh(v):
    if isinstance(v, bool):
        return '是' if v else '否'
    return str(v)


def render(norm):
    """Serialize canonical fields to the model input string.

    Built strictly from CANONICAL_FIELDS so label/reason/risk_factors can
    never leak into the input by construction.
    """
    return ('用户指令：' + _zh(norm['instruction']) +
            '；工具：' + _zh(norm['tool']) +
            '；目标：' + _zh(norm['target']) +
            '；范围：' + _zh(norm['scope']) +
            '；数量：' + _zh(norm['affected_count']) +
            '；授权：' + _zh(norm['authorization']) +
            '；数据分级：' + _zh(norm['data_classification']) +
            '；可恢复：' + _zh(norm['recoverable']) +
            '；有备份：' + _zh(norm['backup_available']) +
            '；目的地信任：' + _zh(norm['destination_trust']))


def render_row(row):
    return render(normalize_row(row))


class RiskDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=128, limit=None):
        self.rows = rows[:limit] if limit else rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        encoded = self.tokenizer(render_row(r), truncation=True,
                                 padding='max_length', max_length=self.max_length,
                                 return_tensors='pt')
        return {**{k: v.squeeze(0) for k, v in encoded.items()
                   if k in ('input_ids', 'attention_mask')},
                'labels': torch.tensor(LABELS.index(r['label']), dtype=torch.long)}


def get_tokenizer(model):
    return AutoTokenizer.from_pretrained(model)
