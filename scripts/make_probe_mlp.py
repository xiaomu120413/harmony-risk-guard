"""M2-A probe: fixed-weight 4-16-3 MLP -> ONNX, with golden input/output for on-device parity.

Usage: python scripts/make_probe_mlp.py [--out-dir harmony/entry/src/main/resources/rawfile]
Writes probe_mlp.onnx + probe_mlp_golden.json (input vector, expected logits/label).
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


class ProbeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='harmony/entry/src/main/resources/rawfile')
    args = p.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    model = ProbeMLP().eval()
    x = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    with torch.inference_mode():
        y = model(x)
    logits = [round(v, 6) for v in y[0].tolist()]
    golden = {
        'input_name': 'input',
        'input_shape': [1, 4],
        'input_values': x[0].tolist(),
        'output_name': 'logits',
        'expected_logits': logits,
        'expected_argmax': int(y.argmax(-1)),
        'note': 'PyTorch CPU reference for MindSpore Lite on-device parity (M2-A).',
    }

    onnx_path = out_dir / 'probe_mlp.onnx'
    torch.onnx.export(model, (x,), str(onnx_path),
                      input_names=['input'], output_names=['logits'],
                      opset_version=17, dynamo=False)
    (out_dir / 'probe_mlp_golden.json').write_text(
        json.dumps(golden, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved', onnx_path, onnx_path.stat().st_size, 'bytes')
    print(json.dumps(golden, ensure_ascii=False))


if __name__ == '__main__':
    main()
