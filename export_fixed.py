#!/usr/bin/env python3
"""Export FC2 or BN-folded ResNet-50 parameters into fixed-point integer text for MP-SPDZ."""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")
from torchvision import models  # type: ignore


def float_to_fixed(arr: np.ndarray, fractional_bits: int) -> np.ndarray:
    scale = 1 << fractional_bits
    return np.rint(arr * scale).astype(np.int64)


def export_fc2(npz_path: str, outdir: str, fractional_bits: int) -> None:
    data = np.load(npz_path)
    params = {
        'W1': data['W1'],
        'b1': data['b1'],
        'W2': data['W2'],
        'b2': data['b2'],
    }
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fixed_path = Path(outdir) / 'fixed_params.txt'
    meta_path = Path(outdir) / 'meta.json'
    with fixed_path.open('w') as f:
        for name in ['W1', 'b1', 'W2', 'b2']:
            fixed = float_to_fixed(params[name].flatten().astype(np.float64), fractional_bits)
            for v in fixed:
                f.write(f'{int(v)}\n')
    meta = {
        'fractional_bits': fractional_bits,
        'param_order': ['W1', 'b1', 'W2', 'b2'],
        'shapes': {k: list(v.shape) for k, v in params.items()},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Wrote {fixed_path} and {meta_path}')


def build_resnet50_for_load() -> torch.nn.Module:
    model = models.resnet50(weights=None)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    return model


def export_resnet50(pth_path: str, outdir: str, fractional_bits: int) -> None:
    model = build_resnet50_for_load()
    state = torch.load(pth_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fixed_path = Path(outdir) / 'fixed_params.txt'
    meta_path = Path(outdir) / 'meta.json'
    shapes = {}
    order = []
    with fixed_path.open('w') as f:
        for name, tensor in model.state_dict().items():
            if name.endswith('.weight') or name.endswith('.bias'):
                arr = tensor.cpu().numpy().astype(np.float64)
                shapes[name] = list(arr.shape)
                order.append(name)
                fixed = float_to_fixed(arr.flatten(), fractional_bits)
                for v in fixed:
                    f.write(f'{int(v)}\n')
    meta = {
        'fractional_bits': fractional_bits,
        'order': order,
        'shapes': shapes,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Wrote {fixed_path} and {meta_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=['fc2', 'resnet50'], required=True)
    parser.add_argument('--npz')
    parser.add_argument('--pth')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--fractional_bits', type=int, default=16)
    args = parser.parse_args()

    if args.arch == 'fc2':
        if not args.npz:
            raise ValueError('--npz is required for fc2')
        export_fc2(args.npz, args.outdir, args.fractional_bits)
    else:
        if not args.pth:
            raise ValueError('--pth is required for resnet50')
        export_resnet50(args.pth, args.outdir, args.fractional_bits)


if __name__ == '__main__':
    main()
