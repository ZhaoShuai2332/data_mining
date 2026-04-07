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


DEFAULT_FC2_CONFIG_PATH = Path(__file__).resolve().parent / 'config' / 'fc2_config.json'


def load_fc2_config(config_path: str | None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_FC2_CONFIG_PATH
    default_cfg = {
        'fractional_bits': 16,
        'input_dim': 28 * 28,
        'hidden_dim': 128,
        'output_dim': 10,
    }
    if not path.exists():
        return default_cfg
    loaded = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(loaded, dict):
        raise ValueError(f'Invalid fc2 config format in {path}')
    merged = dict(default_cfg)
    merged.update(loaded)
    return merged


def float_to_fixed(arr: np.ndarray, fractional_bits: int) -> np.ndarray:
    scale = 1 << fractional_bits
    return np.rint(arr * scale).astype(np.int64)


def quantize_for_sfix_input(arr: np.ndarray, fractional_bits: int) -> np.ndarray:
    scale = 1 << fractional_bits
    return np.rint(arr * scale).astype(np.float64) / scale


def write_sfix_real_values(f, arr: np.ndarray, fractional_bits: int) -> None:
    """Write quantized real numbers for sfix.input_from().

    MP-SPDZ `sfix.get_input_from()` expects real-number text and performs
    internal scaling by `2^f`. Therefore, we write real values quantized to
    the target fractional precision instead of pre-scaled integers.
    """
    quantized = quantize_for_sfix_input(arr.astype(np.float64), fractional_bits).reshape(-1)
    for value in quantized:
        f.write(f'{float(value):.17g}\n')


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
            write_sfix_real_values(f, params[name], fractional_bits)
    meta = {
        'fractional_bits': fractional_bits,
        'input_format': 'sfix_real_text',
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
                write_sfix_real_values(f, arr, fractional_bits)
    meta = {
        'fractional_bits': fractional_bits,
        'input_format': 'sfix_real_text',
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
    parser.add_argument('--fractional_bits', type=int)
    parser.add_argument('--fc2_config', type=str, default=str(DEFAULT_FC2_CONFIG_PATH))
    args = parser.parse_args()
    fc2_cfg = load_fc2_config(args.fc2_config)
    fractional_bits = args.fractional_bits
    if fractional_bits is None:
        fractional_bits = int(fc2_cfg.get('fractional_bits', 16))

    if args.arch == 'fc2':
        if not args.npz:
            raise ValueError('--npz is required for fc2')
        export_fc2(args.npz, args.outdir, fractional_bits)
    else:
        if not args.pth:
            raise ValueError('--pth is required for resnet50')
        export_resnet50(args.pth, args.outdir, fractional_bits)


if __name__ == '__main__':
    main()
