#!/usr/bin/env python3
"""Convert an MNIST test image into fixed-point integers for MP-SPDZ input."""

import argparse
from pathlib import Path
import numpy as np
import os
import json

os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")
from torchvision import datasets, transforms  # type: ignore


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--fractional_bits', type=int)
    parser.add_argument('--fc2_config', type=str, default=str(DEFAULT_FC2_CONFIG_PATH))
    parser.add_argument('--outfile', required=True)
    parser.add_argument('--resize224', action='store_true')
    args = parser.parse_args()
    fc2_cfg = load_fc2_config(args.fc2_config)
    fractional_bits = args.fractional_bits
    if fractional_bits is None:
        fractional_bits = int(fc2_cfg.get('fractional_bits', 16))

    tf = []
    if args.resize224:
        tf.append(transforms.Resize(224))
    tf += [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose(tf))
    image, label = dataset[args.index]
    fixed = quantize_for_sfix_input(image.view(-1).numpy().astype(np.float64), fractional_bits)
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    with open(args.outfile, 'w') as f:
        for v in fixed:
            f.write(f'{float(v):.17g}\n')
    print(
        f'Wrote sample {args.index} label={label} to {args.outfile} '
        f'with fractional_bits={fractional_bits} (sfix real-text input)'
    )


if __name__ == '__main__':
    main()
