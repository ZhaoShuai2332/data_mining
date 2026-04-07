#!/usr/bin/env python3
"""Convert an MNIST test image into fixed-point integers for MP-SPDZ input."""

import argparse
from pathlib import Path
import numpy as np
import os

os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")
from torchvision import datasets, transforms  # type: ignore


def float_to_fixed(arr: np.ndarray, fractional_bits: int) -> np.ndarray:
    scale = 1 << fractional_bits
    return np.rint(arr * scale).astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--fractional_bits', type=int, default=16)
    parser.add_argument('--outfile', required=True)
    parser.add_argument('--resize224', action='store_true')
    args = parser.parse_args()

    tf = []
    if args.resize224:
        tf.append(transforms.Resize(224))
    tf += [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose(tf))
    image, label = dataset[args.index]
    fixed = float_to_fixed(image.view(-1).numpy().astype(np.float64), args.fractional_bits)
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    with open(args.outfile, 'w') as f:
        for v in fixed:
            f.write(f'{int(v)}\n')
    print(f'Wrote sample {args.index} label={label} to {args.outfile}')


if __name__ == '__main__':
    main()
