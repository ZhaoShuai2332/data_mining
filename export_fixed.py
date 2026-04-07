#!/usr/bin/env python3
"""Export FC2 or BN-folded ResNet-50 parameters into fixed-point integer text for MP-SPDZ."""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn

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
    old_conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        1,
        old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        dilation=old_conv1.dilation,
        groups=old_conv1.groups,
        bias=True,
        padding_mode=old_conv1.padding_mode,
    )
    with torch.no_grad():
        model.conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
        model.conv1.bias.zero_()

    def replace_conv_with_bias(root: nn.Module, module_name: str, conv: nn.Conv2d) -> None:
        new_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            if conv.bias is None:
                new_conv.bias.zero_()
            else:
                new_conv.bias.copy_(conv.bias)

        parent = root
        parts = module_name.split(".")
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        leaf = parts[-1]
        if leaf.isdigit():
            parent[int(leaf)] = new_conv
        else:
            setattr(parent, leaf, new_conv)

    # torchvision ResNet uses bias=False by default; after BN folding every conv
    # should have a bias term, so convert all conv layers to bias=True before load.
    for name, module in list(model.named_modules()):
        if name == "":
            continue
        if isinstance(module, nn.Conv2d) and module.bias is None:
            replace_conv_with_bias(model, name, module)

    model.fc = nn.Linear(model.fc.in_features, 10, bias=True)
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
    layouts = {}

    def conv_weight_ohwi(conv: nn.Conv2d) -> np.ndarray:
        # PyTorch conv layout is [O, I, H, W]. MP-SPDZ FixConv2d expects [O, H, W, I].
        return conv.weight.detach().cpu().numpy().astype(np.float64).transpose(0, 2, 3, 1)

    def conv_bias(conv: nn.Conv2d) -> np.ndarray:
        if conv.bias is None:
            return np.zeros(conv.out_channels, dtype=np.float64)
        return conv.bias.detach().cpu().numpy().astype(np.float64)

    def write_tensor(name: str, arr: np.ndarray, layout: str) -> None:
        shapes[name] = list(arr.shape)
        order.append(name)
        layouts[name] = layout
        write_sfix_real_values(f, arr, fractional_bits)

    with fixed_path.open('w') as f:
        write_tensor('conv1.weight', conv_weight_ohwi(model.conv1), 'OHWI')
        write_tensor('conv1.bias', conv_bias(model.conv1), 'O')

        layer_blocks = {
            'layer1': 3,
            'layer2': 4,
            'layer3': 6,
            'layer4': 3,
        }
        for layer_name, block_count in layer_blocks.items():
            layer = getattr(model, layer_name)
            for b in range(block_count):
                block = layer[b]
                for conv_name in ('conv1', 'conv2', 'conv3'):
                    conv = getattr(block, conv_name)
                    base = f'{layer_name}.{b}.{conv_name}'
                    write_tensor(f'{base}.weight', conv_weight_ohwi(conv), 'OHWI')
                    write_tensor(f'{base}.bias', conv_bias(conv), 'O')
                if block.downsample is not None:
                    ds_conv = block.downsample[0]
                    ds_base = f'{layer_name}.{b}.downsample.0'
                    write_tensor(f'{ds_base}.weight', conv_weight_ohwi(ds_conv), 'OHWI')
                    write_tensor(f'{ds_base}.bias', conv_bias(ds_conv), 'O')

        fc_weight = model.fc.weight.detach().cpu().numpy().astype(np.float64).transpose(1, 0)
        fc_bias = model.fc.bias.detach().cpu().numpy().astype(np.float64)
        # MP-SPDZ Dense expects W layout [in_dim, out_dim].
        write_tensor('fc.weight', fc_weight, 'IO')
        write_tensor('fc.bias', fc_bias, 'O')

    meta = {
        'fractional_bits': fractional_bits,
        'input_format': 'sfix_real_text',
        'order': order,
        'shapes': shapes,
        'layouts': layouts,
        'export_note': 'Conv weights are exported in OHWI; fc.weight is exported in IO.',
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
