#!/usr/bin/env python3
"""Fold BatchNorm layers into convolutions for a ResNet-50 style model."""

import argparse
import os
import torch
from torch import nn

os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")
from torchvision import models  # type: ignore


def fold_conv_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    with torch.no_grad():
        w = conv.weight.data.clone()
        if conv.bias is None:
            b = torch.zeros(w.size(0), device=w.device, dtype=w.dtype)
        else:
            b = conv.bias.data.clone()
        gamma = bn.weight.data
        beta = bn.bias.data
        mean = bn.running_mean
        var = bn.running_var
        std = torch.sqrt(var + bn.eps)
        scale = gamma / std
        w.mul_(scale.view(-1, 1, 1, 1))
        b = (b - mean) * scale + beta
        conv.weight.data.copy_(w)
        conv.bias = nn.Parameter(b)


def build_resnet50_for_load() -> nn.Module:
    model = models.resnet50(weights=None)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        1,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def fold_model(model: nn.Module) -> nn.Module:
    fold_conv_bn_pair(model.conv1, model.bn1)
    model.bn1 = nn.Identity()
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for block in layer:
            fold_conv_bn_pair(block.conv1, block.bn1)
            block.bn1 = nn.Identity()
            fold_conv_bn_pair(block.conv2, block.bn2)
            block.bn2 = nn.Identity()
            fold_conv_bn_pair(block.conv3, block.bn3)
            block.bn3 = nn.Identity()
            if block.downsample is not None and len(block.downsample) >= 2:
                if isinstance(block.downsample[0], nn.Conv2d) and isinstance(block.downsample[1], nn.BatchNorm2d):
                    fold_conv_bn_pair(block.downsample[0], block.downsample[1])
                    block.downsample[1] = nn.Identity()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    model = build_resnet50_for_load()
    state = torch.load(args.input, map_location='cpu')
    model.load_state_dict(state)
    model = fold_model(model)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f'Saved fused model to {args.output}')


if __name__ == '__main__':
    main()
