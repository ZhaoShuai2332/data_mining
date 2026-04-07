#!/usr/bin/env python3
"""Train and export an MNIST-adapted ResNet-50 for later BN folding and fixed-point export."""

import argparse
import os
import torch
from torch import nn, optim

os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")
from torchvision import datasets, transforms, models  # type: ignore
from torch.utils.data import DataLoader


def build_model(num_classes: int = 10) -> nn.Module:
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except Exception:
        model = models.resnet50(weights=None)
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        1,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        if old_conv.weight.shape[1] == 3:
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        else:
            new_conv.weight.copy_(old_conv.weight)
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.conv1 = new_conv
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--outdir', type=str, default='models/resnet50')
    parser.add_argument('--freeze_base', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model().to(device)
    if args.freeze_base:
        for name, p in model.named_parameters():
            if not name.startswith('fc'):
                p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f'Epoch {epoch}/{args.epochs} loss={loss:.4f} test_acc={acc * 100:.2f}%')

    os.makedirs(args.outdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.outdir, 'resnet50_model.pth'))
    print(f'Saved to {os.path.join(args.outdir, "resnet50_model.pth")}')


if __name__ == '__main__':
    main()
