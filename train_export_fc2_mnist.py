#!/usr/bin/env python3
"""Train and export a 2-layer fully-connected MNIST model for MP-SPDZ inference."""

import argparse
import os
import numpy as np
import torch
from torch import nn, optim

# Some environments ship a torchvision build that raises on missing NMS ops.
# This line avoids importing the optional fake op registration path in many cases.
os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")
from torchvision import datasets, transforms  # type: ignore
from torch.utils.data import DataLoader


class FC2Net(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


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


def export_parameters(model: FC2Net, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(outdir, 'fc2_model.pth'))
    np.savez(
        os.path.join(outdir, 'fc2_params.npz'),
        W1=model.fc1.weight.detach().cpu().numpy().astype(np.float64),
        b1=model.fc1.bias.detach().cpu().numpy().astype(np.float64),
        W2=model.fc2.weight.detach().cpu().numpy().astype(np.float64),
        b2=model.fc2.bias.detach().cpu().numpy().astype(np.float64),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--outdir', type=str, default='models/fc2')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = FC2Net(hidden_dim=args.hidden).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f'Epoch {epoch}/{args.epochs} loss={loss:.4f} test_acc={acc * 100:.2f}%')

    export_parameters(model, args.outdir)
    print(f'Exported to {args.outdir}')


if __name__ == '__main__':
    main()
