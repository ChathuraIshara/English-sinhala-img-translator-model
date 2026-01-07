import argparse
import json
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model_utils import SimpleCNN, build_datasets, get_device, get_transforms


def split_dataset(dataset, train_ratio: float = 0.8) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
    return correct / max(total, 1)


def save_artifacts(model, idx_to_char, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.pth")
    label_map_path = os.path.join(out_dir, "label_map.json")

    torch.save({"model_state_dict": model.state_dict(), "idx_to_char": idx_to_char}, model_path)
    json.dump({"idx_to_char": idx_to_char}, open(label_map_path, "w"))
    print(f"Saved model to {model_path}")
    print(f"Saved label map to {label_map_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Sinhala character CNN and export artifacts")
    parser.add_argument("--dataset", default="dataset", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="artifacts", help="Output directory for model and label map")
    args = parser.parse_args()

    device = get_device()
    print("Using device:", device)

    dataset, _, idx_to_char = build_datasets(args.dataset)
    train_ds, test_ds = split_dataset(dataset)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = SimpleCNN(num_classes=len(idx_to_char)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {loss:.4f} - val_acc: {acc * 100:.2f}%")

    save_artifacts(model, idx_to_char, args.out)


if __name__ == "__main__":
    main()
