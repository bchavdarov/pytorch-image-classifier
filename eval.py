import argparse
import os

import torch
import torch.nn as nn

from dataset import build_dataloaders
from model import SimpleCNN


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CIFAR-10 classifier")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.1,
    )

    model = SimpleCNN(num_classes=10).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
