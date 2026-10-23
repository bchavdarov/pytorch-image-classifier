import argparse
import copy
import os

import torch
import torch.nn as nn
from torch.optim import AdamW

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
    parser = argparse.ArgumentParser(description="Train a PyTorch image classifier")
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_loader, val_loader, _ = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.1,
    )

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

        train_loss = running_loss / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                },
                os.path.join(args.checkpoint_dir, "best_model.pt"),
            )

        scheduler.step()

    print(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": args.epochs,
            "val_acc": best_val_acc,
        },
        os.path.join(args.checkpoint_dir, "final_model.pt"),
    )


if __name__ == "__main__":
    main()
