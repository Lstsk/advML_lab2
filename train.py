from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_dataset
from model import TransUNet


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TransUNet segmentation model.")
    parser.add_argument("--dataset", default="voc", help="Dataset name. Default: voc")
    parser.add_argument("--data-root", default=None, help="Path to the dataset root")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--output", default="results/model.pth")
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    device = get_device()
    image_size = (args.height, args.width)

    print(f"Using device: {device}")

    train_dataset, config = load_dataset(
        name=args.dataset,
        split="train",
        data_root=args.data_root,
        image_size=image_size,
    )
    if config["task"] != "segmentation":
        raise ValueError(
            f"Dataset '{args.dataset}' is a {config['task']} dataset, but this training script expects segmentation."
        )

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    model = TransUNet(num_classes=config["num_classes"], input_size=image_size).to(device)

    # Downweight the dominant background class (index 0) to prevent mode collapse
    class_weights = torch.ones(config["num_classes"], device=device)
    class_weights[0] = 0.1
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=config["ignore_index"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        total_pixels = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            valid_pixels = (masks != config["ignore_index"]).sum().item()
            running_loss += loss.item() * max(valid_pixels, 1)
            total_pixels += valid_pixels

            pbar.set_postfix(loss=f"{running_loss / max(total_pixels, 1):.4f}")

        epoch_loss = running_loss / max(total_pixels, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {epoch_loss:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    train()
