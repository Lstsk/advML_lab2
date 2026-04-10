import argparse
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from baseline_model import UNet
from dataset import load_dataset


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the U-Net segmentation baseline.")
    parser.add_argument("--data-root", default=None, help="Path to the dataset root")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--output", default="results/baseline_model.pth")
    parser.add_argument("--val-split", default="val", help="Validation split")
    parser.add_argument("--loss-plot-output", default="results/baseline_train_vs_val_loss.png")
    return parser.parse_args()


def train():
    args = parse_args()
    device = get_device()
    image_size = (args.height, args.width)

    print(f"Using device: {device}")

    train_dataset, config = load_dataset(split="train", data_root=args.data_root, image_size=image_size)
    val_dataset, _ = load_dataset(split=args.val_split, data_root=args.data_root, image_size=image_size)

    # Shuffle train, keep val deterministic.
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = UNet(num_classes=config["num_classes"]).to(device)

    # Lower background weight a bit so the model does not over-predict class 0.
    class_weights = torch.ones(config["num_classes"], device=device)
    class_weights[0] = 0.1

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=config["ignore_index"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses: list[float] = []
    val_losses: list[float] = []

    # Standard train loop.
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

    # Validation loop (no gradients).
        model.eval()
        val_running_loss = 0.0
        val_total_pixels = 0
        with torch.no_grad():
            for val_images, val_masks in val_dataloader:
                val_images = val_images.to(device)
                val_masks = val_masks.to(device)

                val_logits = model(val_images)
                val_batch_loss = criterion(val_logits, val_masks)

                val_valid_pixels = (val_masks != config["ignore_index"]).sum().item()
                val_running_loss += val_batch_loss.item() * max(val_valid_pixels, 1)
                val_total_pixels += val_valid_pixels

        val_loss = val_running_loss / max(val_total_pixels, 1)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{args.epochs} - train loss: {epoch_loss:.4f} - val loss: {val_loss:.4f}")

        model.train()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.output)

    # Save a quick train-vs-val curve for reporting.
    os.makedirs(os.path.dirname(args.loss_plot_output) or ".", exist_ok=True)
    plt.figure(figsize=(8, 5))
    epochs = range(1, args.epochs + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.savefig(args.loss_plot_output, dpi=200)
    plt.close()

    print(f"Model saved to {args.output}")
    print(f"Loss curve saved to {args.loss_plot_output}")


if __name__ == "__main__":
    train()
