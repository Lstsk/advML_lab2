from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_model import UNet
from dataset import load_dataset


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the U-Net segmentation baseline.")
    parser.add_argument("--dataset", default="voc", help="Dataset name. Default: voc")
    parser.add_argument("--data-root", default=None, help="Path to the dataset root")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--checkpoint", default="results/baseline_model.pth")
    parser.add_argument("--split", default="val", help="Dataset split. Default: val")
    parser.add_argument("--plot-output", default="results/baseline_confusion_matrix.png")
    return parser.parse_args()


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    predictions: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    valid = masks != ignore_index
    predictions = predictions[valid]
    masks = masks[valid]

    indices = masks * num_classes + predictions
    batch_confusion = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion_matrix += batch_confusion.reshape(num_classes, num_classes)
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix: torch.Tensor, class_names: list[str], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    row_sums = confusion_matrix.sum(dim=1, keepdim=True).clamp_min(1)
    normalized = (confusion_matrix.float() / row_sums).cpu().numpy()

    size = max(8, len(class_names) * 0.45)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Normalized Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlim(-0.5, len(class_names) - 0.5)
    ax.set_ylim(len(class_names) - 0.5, -0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate() -> None:
    args = parse_args()
    device = get_device()
    image_size = (args.height, args.width)

    print(f"Using device: {device}")

    dataset, config = load_dataset(
        name=args.dataset,
        split=args.split,
        data_root=args.data_root,
        image_size=image_size,
    )
    if config["task"] != "segmentation":
        raise ValueError(
            f"Dataset '{args.dataset}' is a {config['task']} dataset, but this evaluation script expects segmentation."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNet(num_classes=config["num_classes"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    confusion_matrix = torch.zeros(
        (config["num_classes"], config["num_classes"]),
        dtype=torch.int64,
    )

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            confusion_matrix = update_confusion_matrix(
                confusion_matrix,
                predictions.cpu(),
                masks.cpu(),
                config["num_classes"],
                config["ignore_index"],
            )

    intersection = confusion_matrix.diag().float()
    union = confusion_matrix.sum(dim=1).float() + confusion_matrix.sum(dim=0).float() - intersection
    valid_classes = union > 0
    iou = intersection / union.clamp_min(1.0)
    mean_iou = iou[valid_classes].mean().item()
    pixel_accuracy = intersection.sum().item() / confusion_matrix.sum().clamp_min(1).item()

    print(f"Pixel Accuracy: {100.0 * pixel_accuracy:.2f}%")
    print(f"Mean IoU: {100.0 * mean_iou:.2f}%")

    plot_confusion_matrix(confusion_matrix, config["class_names"], args.plot_output)
    print(f"Confusion matrix saved to {args.plot_output}")


if __name__ == "__main__":
    evaluate()
