import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from baseline_model import UNet
from dataset import IMAGENET_MEAN, IMAGENET_STD, load_dataset


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the U-Net segmentation baseline.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--checkpoint", default="results/baseline_model.pth")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default="results/eval_baseline")
    parser.add_argument("--viz-samples", type=int, default=3) # this one to control how many sample visualizations to save
    return parser.parse_args()


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    return image * std + mean


def save_prediction_visualization(
    image: torch.Tensor,
    mask: torch.Tensor,
    prediction: torch.Tensor,
    class_names: list[str],
    ignore_index: int,
    output_path: str,
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmap = plt.get_cmap("tab20", len(class_names)).copy()
    cmap.set_bad(color=(0.8, 0.8, 0.8, 1.0))

    image_np = denormalize_image(image).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    mask_np = mask.cpu().numpy()
    pred_np = prediction.cpu().numpy()

    masked_mask = np.ma.masked_where(mask_np == ignore_index, mask_np)

    def draw_panel(ax, panel_image, title: str, overlay: np.ndarray | None = None):
        ax.imshow(panel_image)
        if overlay is not None:
            ax.imshow(
                overlay,
                cmap=cmap,
                vmin=0,
                vmax=len(class_names) - 1,
                alpha=0.55,
                interpolation="nearest",
            )
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    draw_panel(axes[0], image_np, "Input")
    draw_panel(axes[1], image_np, "Ground Truth", masked_mask)
    draw_panel(axes[2], image_np, "Prediction", pred_np)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    predictions: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    # Ignore unlabeled pixels before counting class pairs.
    valid = masks != ignore_index
    predictions = predictions[valid]
    masks = masks[valid]

    indices = masks * num_classes + predictions
    batch_confusion = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion_matrix += batch_confusion.reshape(num_classes, num_classes)
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix: torch.Tensor, class_names: list[str], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Row-normalize so each true class sums to 1.0 in the plot.
    row_sums = confusion_matrix.sum(dim=1, keepdim=True).clamp_min(1)
    normalized = (confusion_matrix.float() / row_sums).cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, shrink=0.85, pad=0.02)

    ax.set_title("Normalized Confusion Matrix", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlim(-0.5, len(class_names) - 0.5)
    ax.set_ylim(len(class_names) - 0.5, -0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate():
    args = parse_args()
    device = get_device()
    image_size = (args.height, args.width)
    plot_output = os.path.join(args.output_dir, "confusion_matrix.png")

    print(f"Using device: {device}")

    dataset, config = load_dataset(split=args.split, data_root=args.data_root, image_size=image_size)

    # Keep evaluation deterministic.
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

    saved_visualizations = 0

    # Build confusion matrix over the whole split.
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

            if saved_visualizations < args.viz_samples:
                remaining = args.viz_samples - saved_visualizations
                batch_images = images.cpu()[:remaining]
                batch_masks = masks.cpu()[:remaining]
                batch_predictions = predictions.cpu()[:remaining]
                for local_index in range(batch_images.shape[0]):
                    sample_index = saved_visualizations + local_index
                    output_path = os.path.join(args.output_dir, f"sample_{sample_index:03d}.png")
                    save_prediction_visualization(
                        batch_images[local_index],
                        batch_masks[local_index],
                        batch_predictions[local_index],
                        config["class_names"],
                        config["ignore_index"],
                        output_path,
                    )
                saved_visualizations += batch_images.shape[0]

    intersection = confusion_matrix.diag().float()
    union = confusion_matrix.sum(dim=1).float() + confusion_matrix.sum(dim=0).float() - intersection
    valid_classes = union > 0
    iou = intersection / union.clamp_min(1.0)
    mean_iou = iou[valid_classes].mean().item()
    pixel_accuracy = intersection.sum().item() / confusion_matrix.sum().clamp_min(1).item()

    print(f"Pixel Accuracy: {100.0 * pixel_accuracy:.2f}%")
    print(f"Mean IoU: {100.0 * mean_iou:.2f}%")

    # Save the matrix as a figure for reports.
    plot_confusion_matrix(confusion_matrix, config["class_names"], plot_output)
    print(f"Confusion matrix saved to {plot_output}")
    if args.viz_samples > 0:
        print(f"Prediction visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    evaluate()
