import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_cifake
from baseline_model import UNet


def get_device() -> torch.device:
    """Pick the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train() -> None:
    device = get_device()
    print(f"Using device: {device}")

    model = UNet(in_channels=3, num_classes=2).to(device)

    train_dataset = load_cifake(split="train")
    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar with running stats
            pbar.set_postfix(
                loss=f"{running_loss / total:.4f}",
                acc=f"{100.0 * correct / total:.1f}%",
            )

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs} — loss: {epoch_loss:.4f}, acc: {epoch_acc:.2f}%\n")

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/baseline_model.pth")
    print("Model saved to results/baseline_model.pth")


if __name__ == "__main__":
    train()
