import torch
from torch.utils.data import DataLoader

from dataset import load_cifake
from model import TransUNet


def evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransUNet(in_channels=3, num_classes=2).to(device)
    model.load_state_dict(torch.load("results/model.pth", map_location=device))

    test_dataset = load_cifake(split="test")
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    evaluate()
