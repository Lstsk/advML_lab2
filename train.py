import torch
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from model import UNet


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    dataset = None
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        print(f"loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()
