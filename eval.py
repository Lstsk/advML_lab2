import torch
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from model import UNet


def evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)

    # need to load dataset
    dataset = None 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for index, (images, _) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            print(f"sample {index}: prediction shape = {tuple(outputs.shape)}")


if __name__ == "__main__":
    evaluate()
