"""CIFAKE dataset loader for binary classification (Real vs AI-Generated)."""

import os

import kagglehub
from torchvision import datasets, transforms


# Default transforms: resize to 32x32, normalize to ImageNet stats
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def download_cifake() -> str:
    """Download the CIFAKE dataset via kagglehub and return the root path."""
    path = kagglehub.dataset_download(
        "birdy654/cifake-real-and-ai-generated-synthetic-images"
    )
    return path


def load_cifake(
    split: str = "train",
    transform: transforms.Compose | None = None,
    data_root: str | None = None,
) -> datasets.ImageFolder:
    """Load the CIFAKE dataset for a given split.

    The dataset is organized as:
        <root>/train/REAL/  — real CIFAR-10 images
        <root>/train/FAKE/  — AI-generated images
        <root>/test/REAL/
        <root>/test/FAKE/

    ImageFolder assigns labels alphabetically: FAKE=0, REAL=1.

    Args:
        split: "train" or "test".
        transform: optional torchvision transforms. Uses DEFAULT_TRANSFORM if None.
        data_root: path to the dataset root. If None, downloads via kagglehub.

    Returns:
        A torchvision ImageFolder dataset.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    if data_root is None:
        data_root = download_cifake()

    if transform is None:
        transform = DEFAULT_TRANSFORM

    split_dir = os.path.join(data_root, split)

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Expected directory '{split_dir}' not found. "
            f"Check that data_root='{data_root}' contains train/ and test/ folders."
        )

    dataset = datasets.ImageFolder(root=split_dir, transform=transform)
    print(f"Loaded CIFAKE {split} split: {len(dataset)} images, classes={dataset.classes}")
    return dataset
