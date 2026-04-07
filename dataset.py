"""Dataset loading utilities.

This project now treats semantic segmentation as the primary task.
`load_dataset()` is the single entrypoint used by training and evaluation.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import random

import kagglehub
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Cityscapes, ImageFolder, VOCSegmentation
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_IGNORE_INDEX = 255
VOC_NUM_CLASSES = 21
VOC_IGNORE_INDEX = 255
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SegmentationTransform:
    """Apply the same resize to image and mask, with task-appropriate interpolation and augmentations."""

    def __init__(self, image_size: tuple[int, int], is_train: bool = False) -> None:
        self.image_size = image_size
        self.is_train = is_train

    def __call__(self, image: Image.Image, target: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_train:
            # Scale up slightly for random cropping
            scaled_size = (int(self.image_size[0] * 1.1), int(self.image_size[1] * 1.1))
            image = TF.resize(image, scaled_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
            target = TF.resize(target, scaled_size, interpolation=InterpolationMode.NEAREST)

            # Random crop down to target size
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.image_size)
            image = TF.crop(image, i, j, h, w)
            target = TF.crop(target, i, j, h, w)

            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                target = TF.hflip(target)
                
            # Mild color jitter
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        else:
            image = TF.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
            target = TF.resize(target, self.image_size, interpolation=InterpolationMode.NEAREST)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        target_tensor = torch.as_tensor(
            TF.pil_to_tensor(target).squeeze(0),
            dtype=torch.long,
        )
        return image_tensor, target_tensor


class CityscapesSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Thin wrapper over torchvision Cityscapes with paired image/mask transforms."""

    def __init__(
        self,
        root: str,
        split: str,
        image_size: tuple[int, int] = (256, 512),
        mode: str = "fine",
        target_type: str = "semantic",
    ) -> None:
        self.base_dataset = Cityscapes(
            root=root,
            split=split,
            mode=mode,
            target_type=target_type,
        )
        self.transform = SegmentationTransform(image_size, is_train=(split == "train"))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, target = self.base_dataset[index]
        return self.transform(image, target)


class VOCSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Wrapper over torchvision VOCSegmentation with paired image/mask transforms."""

    def __init__(
        self,
        root: str,
        image_set: str,
        image_size: tuple[int, int] = (256, 256),
        download: bool = False,
    ) -> None:
        self.base_dataset = VOCSegmentation(
            root=root,
            year="2012",
            image_set=image_set,
            download=download,
        )
        self.transform = SegmentationTransform(image_size, is_train=(image_set == "train"))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, target = self.base_dataset[index]
        return self.transform(image, target)


def download_cifake() -> str:
    """Download the CIFAKE dataset via kagglehub and return the root path."""
    return kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")


def load_cifake(
    split: str = "train",
    data_root: str | None = None,
) -> tuple[ImageFolder, dict[str, Any]]:
    """Legacy CIFAKE loader kept for reference.

    CIFAKE is a classification dataset and is incompatible with the segmentation
    training pipeline implemented in this project.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    if data_root is None:
        data_root = download_cifake()

    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Expected directory '{split_dir}' not found. "
            f"Check that data_root='{data_root}' contains train/ and test/ folders."
        )

    dataset = ImageFolder(root=split_dir)
    config = {
        "name": "cifake",
        "task": "classification",
        "num_classes": len(dataset.classes),
        "class_names": dataset.classes,
    }
    return dataset, config


def load_cityscapes(
    split: str = "train",
    data_root: str | None = None,
    image_size: tuple[int, int] = (256, 512),
) -> tuple[CityscapesSegmentationDataset, dict[str, Any]]:
    """Load the Cityscapes semantic segmentation dataset."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

    if data_root is None:
        data_root = os.path.join("data", "cityscapes")

    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Cityscapes root '{data_root}' was not found. "
            "Download the dataset from the official site and extract it there."
        )

    dataset = CityscapesSegmentationDataset(
        root=data_root,
        split=split,
        image_size=image_size,
    )
    config = {
        "name": "cityscapes",
        "task": "segmentation",
        "num_classes": CITYSCAPES_NUM_CLASSES,
        "ignore_index": CITYSCAPES_IGNORE_INDEX,
        "class_names": [cls.name for cls in Cityscapes.classes if cls.train_id not in (-1, 255)],
        "image_size": image_size,
    }
    return dataset, config


def load_pascal_voc(
    split: str = "train",
    data_root: str | None = None,
    image_size: tuple[int, int] = (256, 256),
) -> tuple[VOCSegmentationDataset, dict[str, Any]]:
    """Load the Pascal VOC 2012 semantic segmentation dataset."""
    if split not in ("train", "val", "trainval"):
        raise ValueError(f"split must be 'train', 'val', or 'trainval', got '{split}'")

    if data_root is None:
        data_root = os.path.join("data", "voc")

    os.makedirs(data_root, exist_ok=True)

    dataset = VOCSegmentationDataset(
        root=data_root,
        image_set=split,
        image_size=image_size,
        download=True,
    )
    config = {
        "name": "pascal_voc",
        "task": "segmentation",
        "num_classes": VOC_NUM_CLASSES,
        "ignore_index": VOC_IGNORE_INDEX,
        "class_names": [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ],
        "image_size": image_size,
    }
    return dataset, config


DATASET_LOADERS: dict[str, Callable[..., tuple[Dataset[Any], dict[str, Any]]]] = {
    "cifake": load_cifake,
    "cityscapes": load_cityscapes,
    "pascal_voc": load_pascal_voc,
    "voc": load_pascal_voc,
}


def load_dataset(
    name: str,
    split: str,
    data_root: str | None = None,
    image_size: tuple[int, int] = (256, 512),
) -> tuple[Dataset[Any], dict[str, Any]]:
    """Load a supported dataset and return `(dataset, config)`."""
    name = name.lower()
    if name not in DATASET_LOADERS:
        supported = ", ".join(sorted(DATASET_LOADERS))
        raise ValueError(f"Unsupported dataset '{name}'. Supported datasets: {supported}")

    if name == "cityscapes":
        return load_cityscapes(split=split, data_root=data_root, image_size=image_size)

    if name in ("pascal_voc", "voc"):
        return load_pascal_voc(split=split, data_root=data_root, image_size=image_size)

    return DATASET_LOADERS[name](split=split, data_root=data_root)
