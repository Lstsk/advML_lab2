import os
import random
from functools import partial
from typing import Any
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


VOC_NUM_CLASSES = 21
VOC_IGNORE_INDEX = 255
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def image_transform(
    image: Image.Image,
    target: Image.Image,
    image_size: tuple[int, int],
    train: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if train:
        big_size = (int(image_size[0] * 1.1), int(image_size[1] * 1.1))

        image = TF.resize(image, big_size, interpolation=InterpolationMode.BILINEAR, antialias=True)

        # keep the mask labels as labels, not blended colors
        target = TF.resize(target, big_size, interpolation=InterpolationMode.NEAREST)

        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=image_size)
        image = TF.crop(image, i, j, h, w)
        target = TF.crop(target, i, j, h, w)

        if random.random() > 0.5:
            image = TF.hflip(image)
            target = TF.hflip(target)

        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
    else:
        image = TF.resize(image, image_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        target = TF.resize(target, image_size, interpolation=InterpolationMode.NEAREST)

    image = TF.normalize(TF.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
    target = torch.as_tensor(TF.pil_to_tensor(target).squeeze(0), dtype=torch.long)
    return image, target


def load_dataset(
    split: str = "train",
    data_root: str | None = None,
    image_size: tuple[int, int] = (256, 256),
) -> tuple[VOCSegmentation, dict[str, Any]]:
    if split not in ("train", "val", "trainval"):
        raise ValueError(f"split must be 'train', 'val', or 'trainval', got '{split}'")

    if data_root is None:
        data_root = os.path.join("data", "voc")

    os.makedirs(data_root, exist_ok=True)
    transform = partial(image_transform, image_size=image_size, train=(split == "train"))
    dataset = VOCSegmentation(root=data_root, year="2012", image_set=split, download=True, transforms=transform)
    config = {
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
    }
    return dataset, config
