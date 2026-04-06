# TransUNet and U-Net for Semantic Segmentation

This project compares:

- `baseline_model.py`: plain U-Net baseline
- `model.py`: TransUNet

The default dataset is now **Pascal VOC 2012** via torchvision.

## Files

- `dataset.py`: single dataset-loading entrypoint
- `train.py`: train TransUNet
- `train_baseline.py`: train U-Net baseline
- `eval.py`: evaluate TransUNet
- `eval_baseline.py`: evaluate U-Net baseline

## Dataset

### Pascal VOC 2012

The code uses `torchvision.datasets.VOCSegmentation` and can download VOC 2012 automatically.

- Dataset name in scripts: `voc`
- Default root: `data/voc`
- Default image size: `256x256`
- Number of classes: `21`

The first time you run training or evaluation, torchvision will download the dataset into `data/voc`.

## Setup

```bash
uv sync
```

## Training

Train the U-Net baseline:

```bash
uv run python train_baseline.py
```

Train TransUNet:

```bash
uv run python train.py
```

You can also pass the dataset explicitly:

```bash
uv run python train_baseline.py --dataset voc --data-root data/voc
uv run python train.py --dataset voc --data-root data/voc
```

## Evaluation

Evaluate the U-Net baseline:

```bash
uv run python eval_baseline.py
```

Evaluate TransUNet:

```bash
uv run python eval.py
```

## Notes

- Evaluation defaults to the VOC `val` split.
- Metrics reported are pixel accuracy and mean IoU.
- `dataset.py` still contains Cityscapes and legacy CIFAKE loaders, but the default path is VOC.
