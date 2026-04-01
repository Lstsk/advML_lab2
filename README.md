# TransUNet for AI-Generated Image Detection

Binary classification of real vs. AI-generated images using a **TransUNet** (Transformer + U-Net) architecture on the **CIFAKE** dataset.

## Overview

This project adapts the TransUNet architecture — which combines a Vision Transformer encoder with a U-Net decoder — for the task of detecting AI-generated images. Rather than segmentation, the model is repurposed for binary classification: **Real (0)** vs. **AI-Generated (1)**.

## Dataset

**CIFAKE: Real and AI-Generated Synthetic Images**
- Source: [Kaggle – birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- 60,000 training images and 10,000 test images (32×32, RGB)
- Two classes: `REAL` and `FAKE`

```python
import kagglehub

path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
print("Path to dataset files:", path)
```

## Architecture

**TransUNet** merges the strengths of Transformers and U-Net:

1. **CNN Encoder** — extracts local feature maps from input images
2. **Transformer Encoder** — captures global context via self-attention on patch embeddings
3. **U-Net Decoder** — upsamples with skip connections from the CNN encoder
4. **Classification Head** — global average pooling + fully connected layer for binary output

## Project Structure

```
├── model.py          # TransUNet model implementation
├── dataset.py        # CIFAKE dataset loading and preprocessing
├── train.py          # Training loop
├── eval.py           # Evaluation script
├── pyproject.toml    # Project config and dependencies (managed by uv)
├── uv.lock           # Locked dependency versions
├── data/             # Dataset files (not tracked)
└── results/          # Checkpoints and outputs
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync

# Download the dataset
uv run python -c "import kagglehub; kagglehub.dataset_download('birdy654/cifake-real-and-ai-generated-synthetic-images')"
```

## Usage

```bash
# Train the model
uv run python train.py

# Evaluate
uv run python eval.py
```

## Dependencies

- Python ≥ 3.12
- torch
- torchvision
- kagglehub
- numpy
- Pillow
- matplotlib

All dependencies are managed via `pyproject.toml`. Run `uv sync` to install them.
