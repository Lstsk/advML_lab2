"""UNet: Plain U-Net baseline for binary image classification.

Architecture for 32×32 CIFAKE images (no Transformer bottleneck):
  CNN Encoder → DoubleConv Bottleneck → U-Net Decoder → Classification Head

This mirrors the TransUNet encoder/decoder structure exactly, replacing only
the Transformer bottleneck with a simple DoubleConv block for comparison.
"""

import torch
import torch.nn as nn

from model import DoubleConv, DownBlock, UpBlock


class UNet(nn.Module):
    """Plain U-Net for binary classification on 32×32 images.

    Architecture:
        Encoder:
            enc1: 32×32 → 64 channels
            enc2: 16×16 → 128 channels
            enc3:  8×8  → 256 channels
        Bottleneck:
            DoubleConv(256, 256)         ← no Transformer
        Decoder:
            dec1: 16×16 → 128 channels (skip from enc2)
            dec2: 32×32 → 64 channels  (skip from enc1)
        Classification head:
            Global Average Pooling → FC → num_classes
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_ch: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # --- CNN Encoder ---
        self.enc1 = DoubleConv(in_channels, base_ch)        # 32×32
        self.enc2 = DownBlock(base_ch, base_ch * 2)          # 16×16
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4)      #  8×8

        # --- Bottleneck (replaces Transformer) ---
        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 4)  # 8×8, 256→256

        # --- U-Net Decoder ---
        self.dec1 = UpBlock(base_ch * 4, base_ch * 2)   # 8→16, skip from enc2
        self.dec2 = UpBlock(base_ch * 2, base_ch)        # 16→32, skip from enc1

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)           # (B, 64,  32, 32)
        s2 = self.enc2(s1)          # (B, 128, 16, 16)
        s3 = self.enc3(s2)          # (B, 256,  8,  8)

        # Bottleneck
        b = self.bottleneck(s3)     # (B, 256,  8,  8)

        # Decoder with skip connections
        d1 = self.dec1(b, s2)       # (B, 128, 16, 16)
        d2 = self.dec2(d1, s1)      # (B, 64,  32, 32)

        # Classification
        out = self.classifier(d2)   # (B, num_classes)
        return out
