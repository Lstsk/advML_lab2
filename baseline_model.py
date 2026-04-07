"""Plain U-Net baseline for semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn


class UNetDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNetDownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            UNetDoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = UNetDoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 19, base_ch: int = 64) -> None:
        super().__init__()
        self.enc1 = UNetDoubleConv(in_channels, base_ch)
        self.enc2 = UNetDownBlock(base_ch, base_ch * 2)
        self.enc3 = UNetDownBlock(base_ch * 2, base_ch * 4)
        self.enc4 = UNetDownBlock(base_ch * 4, base_ch * 8)
        self.bottleneck = UNetDownBlock(base_ch * 8, base_ch * 16)

        self.dec1 = UNetUpBlock(base_ch * 16, base_ch * 8, base_ch * 8)
        self.dec2 = UNetUpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec3 = UNetUpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec4 = UNetUpBlock(base_ch * 2, base_ch, base_ch)

        self.segmentation_head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)

        d1 = self.dec1(b, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)
        return self.segmentation_head(d4)
