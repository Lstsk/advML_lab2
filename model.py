"""TransUNet for semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive convolution blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownBlock(nn.Module):
    """Downsample via max-pooling followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class UpBlock(nn.Module):
    """Upsample, concatenate skip features, then refine."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class PatchEmbedding(nn.Module):
    """Flatten spatial feature maps to tokens and add positional embeddings."""

    def __init__(self, embed_dim: int, num_patches: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, _ = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x + self.pos_embed


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        return x + self.mlp(self.norm2(x))


class TransformerEncoder(nn.Module):
    """Transformer encoder operating on flattened bottleneck tokens."""

    def __init__(self, embed_dim: int, num_heads: int, depth: int, num_patches: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(embed_dim, num_patches)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x.transpose(1, 2).view(b, c, h, w)


class TransUNet(nn.Module):
    """TransUNet for semantic segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 19,
        base_ch: int = 64,
        transformer_heads: int = 8,
        transformer_depth: int = 6,
        dropout: float = 0.1,
        input_size: tuple[int, int] = (256, 512),
    ) -> None:
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_ch)
        self.enc2 = DownBlock(base_ch, base_ch * 2)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4)
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8)

        bottleneck_h = input_size[0] // 8
        bottleneck_w = input_size[1] // 8
        num_patches = bottleneck_h * bottleneck_w

        self.transformer = TransformerEncoder(
            embed_dim=base_ch * 8,
            num_heads=transformer_heads,
            depth=transformer_depth,
            num_patches=num_patches,
            dropout=dropout,
        )

        self.dec1 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec3 = UpBlock(base_ch * 2, base_ch, base_ch)
        self.segmentation_head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        t = self.transformer(s4)

        d1 = self.dec1(t, s3)
        d2 = self.dec2(d1, s2)
        d3 = self.dec3(d2, s1)
        return self.segmentation_head(d3)
