"""TransUNet: Transformer + U-Net for binary image classification.

Architecture adapted for 32×32 CIFAKE images:
  CNN Encoder → Transformer Encoder → U-Net Decoder → Classification Head
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive (Conv2d → BatchNorm → ReLU) blocks."""

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
    """Downsample via MaxPool then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class UpBlock(nn.Module):
    """Upsample via ConvTranspose2d, concatenate skip, then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Transformer components
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Flatten spatial feature map into patch tokens + add positional encoding."""

    def __init__(self, embed_dim: int, num_patches: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, H*W, C)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C) where N = H*W
        x = x + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block: LayerNorm → MHSA → LayerNorm → FFN."""

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
        # Pre-norm self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # Pre-norm FFN
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer blocks operating on flattened spatial tokens."""

    def __init__(self, embed_dim: int, num_heads: int, depth: int, num_patches: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(embed_dim, num_patches)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.patch_embed(x)             # (B, N, C)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)  # back to (B, C, H, W)
        return x


# ---------------------------------------------------------------------------
# TransUNet
# ---------------------------------------------------------------------------

class TransUNet(nn.Module):
    """TransUNet adapted for binary classification on 32×32 images.

    Architecture (for a 32×32 input):
        Encoder:
            enc1: 32×32 → 64 channels
            enc2: 16×16 → 128 channels
            enc3:  8×8  → 256 channels
        Bottleneck:
            enc3 output → Transformer encoder (self-attention on 8×8 = 64 tokens)
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
        transformer_heads: int = 8,
        transformer_depth: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # --- CNN Encoder ---
        self.enc1 = DoubleConv(in_channels, base_ch)        # 32×32
        self.enc2 = DownBlock(base_ch, base_ch * 2)          # 16×16
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4)      #  8×8

        # --- Transformer Encoder (bottleneck) ---
        bottleneck_dim = base_ch * 4  # 256
        num_patches = 8 * 8           # spatial size at bottleneck for 32×32 input
        self.transformer = TransformerEncoder(
            embed_dim=bottleneck_dim,
            num_heads=transformer_heads,
            depth=transformer_depth,
            num_patches=num_patches,
            dropout=dropout,
        )

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
        s1 = self.enc1(x)       # (B, 64,  32, 32)
        s2 = self.enc2(s1)      # (B, 128, 16, 16)
        s3 = self.enc3(s2)      # (B, 256,  8,  8)

        # Transformer on bottleneck
        t = self.transformer(s3) # (B, 256,  8,  8)

        # Decoder with skip connections
        d1 = self.dec1(t, s2)    # (B, 128, 16, 16)
        d2 = self.dec2(d1, s1)   # (B, 64,  32, 32)

        # Classification
        out = self.classifier(d2)  # (B, num_classes)
        return out
