import torch
import torch.nn as nn
import torchvision.models as models


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim: int, num_patches: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, _ = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x + self.pos_embed


class TransformerBlock(nn.Module):
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
    def __init__(
        self,
        num_classes: int = 21,
        transformer_heads: int = 8,
        transformer_depth: int = 6,
        dropout: float = 0.1,
        input_size: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()

        # Use a ResNet-50 backbone for the encoder side.
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        bottleneck_h = input_size[0] // 16
        bottleneck_w = input_size[1] // 16
        num_patches = bottleneck_h * bottleneck_w

        # transformer runs on the 1/16 resolution feature map.
        self.transformer = TransformerEncoder(
            embed_dim=1024,
            num_heads=transformer_heads,
            depth=transformer_depth,
            num_patches=num_patches,
            dropout=dropout,
        )

        # Decoder mirrors the encoder, but the last step just upsamples to full size.
        self.dec1 = UpBlock(in_ch=1024, skip_ch=512, out_ch=512)
        self.dec2 = UpBlock(in_ch=512, skip_ch=256, out_ch=256)
        self.dec3 = UpBlock(in_ch=256, skip_ch=64, out_ch=128)
        self.dec4 = UpBlock(in_ch=128, skip_ch=0, out_ch=64)
        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.bn1(self.conv1(x)))
        x_p = self.maxpool(x1)
        x2 = self.layer1(x_p)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)

        t = self.transformer(x4)

        d1 = self.dec1(t, x3)
        d2 = self.dec2(d1, x2)
        d3 = self.dec3(d2, x1)
        d4 = self.dec4(d3, skip=None)

        return self.segmentation_head(d4)
