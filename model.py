import torch
import torch.nn as nn


# the left side is bigger than the right side at the same level so we need to crop it. because padding = 2 
def center_crop(feature_map: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    _, _, target_height, target_width = target_tensor.shape
    _, _, feature_height, feature_width = feature_map.shape

    delta_height = feature_height - target_height
    delta_width = feature_width - target_width

    top = delta_height // 2
    left = delta_width // 2
    bottom = top + target_height
    right = left + target_width

    return feature_map[:, :, top:bottom, left:right]


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        skip = center_crop(skip, x)
        

        # concat the skip connection to current
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DownSampleBlock(64, 128)
        self.enc3 = DownSampleBlock(128, 256)
        self.enc4 = DownSampleBlock(256, 512)
        self.bottleneck = DownSampleBlock(512, 1024)

        self.dec1 = UpSampleBlock(1024, 512)
        self.dec2 = UpSampleBlock(512, 256)
        self.dec3 = UpSampleBlock(256, 128)
        self.dec4 = UpSampleBlock(128, 64)

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        skip4 = self.enc4(skip3)
        bottleneck = self.bottleneck(skip4)

        x = self.dec1(bottleneck, skip4)
        x = self.dec2(x, skip3)
        x = self.dec3(x, skip2)
        x = self.dec4(x, skip1)

        return self.output(x)
