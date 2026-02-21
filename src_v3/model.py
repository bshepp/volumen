# Pipeline: V3 (src_v3/) — See PIPELINES.md
"""
Multi-Scale Fusion UNet for Pipeline V3.

Processes patches at 32³, 64³, 128³ and learns to fuse predictions.
The model selects when to trust fine vs coarse scale via learned fusion.

WARNING: This is Pipeline V3. Do NOT import from src/ or src_v2/.
See PIPELINES.md in the project root.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Two 3x3x3 convolutions with InstanceNorm and LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Encoder block: strided conv for downsampling, then ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm3d(in_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv = ConvBlock3D(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Decoder block: upsample + concat skip + ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2, bias=False)
        self.conv = ConvBlock3D(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            diff_z = skip.shape[2] - x.shape[2]
            diff_y = skip.shape[3] - x.shape[3]
            diff_x = skip.shape[4] - x.shape[4]
            x = F.pad(
                x,
                [
                    diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2,
                    diff_z // 2, diff_z - diff_z // 2,
                ],
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3DSimple(nn.Module):
    """Standard 3D U-Net that returns logits only (no aux heads)."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 3,
        base_filters: int = 32,
        depth: int = 4,
    ):
        super().__init__()
        filters = [base_filters * (2 ** i) for i in range(depth)]
        self.init_conv = ConvBlock3D(in_channels, filters[0])
        self.encoders = nn.ModuleList()
        for i in range(1, depth):
            self.encoders.append(DownBlock(filters[i - 1], filters[i]))
        self.bottleneck = DownBlock(filters[-1], filters[-1] * 2)
        self.decoders = nn.ModuleList()
        dec_in = filters[-1] * 2
        for i in range(depth - 1, -1, -1):
            self.decoders.append(UpBlock(dec_in, filters[i], filters[i]))
            dec_in = filters[i]
        self.final_conv = nn.Conv3d(filters[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.init_conv(x)
        skips.append(x)
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        x = self.bottleneck(x)
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            x = dec(x, skip)
        return self.final_conv(x)


class MultiScaleFusionUNet(nn.Module):
    """Multi-scale fusion: 3 UNets (32³, 64³, 128³) + learned fusion.

    Input: (B, 6, 128, 128, 128) features.
    Extracts center 64³ and 32³ crops internally.
    Returns fused logits at 128³ resolution.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 3,
        base_filters: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.unet_128 = UNet3DSimple(
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=base_filters,
            depth=4,
        )
        self.unet_64 = UNet3DSimple(
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=base_filters,
            depth=4,
        )
        self.unet_32 = UNet3DSimple(
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=base_filters,
            depth=3,
        )
        self.fusion = nn.Conv3d(num_classes * 3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, 128, 128, 128) feature tensor.

        Returns:
            Fused logits (B, num_classes, 128, 128, 128).
        """
        # Branch 1: full 128³
        logits_128 = self.unet_128(x)

        # Branch 2: center 64³
        crop_64 = x[:, :, 32:96, 32:96, 32:96]
        logits_64 = self.unet_64(crop_64)
        logits_64_up = F.interpolate(
            logits_64, size=(128, 128, 128), mode="trilinear", align_corners=False
        )

        # Branch 3: center 32³
        crop_32 = x[:, :, 48:80, 48:80, 48:80]
        logits_32 = self.unet_32(crop_32)
        logits_32_up = F.interpolate(
            logits_32, size=(128, 128, 128), mode="trilinear", align_corners=False
        )

        stacked = torch.cat([logits_128, logits_64_up, logits_32_up], dim=1)
        return self.fusion(stacked)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(
    in_channels: int = 6,
    num_classes: int = 3,
    base_filters: int = 16,
) -> MultiScaleFusionUNet:
    """Factory for V3 multi-scale fusion model."""
    return MultiScaleFusionUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=base_filters,
    )
