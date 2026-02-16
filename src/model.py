# Pipeline: V1 (src/) — See PIPELINES.md
"""
3D U-Net architecture for volumetric surface segmentation.

Architecture:
  - 4 encoder stages: [32, 64, 128, 256] filters
  - 4 decoder stages with skip connections
  - Instance normalization + LeakyReLU
  - 6 input channels (periodicity-aware features)
  - 3 output classes (background, surface, interior)
  - Compatible with mixed precision (FP16) training
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
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """Decoder block: upsample + concat skip + ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2, bias=False)
        self.conv = ConvBlock3D(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle potential size mismatch from odd dimensions
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
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    """3D U-Net for volumetric segmentation.

    Args:
        in_channels: Number of input feature channels (default 6).
        num_classes: Number of output classes (default 3).
        base_filters: Number of filters in the first stage (default 32).
        depth: Number of encoder/decoder stages (default 4).
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 3,
        base_filters: int = 32,
        depth: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth

        # Compute filter sizes: [32, 64, 128, 256] for depth=4
        filters = [base_filters * (2 ** i) for i in range(depth)]

        # Initial convolution block
        self.init_conv = ConvBlock3D(in_channels, filters[0])

        # Encoder path
        self.encoders = nn.ModuleList()
        for i in range(1, depth):
            self.encoders.append(DownBlock(filters[i - 1], filters[i]))

        # Bottleneck
        self.bottleneck = DownBlock(filters[-1], filters[-1] * 2)

        # Decoder path
        self.decoders = nn.ModuleList()
        dec_in = filters[-1] * 2
        for i in range(depth - 1, -1, -1):
            self.decoders.append(UpBlock(dec_in, filters[i], filters[i]))
            dec_in = filters[i]

        # Final 1x1x1 convolution for classification
        self.final_conv = nn.Conv3d(filters[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, Z, Y, X) input tensor.

        Returns:
            logits: (B, num_classes, Z, Y, X) raw logits.
        """
        # Encoder
        skips = []
        x = self.init_conv(x)
        skips.append(x)

        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            x = decoder(x, skip)

        # Classification
        logits = self.final_conv(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities via softmax.

        Args:
            x: (B, C, Z, Y, X) input tensor.

        Returns:
            probs: (B, num_classes, Z, Y, X) probabilities.
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(
    in_channels: int = 6,
    num_classes: int = 3,
    base_filters: int = 32,
    depth: int = 4,
) -> UNet3D:
    """Factory function to create the model."""
    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=base_filters,
        depth=depth,
    )
    return model
