# Pipeline: V2 (src_v2/) — See PIPELINES.md
"""
3D U-Net with Deep Supervision for volumetric surface segmentation.

Pipeline V2 model. See PIPELINES.md for context.
DO NOT import from src/ (Pipeline V1). This file is self-contained.

Architecture changes vs V1:
  - Auxiliary 1x1x1 segmentation heads at each decoder stage
  - During training: returns dict with "logits" + "aux_1", "aux_2", "aux_3"
  - During eval:     returns only the main logits tensor (backward-compatible)

Auxiliary outputs are at 2x, 4x, 8x downsampled resolution relative to input,
used for deep supervision loss computation.
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


class UNet3DDeepSup(nn.Module):
    """3D U-Net with deep supervision auxiliary heads.

    During training (self.training == True):
      Returns a dict:
        {"logits": (B, C, Z, Y, X),
         "aux_1": (B, C, Z/2, Y/2, X/2),
         "aux_2": (B, C, Z/4, Y/4, X/4),
         "aux_3": (B, C, Z/8, Y/8, X/8)}

    During eval (self.training == False):
      Returns only: (B, C, Z, Y, X) logits tensor.

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

        # Deep supervision auxiliary heads (one per decoder stage except the last)
        # Decoder 0 outputs at filters[depth-1] resolution (8x downsampled)
        # Decoder 1 outputs at filters[depth-2] resolution (4x downsampled)
        # Decoder 2 outputs at filters[depth-3] resolution (2x downsampled)
        # Decoder 3 outputs at filters[0] resolution (1x) -> this is the main head
        self.aux_heads = nn.ModuleList()
        for i in range(depth - 1):
            # Decoder i outputs at filters[depth-1-i] channels
            dec_out_ch = filters[depth - 1 - i]
            self.aux_heads.append(
                nn.Conv3d(dec_out_ch, num_classes, kernel_size=1)
            )

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: (B, C, Z, Y, X) input tensor.

        Returns:
            Training: dict with "logits", "aux_1", "aux_2", "aux_3"
            Eval: (B, num_classes, Z, Y, X) logits tensor.
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

        # Decoder with deep supervision collection
        aux_outputs = []
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            x = decoder(x, skip)

            # Collect auxiliary outputs (all decoder stages except the last)
            if i < self.depth - 1:
                aux_outputs.append(x)

        # Main classification head
        logits = self.final_conv(x)

        if not self.training:
            return logits

        # Deep supervision: compute aux logits
        # Reverse so aux_1 = finest auxiliary (2x downsampled),
        # aux_3 = coarsest (8x downsampled). This matches the loss
        # weights (0.5, 0.25, 0.125) so the most refined aux gets the
        # highest weight.
        result = {"logits": logits}
        num_aux = len(aux_outputs)
        for i, (aux_feat, aux_head) in enumerate(zip(aux_outputs, self.aux_heads)):
            result[f"aux_{num_aux - i}"] = aux_head(aux_feat)

        return result

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities via softmax (eval mode only)."""
        self.eval()
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
) -> UNet3DDeepSup:
    """Factory function to create the V2 deep supervision model."""
    return UNet3DDeepSup(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=base_filters,
        depth=depth,
    )
