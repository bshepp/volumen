# Pipeline: V3 (src_v3/) — See PIPELINES.md
"""
Feature computation module for periodicity-aware surface detection.

WARNING: This is Pipeline V3. Do NOT import from src/ or src_v2/.
See PIPELINES.md in the project root.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_1d(sigma: float) -> torch.Tensor:
    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-0.5 * x ** 2 / sigma ** 2)
    return kernel / kernel.sum()


class GPUFeatureExtractor(nn.Module):
    """Compute 6-channel periodicity features entirely on GPU."""

    def __init__(self):
        super().__init__()
        for sigma in [1.0, 1.5, 2.0, 3.0]:
            k1d = _gaussian_kernel_1d(sigma)
            name = f"gauss_{str(sigma).replace('.', '_')}"
            K = len(k1d)
            self.register_buffer(f"{name}_z", k1d.view(1, 1, K, 1, 1))
            self.register_buffer(f"{name}_y", k1d.view(1, 1, 1, K, 1))
            self.register_buffer(f"{name}_x", k1d.view(1, 1, 1, 1, K))
            self.register_buffer(f"{name}_pad", torch.tensor(K // 2))
        grad = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32)
        self.register_buffer("grad_z", grad.view(1, 1, 3, 1, 1))
        self.register_buffer("grad_y", grad.view(1, 1, 1, 3, 1))
        self.register_buffer("grad_x", grad.view(1, 1, 1, 1, 3))
        d2 = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32)
        self.register_buffer("d2_z", d2.view(1, 1, 3, 1, 1))
        self.register_buffer("d2_y", d2.view(1, 1, 1, 3, 1))
        self.register_buffer("d2_x", d2.view(1, 1, 1, 1, 3))

    def _gauss_smooth(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        name = f"gauss_{str(sigma).replace('.', '_')}"
        kz = getattr(self, f"{name}_z")
        ky = getattr(self, f"{name}_y")
        kx = getattr(self, f"{name}_x")
        pad = int(getattr(self, f"{name}_pad").item())
        x = F.conv3d(x, kz, padding=(pad, 0, 0))
        x = F.conv3d(x, ky, padding=(0, pad, 0))
        x = F.conv3d(x, kx, padding=(0, 0, pad))
        return x

    @torch.no_grad()
    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        ch0 = raw
        g1_0 = self._gauss_smooth(raw, 1.0)
        g1_5 = self._gauss_smooth(raw, 1.5)
        ch1 = g1_0 - g1_5
        g2_0 = self._gauss_smooth(raw, 2.0)
        g3_0 = self._gauss_smooth(raw, 3.0)
        ch2 = g2_0 - g3_0
        smooth = g1_0
        gz = F.conv3d(smooth, self.grad_z, padding=(1, 0, 0))
        gy = F.conv3d(smooth, self.grad_y, padding=(0, 1, 0))
        gx = F.conv3d(smooth, self.grad_x, padding=(0, 0, 1))
        hzz = F.conv3d(smooth, self.d2_z, padding=(1, 0, 0))
        hyy = F.conv3d(smooth, self.d2_y, padding=(0, 1, 0))
        hxx = F.conv3d(smooth, self.d2_x, padding=(0, 0, 1))
        ch3 = hzz + hyy + hxx
        ch4 = torch.sqrt(gz ** 2 + gy ** 2 + gx ** 2 + 1e-8)
        ch5 = torch.clamp(torch.abs(gy) / (torch.abs(gx) + 1e-3), 0.0, 20.0)
        features = torch.cat([ch0, ch1, ch2, ch3, ch4, ch5], dim=1)
        mean = features.mean(dim=(2, 3, 4), keepdim=True)
        std = features.std(dim=(2, 3, 4), keepdim=True)
        return (features - mean) / (std + 1e-8)
