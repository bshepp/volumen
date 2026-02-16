# Pipeline: V1 (src/) — See PIPELINES.md
"""
Feature computation module for periodicity-aware surface detection.

Computes 6-channel input from raw CT volume:
  Channel 0: raw_CT (normalized to [0,1])
  Channel 1: LoG at sigma=1.0 (ridge detection, fine scale)
  Channel 2: LoG at sigma=2.0 (ridge detection, coarse scale)
  Channel 3: Hessian trace (sheet-like curvature)
  Channel 4: gradient magnitude (edge strength)
  Channel 5: gy/gx ratio (fiber orientation anisotropy)

From our analysis:
  - LoG_s2 separates surface vs interior by 1103%
  - Hessian trace separates by 878%
  - gy/gx ratio separates by 40%
  - Raw intensity separates by only 9.6%

Two implementations:
  - CPU (scipy): used for small-scale testing and legacy compatibility
  - GPU (PyTorch): uses separable 3D conv with fixed kernels, 10-50x faster
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_FEATURE_CHANNELS = 6

FEATURE_NAMES = [
    "raw_CT",
    "LoG_s1",
    "LoG_s2",
    "hessian_trace",
    "grad_mag",
    "gy_gx_ratio",
]


# ---------------------------------------------------------------------------
# GPU Feature Extractor (PyTorch)
# ---------------------------------------------------------------------------


def _gaussian_kernel_1d(sigma: float) -> torch.Tensor:
    """Create a 1D Gaussian kernel with appropriate size for the given sigma."""
    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-0.5 * x ** 2 / sigma ** 2)
    kernel = kernel / kernel.sum()
    return kernel


class GPUFeatureExtractor(nn.Module):
    """Compute 6-channel periodicity features entirely on GPU.

    Uses separable 3D Gaussian convolutions with fixed (non-learnable) kernels.
    Input:  (B, 1, Z, Y, X) raw CT normalized to [0, 1]
    Output: (B, 6, Z, Y, X) feature channels, per-sample normalized
    """

    def __init__(self):
        super().__init__()

        # Pre-build all fixed kernels as buffers (move with .to(device))
        # Gaussian kernels for LoG computation (DoG approximation)
        for sigma in [1.0, 1.5, 2.0, 3.0]:
            k1d = _gaussian_kernel_1d(sigma)
            name = f"gauss_{str(sigma).replace('.', '_')}"
            K = len(k1d)
            self.register_buffer(f"{name}_z", k1d.view(1, 1, K, 1, 1))
            self.register_buffer(f"{name}_y", k1d.view(1, 1, 1, K, 1))
            self.register_buffer(f"{name}_x", k1d.view(1, 1, 1, 1, K))
            self.register_buffer(f"{name}_pad", torch.tensor(K // 2))

        # Gradient kernels (central difference)
        grad = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32)
        self.register_buffer("grad_z", grad.view(1, 1, 3, 1, 1))
        self.register_buffer("grad_y", grad.view(1, 1, 1, 3, 1))
        self.register_buffer("grad_x", grad.view(1, 1, 1, 1, 3))

        # Second derivative kernels
        d2 = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32)
        self.register_buffer("d2_z", d2.view(1, 1, 3, 1, 1))
        self.register_buffer("d2_y", d2.view(1, 1, 1, 3, 1))
        self.register_buffer("d2_x", d2.view(1, 1, 1, 1, 3))

    def _gauss_smooth(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply separable 3D Gaussian smoothing."""
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
        """Compute 6-channel features from raw CT input.

        Args:
            raw: (B, 1, Z, Y, X) float32, values in [0, 1].

        Returns:
            features: (B, 6, Z, Y, X) float32, per-channel normalized.
        """
        # Channel 0: raw CT (already normalized to [0,1])
        ch0 = raw

        # LoG via Difference of Gaussians
        # Channel 1: LoG sigma=1.0 ≈ G(1.0) - G(1.5)
        g1_0 = self._gauss_smooth(raw, 1.0)
        g1_5 = self._gauss_smooth(raw, 1.5)
        ch1 = g1_0 - g1_5

        # Channel 2: LoG sigma=2.0 ≈ G(2.0) - G(3.0)
        g2_0 = self._gauss_smooth(raw, 2.0)
        g3_0 = self._gauss_smooth(raw, 3.0)
        ch2 = g2_0 - g3_0

        # Smoothed volume for gradient/Hessian (same as CPU version)
        smooth = g1_0  # reuse G(1.0) we already computed

        # Gradients
        gz = F.conv3d(smooth, self.grad_z, padding=(1, 0, 0))
        gy = F.conv3d(smooth, self.grad_y, padding=(0, 1, 0))
        gx = F.conv3d(smooth, self.grad_x, padding=(0, 0, 1))

        # Channel 3: Hessian trace = d²/dz² + d²/dy² + d²/dx²
        hzz = F.conv3d(smooth, self.d2_z, padding=(1, 0, 0))
        hyy = F.conv3d(smooth, self.d2_y, padding=(0, 1, 0))
        hxx = F.conv3d(smooth, self.d2_x, padding=(0, 0, 1))
        ch3 = hzz + hyy + hxx

        # Channel 4: gradient magnitude
        ch4 = torch.sqrt(gz ** 2 + gy ** 2 + gx ** 2 + 1e-8)

        # Channel 5: |gy| / |gx| ratio (fiber anisotropy)
        abs_gy = torch.abs(gy)
        abs_gx = torch.abs(gx)
        ch5 = abs_gy / (abs_gx + 1e-3)
        ch5 = torch.clamp(ch5, 0.0, 20.0)

        # Stack: (B, 6, Z, Y, X)
        features = torch.cat([ch0, ch1, ch2, ch3, ch4, ch5], dim=1)

        # Per-channel, per-sample normalization to zero mean / unit std
        mean = features.mean(dim=(2, 3, 4), keepdim=True)
        std = features.std(dim=(2, 3, 4), keepdim=True)
        features = (features - mean) / (std + 1e-8)

        return features


# ---------------------------------------------------------------------------
# CPU Feature Computation (scipy) — kept for legacy / small-scale testing
# ---------------------------------------------------------------------------


def compute_features(volume: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute 6-channel feature volume from a raw CT volume (CPU/scipy).

    Args:
        volume: 3D uint8 array (Z, Y, X), raw CT scan.
        normalize: If True, normalize each channel to zero mean / unit std.

    Returns:
        features: float32 array of shape (6, Z, Y, X).
    """
    from scipy.ndimage import gaussian_filter

    vol = volume.astype(np.float32)

    raw_norm = vol / 255.0
    smooth = gaussian_filter(vol, sigma=1.0)
    log_s1 = gaussian_filter(vol, sigma=1.0) - gaussian_filter(vol, sigma=1.5)
    log_s2 = gaussian_filter(vol, sigma=2.0) - gaussian_filter(vol, sigma=3.0)

    gz = np.gradient(smooth, axis=0)
    gy = np.gradient(smooth, axis=1)
    gx = np.gradient(smooth, axis=2)

    hzz = np.gradient(gz, axis=0)
    hyy = np.gradient(gy, axis=1)
    hxx = np.gradient(gx, axis=2)
    hessian_trace = hzz + hyy + hxx

    grad_mag = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)

    abs_gy = np.abs(gy)
    abs_gx = np.abs(gx)
    gy_gx_ratio = abs_gy / (abs_gx + 1e-3)
    gy_gx_ratio = np.clip(gy_gx_ratio, 0.0, 20.0)

    features = np.stack(
        [raw_norm, log_s1, log_s2, hessian_trace, grad_mag, gy_gx_ratio],
        axis=0,
    ).astype(np.float32)

    if normalize:
        features = normalize_features(features)

    return features


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize each feature channel to zero mean, unit variance."""
    out = np.empty_like(features)
    for c in range(features.shape[0]):
        ch = features[c]
        mu = ch.mean()
        std = ch.std()
        if std < 1e-8:
            out[c] = ch - mu
        else:
            out[c] = (ch - mu) / std
    return out


def compute_features_for_patch(
    patch: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Compute features for a small patch (CPU, legacy)."""
    pad = 8
    padded = np.pad(patch, pad, mode="reflect")
    feat = compute_features(padded, normalize=normalize)
    feat = feat[:, pad:-pad, pad:-pad, pad:-pad]
    return feat
