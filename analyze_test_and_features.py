"""Analyze the test image's periodicity and compare with training data.
Also check what engineered features look most discriminative."""
import tifffile, numpy as np
import os
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

base = r'f:\kaggle\vesuvius_challenge\vesuvius-challenge-surface-detection'

# === Test image periodicity ===
print("=== TEST IMAGE ANALYSIS (1407735, scroll 26002) ===")
test_img = tifffile.imread(os.path.join(base, 'test_images', '1407735.tif'))
print(f"Shape: {test_img.shape}")

# Check z-profiles at multiple locations
for name, cx, cy in [("center", 160, 160), ("q1", 80, 80), ("q3", 240, 240), ("q2", 80, 240), ("q4", 240, 80)]:
    profile = test_img[:, cx, cy].astype(float)
    
    # Autocorrelation
    p = profile - profile.mean()
    ac = np.correlate(p, p, mode='full')[len(p)-1:]
    if ac[0] > 0:
        ac /= ac[0]
    peaks, _ = find_peaks(ac, height=0.05, distance=8)
    
    # FFT
    fft = np.abs(np.fft.rfft(p))
    freqs = np.fft.rfftfreq(len(p))
    top_idx = np.argsort(fft[1:])[-3:] + 1
    top_periods = [(1.0/freqs[i], fft[i]) for i in sorted(top_idx, key=lambda x: fft[x], reverse=True)]
    
    print(f"  {name}: AC peaks at {peaks[:5].tolist()}, "
          f"FFT top periods: {[f'{p:.0f}vox' for p,m in top_periods]}")

# === Compare deprecated label for 1407735 with test image structure ===
print("\n=== DEPRECATED LABEL FOR TEST ID (1407735) ===")
dep_lbl = tifffile.imread(os.path.join(base, 'deprecated_train_labels', '1407735.tif'))
dep_img = tifffile.imread(os.path.join(base, 'deprecated_train_images', '1407735.tif'))

# Are the deprecated image and test image the same?
if dep_img.shape == test_img.shape:
    diff = np.abs(dep_img.astype(int) - test_img.astype(int))
    print(f"  Deprecated image vs test image: max_diff={diff.max()}, mean_diff={diff.mean():.4f}")
    if diff.max() == 0:
        print(f"  ** IDENTICAL ** - deprecated image IS the test image!")

# Analyze sheets from deprecated label
for name, cx, cy in [("center", 160, 160), ("q1", 80, 80), ("q3", 240, 240)]:
    lbl_line = dep_lbl[:, cx, cy]
    surf_z = np.where(lbl_line == 1)[0]
    if len(surf_z) > 0:
        sheets = [[surf_z[0]]]
        for i in range(1, len(surf_z)):
            if surf_z[i] - surf_z[i-1] > 3:
                sheets.append([surf_z[i]])
            else:
                sheets[-1].append(surf_z[i])
        centers = [np.mean(s) for s in sheets]
        spacings = [centers[i+1]-centers[i] for i in range(len(centers)-1)]
        thicknesses = [len(s) for s in sheets]
        print(f"  {name}: {len(sheets)} sheets, spacings={[f'{s:.0f}' for s in spacings]}, "
              f"thicknesses={thicknesses}")

# === Feature discriminability analysis ===
print("\n=== FEATURE DISCRIMINABILITY (which features separate surface from non-surface?) ===")
# Load a sample with good sheet coverage
img = tifffile.imread(os.path.join(base, 'train_images', '2203617984.tif'))
lbl = tifffile.imread(os.path.join(base, 'train_labels', '2203617984.tif'))

# Compute various features
print("Computing features...")

# 1. Raw intensity
# 2. Smoothed intensity at multiple scales
img_s1 = gaussian_filter(img.astype(float), sigma=1.0)
img_s2 = gaussian_filter(img.astype(float), sigma=2.0)
img_s4 = gaussian_filter(img.astype(float), sigma=4.0)

# 3. Laplacian of Gaussian (blob/edge detector)
log_s1 = gaussian_filter(img.astype(float), sigma=1.0) - gaussian_filter(img.astype(float), sigma=1.5)
log_s2 = gaussian_filter(img.astype(float), sigma=2.0) - gaussian_filter(img.astype(float), sigma=3.0)

# 4. Gradient magnitude
gz = np.gradient(img_s1, axis=0)
gy = np.gradient(img_s1, axis=1)
gx = np.gradient(img_s1, axis=2)
grad_mag = np.sqrt(gz**2 + gy**2 + gx**2)

# 5. Directional gradient ratios
gy_over_gx = np.abs(gy) / (np.abs(gx) + 1e-6)
gz_over_gx = np.abs(gz) / (np.abs(gx) + 1e-6)

# 6. Hessian-based features (approximate eigenvalues for sheet detection)
# For a sheet: two large eigenvalues, one near zero
# Trace of Hessian = sum of eigenvalues
hzz = np.gradient(gz, axis=0)
hyy = np.gradient(gy, axis=1)
hxx = np.gradient(gx, axis=2)
hessian_trace = hzz + hyy + hxx

features = {
    'raw_intensity': img.astype(float),
    'smooth_s1': img_s1,
    'smooth_s2': img_s2,
    'smooth_s4': img_s4,
    'LoG_s1': log_s1,
    'LoG_s2': log_s2,
    'grad_mag': grad_mag,
    'gy/gx_ratio': gy_over_gx,
    'gz/gx_ratio': gz_over_gx,
    'hessian_trace': hessian_trace,
}

# Compare distributions for each label
print(f"\n{'Feature':<20} {'Surface(1)':>12} {'Interior(0)':>12} {'Background(2)':>12} {'Surf-Int diff':>14}")
print("-" * 72)
for feat_name, feat_vol in features.items():
    means = {}
    for lv, ln in [(1, 'Surface'), (0, 'Interior'), (2, 'Background')]:
        mask = lbl == lv
        if mask.sum() > 0:
            means[lv] = feat_vol[mask].mean()
        else:
            means[lv] = float('nan')
    
    # How well does this feature separate surface (1) from interior (0)?
    diff = abs(means[1] - means[0]) / max(abs(means[0]), 1e-6) * 100
    print(f"{feat_name:<20} {means[1]:>12.3f} {means[0]:>12.3f} {means[2]:>12.3f} {diff:>12.1f}%")
