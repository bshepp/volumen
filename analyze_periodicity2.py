"""Deeper periodicity analysis - multiple lines per volume, more samples, 
directional texture at surface vs non-surface regions."""
import tifffile, numpy as np
import os, csv
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

base = r'f:\kaggle\vesuvius_challenge\vesuvius-challenge-surface-detection'

with open(os.path.join(base, 'train.csv')) as f:
    scroll_map = {r['id']: r['scroll_id'] for r in csv.DictReader(f)}

# Analyze sheet spacing more systematically across multiple samples and multiple lines
print("=== SYSTEMATIC SHEET SPACING ANALYSIS ===")
# Pick samples from different scrolls
samples = {
    '26002': ['105796630', '26894125', '1407735'],  # note: 1407735 is in deprecated
    '34117': ['102536988', '108672114', '11460685'],
    '35360': ['2203617984', '3320274', '90569866'],
    '26010': ['19797301', '45525309', '63469416'],
}

all_spacings = []
for scroll_id, ids in samples.items():
    print(f"\nScroll {scroll_id}:")
    for sid in ids:
        # Check if in train or deprecated
        img_path = os.path.join(base, 'train_images', sid + '.tif')
        lbl_path = os.path.join(base, 'train_labels', sid + '.tif')
        if not os.path.exists(img_path):
            img_path = os.path.join(base, 'deprecated_train_images', sid + '.tif')
            lbl_path = os.path.join(base, 'deprecated_train_labels', sid + '.tif')
        if not os.path.exists(img_path):
            print(f"  {sid}: FILE NOT FOUND")
            continue
            
        lbl_vol = tifffile.imread(lbl_path)
        sz = lbl_vol.shape[0]
        
        # Sample multiple z-lines
        sample_spacings = []
        sheet_counts = []
        for cx in range(sz//4, 3*sz//4, sz//8):
            for cy in range(sz//4, 3*sz//4, sz//8):
                lbl_line = lbl_vol[:, cx, cy]
                surf_z = np.where(lbl_line == 1)[0]
                if len(surf_z) > 0:
                    # Group into sheets
                    sheets = [[surf_z[0]]]
                    for i in range(1, len(surf_z)):
                        if surf_z[i] - surf_z[i-1] > 3:
                            sheets.append([surf_z[i]])
                        else:
                            sheets[-1].append(surf_z[i])
                    # Sheet centers
                    centers = [np.mean(s) for s in sheets]
                    sheet_counts.append(len(centers))
                    if len(centers) > 1:
                        for i in range(len(centers)-1):
                            sample_spacings.append(centers[i+1] - centers[i])
        
        if sample_spacings:
            sp = np.array(sample_spacings)
            all_spacings.extend(sample_spacings)
            sc = np.array(sheet_counts)
            print(f"  {sid}: sheets/line={sc.mean():.1f}+-{sc.std():.1f}, "
                  f"spacing: mean={sp.mean():.1f}, median={np.median(sp):.1f}, "
                  f"std={sp.std():.1f}, range=[{sp.min():.0f}, {sp.max():.0f}]")
        else:
            print(f"  {sid}: no multi-sheet lines found")

if all_spacings:
    sp = np.array(all_spacings)
    print(f"\nOVERALL: n={len(sp)}, mean={sp.mean():.1f}, median={np.median(sp):.1f}, "
          f"std={sp.std():.1f}, range=[{sp.min():.0f}, {sp.max():.0f}]")
    # Histogram bins
    bins = np.arange(0, 200, 10)
    hist, _ = np.histogram(sp, bins)
    print("Spacing histogram (10-voxel bins):")
    for i in range(len(hist)):
        if hist[i] > 0:
            bar = '#' * min(hist[i], 50)
            print(f"  {bins[i]:3d}-{bins[i+1]:3d}: {hist[i]:4d} {bar}")

# === DIRECTIONAL ANALYSIS: Does fiber orientation differ between recto surface and papyrus interior? ===
print("\n=== FIBER ORIENTATION: Surface (label 1) vs Interior (label 0) ===")
# Load a couple samples and compute structure tensor
for sid in ['102536988', '2203617984']:
    img = tifffile.imread(os.path.join(base, 'train_images', sid + '.tif'))
    lbl = tifffile.imread(os.path.join(base, 'train_labels', sid + '.tif'))
    
    # Compute gradients
    gz = np.diff(img.astype(float), axis=0)  # z gradient
    gy = np.diff(img.astype(float), axis=1)  # y gradient  
    gx = np.diff(img.astype(float), axis=2)  # x gradient
    
    # Crop to common size
    s0 = min(gz.shape[0], gy.shape[0], gx.shape[0])
    s1 = min(gz.shape[1], gy.shape[1], gx.shape[1])
    s2 = min(gz.shape[2], gy.shape[2], gx.shape[2])
    gz = gz[:s0, :s1, :s2]
    gy = gy[:s0, :s1, :s2]
    gx = gx[:s0, :s1, :s2]
    lbl_crop = lbl[:s0, :s1, :s2]
    
    for label_val, label_name in [(1, "Surface"), (0, "Interior"), (2, "Background")]:
        mask = lbl_crop == label_val
        if mask.sum() < 100:
            continue
        mgz = np.abs(gz[mask]).mean()
        mgy = np.abs(gy[mask]).mean()
        mgx = np.abs(gx[mask]).mean()
        grad_mag = np.sqrt(gz[mask]**2 + gy[mask]**2 + gx[mask]**2).mean()
        print(f"  {sid} {label_name:>10}: |gz|={mgz:.2f} |gy|={mgy:.2f} |gx|={mgx:.2f}  "
              f"mag={grad_mag:.2f}  gy/gx={mgy/mgx:.3f}  gz/gx={mgz/mgx:.3f}")

# === SHEET THICKNESS (how many voxels thick is each sheet?) ===
print("\n=== SHEET THICKNESS ===")
for sid in ['102536988', '2203617984', '105796630']:
    lbl_vol = tifffile.imread(os.path.join(base, 'train_labels', sid + '.tif'))
    sz = lbl_vol.shape[0]
    
    thicknesses = []
    for cx in range(sz//4, 3*sz//4, sz//4):
        for cy in range(sz//4, 3*sz//4, sz//4):
            lbl_line = lbl_vol[:, cx, cy]
            surf_z = np.where(lbl_line == 1)[0]
            if len(surf_z) > 0:
                # Group consecutive
                sheets = [[surf_z[0]]]
                for i in range(1, len(surf_z)):
                    if surf_z[i] - surf_z[i-1] > 3:
                        sheets.append([surf_z[i]])
                    else:
                        sheets[-1].append(surf_z[i])
                for s in sheets:
                    thicknesses.append(len(s))
    
    if thicknesses:
        t = np.array(thicknesses)
        print(f"  {sid}: n={len(t)}, mean={t.mean():.1f}, median={np.median(t):.0f}, "
              f"std={t.std():.1f}, range=[{t.min()}, {t.max()}]")
