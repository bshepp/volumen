import tifffile, numpy as np

base = r'f:\kaggle\vesuvius_challenge\vesuvius-challenge-surface-detection'

# Load one sample and analyze periodicity
img = tifffile.imread(base + r'\train_images\102536988.tif')
lbl = tifffile.imread(base + r'\train_labels\102536988.tif')

print(f"Volume shape: {img.shape}")

# === 1. Z-axis profiles at multiple x,y locations ===
print("\n=== Z-AXIS PERIODICITY (sheet spacing) ===")
for name, cx, cy in [("center", 160, 160), ("q1", 80, 80), ("q3", 240, 240), ("mid-left", 160, 80)]:
    profile = img[:, cx, cy].astype(float)
    lbl_line = lbl[:, cx, cy]
    
    # Where are surfaces?
    surf_z = np.where(lbl_line == 1)[0]
    papyrus_z = np.where(lbl_line == 0)[0]
    
    # Find sheet boundaries (clusters of surface voxels)
    if len(surf_z) > 0:
        # Group consecutive surface voxels into sheets
        sheet_starts = [surf_z[0]]
        for i in range(1, len(surf_z)):
            if surf_z[i] - surf_z[i-1] > 3:
                sheet_starts.append(surf_z[i])
        print(f"  {name} ({cx},{cy}): {len(sheet_starts)} sheets detected at z={sheet_starts}")
        if len(sheet_starts) > 1:
            spacings = [sheet_starts[i+1] - sheet_starts[i] for i in range(len(sheet_starts)-1)]
            print(f"    Sheet spacings: {spacings}, mean={np.mean(spacings):.1f}")
    else:
        print(f"  {name} ({cx},{cy}): no surface voxels on this line")
    
    # FFT
    fft = np.abs(np.fft.rfft(profile - profile.mean()))
    freqs = np.fft.rfftfreq(len(profile))
    top_idx = np.argsort(fft[1:])[-3:] + 1
    top_freqs = [(freqs[i], 1.0/freqs[i] if freqs[i] > 0 else 0, fft[i]) for i in sorted(top_idx, key=lambda x: fft[x], reverse=True)]
    print(f"    Top FFT: " + ", ".join(f"period={p:.0f}vox mag={m:.0f}" for f,p,m in top_freqs))

# === 2. Hessian-based sheet detection ===
print("\n=== HESSIAN EIGENVALUE ANALYSIS (sheet-like structures) ===")
from scipy.ndimage import gaussian_filter
# Compute Hessian at a small region
sub = img[140:180, 140:180, 140:180].astype(float)
sub_smooth = gaussian_filter(sub, sigma=1.5)

# Approximate Hessian via finite differences
dzz = np.diff(sub_smooth, n=2, axis=0)
dyy = np.diff(sub_smooth, n=2, axis=1)
dxx = np.diff(sub_smooth, n=2, axis=2)
# Crop to common size
s = min(dzz.shape[0], dyy.shape[1], dxx.shape[2])
dzz = dzz[:s, :s, :s]
dyy = dyy[:s, :s, :s]
dxx = dxx[:s, :s, :s]

# At each voxel, sheet-like = two large eigenvalues, one small
# Simple proxy: |dzz + dyy| >> |dxx| or similar
print(f"  Hessian component stats (40^3 subregion):")
print(f"    dzz: mean={dzz.mean():.2f}, std={dzz.std():.2f}")
print(f"    dyy: mean={dyy.mean():.2f}, std={dyy.std():.2f}")
print(f"    dxx: mean={dxx.mean():.2f}, std={dxx.std():.2f}")

# === 3. Directional texture analysis (recto vs verso fiber orientation) ===
print("\n=== DIRECTIONAL TEXTURE (fiber orientation) ===")
# Look at a thin slab where label=1 (surface)
# Get a z-slice that has label=1 voxels
z_mid = 160
xy_slice = img[z_mid]
xy_label = lbl[z_mid]
surf_mask = xy_label == 1
papyrus_mask = xy_label == 0
bg_mask = xy_label == 2

# Compute local gradient magnitude in x vs y
if surf_mask.sum() > 100:
    gy = np.abs(np.diff(xy_slice.astype(float), axis=0))
    gx = np.abs(np.diff(xy_slice.astype(float), axis=1))
    # Crop masks to match
    surf_y = surf_mask[:-1, :]
    surf_x = surf_mask[:, :-1]
    pap_y = papyrus_mask[:-1, :]
    pap_x = papyrus_mask[:, :-1]
    
    print(f"  At z={z_mid}:")
    print(f"    Surface voxels: {surf_mask.sum()}")
    if surf_y.sum() > 0 and surf_x.sum() > 0:
        gy_surf = gy[surf_y].mean()
        gx_surf = gx[surf_x].mean()
        print(f"    Surface: mean |gy|={gy_surf:.2f}, mean |gx|={gx_surf:.2f}, ratio={gy_surf/gx_surf:.2f}")
    if pap_y.sum() > 0 and pap_x.sum() > 0:
        gy_pap = gy[pap_y].mean()
        gx_pap = gx[pap_x].mean()
        print(f"    Papyrus: mean |gy|={gy_pap:.2f}, mean |gx|={gx_pap:.2f}, ratio={gy_pap/gx_pap:.2f}")

# === 4. Autocorrelation for sheet spacing ===
print("\n=== AUTOCORRELATION (sheet spacing regularity) ===")
# Take a line through the volume perpendicular to sheets
profile = img[:, 160, 160].astype(float)
profile -= profile.mean()
autocorr = np.correlate(profile, profile, mode='full')
autocorr = autocorr[len(autocorr)//2:]  # positive lags only
autocorr /= autocorr[0]  # normalize

# Find peaks in autocorrelation (periodic sheet spacing)
from scipy.signal import find_peaks
peaks, props = find_peaks(autocorr, height=0.1, distance=5)
print(f"  Autocorrelation peaks at lags: {peaks[:10].tolist()}")
print(f"  Peak heights: {[f'{autocorr[p]:.3f}' for p in peaks[:10]]}")
if len(peaks) > 1:
    peak_spacings = np.diff(peaks[:10])
    print(f"  Peak-to-peak spacings: {peak_spacings.tolist()}")

# === 5. Check multiple samples for consistency ===
print("\n=== SHEET SPACING ACROSS SAMPLES ===")
import csv, os
with open(os.path.join(base, 'train.csv')) as f:
    rows = {r['id']: r['scroll_id'] for r in csv.DictReader(f)}

samples = ['102536988', '1004283650', '105796630', '1240194203', '2203617984']
for sid in samples:
    lbl_vol = tifffile.imread(os.path.join(base, 'train_labels', sid + '.tif'))
    scroll = rows.get(sid, '?')
    
    # Count sheets along a central z-line
    lbl_line = lbl_vol[:, lbl_vol.shape[1]//2, lbl_vol.shape[2]//2]
    surf_z = np.where(lbl_line == 1)[0]
    if len(surf_z) > 0:
        sheet_starts = [surf_z[0]]
        for i in range(1, len(surf_z)):
            if surf_z[i] - surf_z[i-1] > 3:
                sheet_starts.append(surf_z[i])
        spacings = [sheet_starts[i+1] - sheet_starts[i] for i in range(len(sheet_starts)-1)] if len(sheet_starts) > 1 else []
        print(f"  {sid} (scroll {scroll}): {len(sheet_starts)} sheets, spacings={spacings}")
    else:
        print(f"  {sid} (scroll {scroll}): no surface on center line")
