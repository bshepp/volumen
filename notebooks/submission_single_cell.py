# Vesuvius Challenge - Surface Detection | Submission (paste this entire file into ONE Kaggle code cell)
# After pasting: set MODEL_DIR to your dataset path, e.g. MODEL_DIR = '/kaggle/input/your-dataset-slug'
# If your weight file is run1_best_model.pth, set MODEL_PATH = os.path.join(MODEL_DIR, 'run1_best_model.pth')

# Install imagecodecs first (needed for LZW TIFFs). In Kaggle, this line runs as a shell command.
!pip install imagecodecs -q

import os
import zipfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from scipy.ndimage import gaussian_filter
from scipy.ndimage import (
    binary_dilation, binary_erosion, generate_binary_structure,
    label as ndimage_label,
)
from scipy import ndimage

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')

# --- Configuration ---
DATA_DIR = '/kaggle/input/vesuvius-challenge-surface-detection'
MODEL_DIR = '/kaggle/input/vesuvius-model-weights'  # CHANGE THIS to your dataset slug
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')

PATCH_SIZE = 128
STRIDE = 64
USE_TTA = True
USE_POSTPROCESS = True

IN_CHANNELS = 6
NUM_CLASSES = 3
BASE_FILTERS = 32
DEPTH = 4

MIN_COMPONENT_SIZE = 500
BRIDGE_THRESHOLD = 3
MIN_SHEET_SPACING = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = device.type == 'cuda'
print(f'Device: {device}, AMP: {use_amp}')

# --- Feature Computation ---
def compute_features(volume, normalize=True):
    """Compute 6-channel feature volume from raw CT."""
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
    grad_mag = np.sqrt(gz**2 + gy**2 + gx**2)
    abs_gy = np.abs(gy)
    abs_gx = np.abs(gx)
    gy_gx_ratio = np.clip(abs_gy / (abs_gx + 1e-3), 0.0, 20.0)
    features = np.stack([raw_norm, log_s1, log_s2, hessian_trace, grad_mag, gy_gx_ratio], axis=0).astype(np.float32)
    if normalize:
        for c in range(features.shape[0]):
            mu = features[c].mean()
            std = features[c].std()
            if std > 1e-8:
                features[c] = (features[c] - mu) / std
            else:
                features[c] = features[c] - mu
    return features

# --- Model Definition ---
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 2, stride=2, bias=False),
            nn.InstanceNorm3d(in_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv = ConvBlock3D(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.down(x))

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2, bias=False)
        self.conv = ConvBlock3D(in_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            dz = skip.shape[2] - x.shape[2]
            dy = skip.shape[3] - x.shape[3]
            dx = skip.shape[4] - x.shape[4]
            x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2, dz//2, dz-dz//2])
        return self.conv(torch.cat([x, skip], dim=1))

class UNet3D(nn.Module):
    def __init__(self, in_channels=6, num_classes=3, base_filters=32, depth=4):
        super().__init__()
        self.depth = depth
        filters = [base_filters * (2**i) for i in range(depth)]
        self.init_conv = ConvBlock3D(in_channels, filters[0])
        self.encoders = nn.ModuleList([DownBlock(filters[i-1], filters[i]) for i in range(1, depth)])
        self.bottleneck = DownBlock(filters[-1], filters[-1]*2)
        self.decoders = nn.ModuleList()
        dec_in = filters[-1]*2
        for i in range(depth-1, -1, -1):
            self.decoders.append(UpBlock(dec_in, filters[i], filters[i]))
            dec_in = filters[i]
        self.final_conv = nn.Conv3d(filters[0], num_classes, 1)

    def forward(self, x):
        skips = []
        x = self.init_conv(x)
        skips.append(x)
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        x = self.bottleneck(x)
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i+1)])
        return self.final_conv(x)

# --- Post-Processing ---
def get_struct(conn=26):
    return generate_binary_structure(3, {6:1, 18:2, 26:3}[conn])

def remove_small_components(mask, min_size=500, conn=26):
    labeled, n = ndimage_label(mask, structure=get_struct(conn))
    if n == 0: return mask
    sizes = ndimage.sum(mask, labeled, range(1, n+1))
    keep = np.zeros_like(mask)
    for i, s in enumerate(sizes, 1):
        if s >= min_size: keep[labeled == i] = 1
    return keep.astype(np.uint8)

def remove_bridges(mask, thickness=3, conn=26):
    if mask.sum() == 0: return mask
    struct = get_struct(conn)
    eroded = binary_erosion(mask, structure=struct, iterations=thickness).astype(np.uint8)
    if eroded.sum() == 0: return mask
    labeled, n = ndimage_label(eroded, structure=struct)
    if n <= 1: return mask
    territories = np.zeros(mask.shape, dtype=np.int32)
    conflict = np.zeros(mask.shape, dtype=bool)
    for cid in range(1, n+1):
        dilated = binary_dilation((labeled==cid).astype(np.uint8), structure=struct, iterations=thickness+1)
        dilated = dilated & mask.astype(bool)
        conflict |= ((territories > 0) & dilated)
        territories[dilated & (territories == 0)] = cid
    result = mask.copy()
    result[conflict] = 0
    return remove_small_components(result, min_size=100, conn=conn)

def fill_small_holes(mask, max_size=200):
    inv = 1 - mask.astype(np.uint8)
    labeled, n = ndimage_label(inv, structure=generate_binary_structure(3, 1))
    result = mask.copy()
    for i in range(1, n+1):
        hole = labeled == i
        if hole.sum() > max_size: continue
        touches = hole[0].any() or hole[-1].any() or hole[:,0].any() or hole[:,-1].any() or hole[:,:,0].any() or hole[:,:,-1].any()
        if not touches: result[hole] = 1
    return result.astype(np.uint8)

def postprocess(pred, min_comp=500, bridge_t=3, min_spacing=10):
    pred = pred.astype(np.uint8)
    pred = remove_small_components(pred, min_comp)
    pred = remove_bridges(pred, bridge_t)
    pred = fill_small_holes(pred)
    return pred

# --- Sliding Window Inference + TTA ---
def sliding_window(model, features, patch_size=128, stride=64, device='cpu', use_amp=True, num_classes=3):
    model.eval()
    C, Z, Y, X = features.shape
    pad_z, pad_y, pad_x = max(0, patch_size-Z), max(0, patch_size-Y), max(0, patch_size-X)
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        features = np.pad(features, ((0,0),(0,pad_z),(0,pad_y),(0,pad_x)), mode='reflect')
    _, Zp, Yp, Xp = features.shape
    prob_sum = np.zeros((num_classes, Zp, Yp, Xp), dtype=np.float32)
    count = np.zeros((Zp, Yp, Xp), dtype=np.float32)
    zs = list(range(0, Zp-patch_size+1, stride))
    ys = list(range(0, Yp-patch_size+1, stride))
    xs = list(range(0, Xp-patch_size+1, stride))
    if zs[-1]+patch_size < Zp: zs.append(Zp-patch_size)
    if ys[-1]+patch_size < Yp: ys.append(Yp-patch_size)
    if xs[-1]+patch_size < Xp: xs.append(Xp-patch_size)
    total = len(zs)*len(ys)*len(xs)
    print(f'  {total} patches ({len(zs)}x{len(ys)}x{len(xs)})')
    with torch.no_grad():
        for z0 in zs:
            for y0 in ys:
                for x0 in xs:
                    p = features[:, z0:z0+patch_size, y0:y0+patch_size, x0:x0+patch_size]
                    pt = torch.from_numpy(p[np.newaxis]).float().to(device)
                    if use_amp and device.type == 'cuda':
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            logits = model(pt)
                    else:
                        logits = model(pt)
                    probs = F.softmax(logits, dim=1)[0].cpu().float().numpy()
                    prob_sum[:, z0:z0+patch_size, y0:y0+patch_size, x0:x0+patch_size] += probs
                    count[z0:z0+patch_size, y0:y0+patch_size, x0:x0+patch_size] += 1.0
    count = np.maximum(count, 1.0)
    return (prob_sum / count[np.newaxis])[:, :Z, :Y, :X]

def run_tta(model, features, patch_size=128, stride=64, device='cpu', use_amp=True):
    C, Z, Y, X = features.shape
    prob_sum = np.zeros((NUM_CLASSES, Z, Y, X), dtype=np.float32)
    for i, (fz, fy, fx) in enumerate([(a,b,c) for a in [0,1] for b in [0,1] for c in [0,1]]):
        print(f'TTA {i+1}/8: flip z={fz} y={fy} x={fx}')
        feat = features.copy()
        if fz: feat = np.flip(feat, 1)
        if fy: feat = np.flip(feat, 2)
        if fx: feat = np.flip(feat, 3)
        feat = np.ascontiguousarray(feat)
        probs = sliding_window(model, feat, patch_size, stride, device, use_amp, NUM_CLASSES)
        if fx: probs = np.flip(probs, 3)
        if fy: probs = np.flip(probs, 2)
        if fz: probs = np.flip(probs, 1)
        prob_sum += np.ascontiguousarray(probs)
    return prob_sum / 8

# --- Run Inference ---
t_start = time.time()

print('Loading model...')
model = UNet3D(IN_CHANNELS, NUM_CLASSES, BASE_FILTERS, DEPTH)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
if 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)
model = model.to(device)
model.eval()
print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')

import csv
with open(os.path.join(DATA_DIR, 'test.csv')) as f:
    test_ids = [row['id'] for row in csv.DictReader(f)]
print(f'Test IDs: {test_ids}')

predictions = {}
for test_id in test_ids:
    print(f'\n=== Processing {test_id} ===')
    vol_path = os.path.join(DATA_DIR, 'test_images', f'{test_id}.tif')
    volume = tifffile.imread(vol_path)
    print(f'Volume: {volume.shape}, dtype={volume.dtype}')
    print('Computing features...')
    t0 = time.time()
    features = compute_features(volume, normalize=True)
    print(f'Features computed in {time.time()-t0:.1f}s, shape={features.shape}')
    if USE_TTA:
        print('Running TTA inference...')
        probs = run_tta(model, features, PATCH_SIZE, STRIDE, device, use_amp)
    else:
        print('Running inference...')
        probs = sliding_window(model, features, PATCH_SIZE, STRIDE, device, use_amp, NUM_CLASSES)
    pred = (probs.argmax(axis=0) == 1).astype(np.uint8)
    print(f'Raw prediction: {pred.sum()} surface voxels ({pred.sum()/pred.size*100:.1f}%)')
    if USE_POSTPROCESS:
        print('Post-processing...')
        pred = postprocess(pred, MIN_COMPONENT_SIZE, BRIDGE_THRESHOLD, MIN_SHEET_SPACING)
        print(f'After post-processing: {pred.sum()} voxels ({pred.sum()/pred.size*100:.1f}%)')
    predictions[test_id] = pred

print(f'\nTotal inference time: {time.time()-t_start:.1f}s')

# --- Create Submission ---
os.makedirs('/kaggle/working/submission_files', exist_ok=True)
for test_id, pred in predictions.items():
    tif_path = f'/kaggle/working/submission_files/{test_id}.tif'
    tifffile.imwrite(tif_path, pred)
    print(f'Saved {tif_path}: shape={pred.shape}, dtype={pred.dtype}')
with zipfile.ZipFile('/kaggle/working/submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for test_id in predictions:
        tif_path = f'/kaggle/working/submission_files/{test_id}.tif'
        zf.write(tif_path, f'{test_id}.tif')
print(f'\nSubmission saved to /kaggle/working/submission.zip')
print(f'Total time: {time.time()-t_start:.1f}s')
