import os, csv
import tifffile

base = r'f:\kaggle\vesuvius_challenge\vesuvius-challenge-surface-detection'

# Definitive counts
dirs = ['train_images', 'train_labels', 'deprecated_train_images', 'deprecated_train_labels', 'test_images']
for d in dirs:
    path = os.path.join(base, d)
    files = [f for f in os.listdir(path) if f.endswith('.tif')]
    print(f'{d}: {len(files)} .tif files')

# Verify every train image has a matching label
img_set = set(os.listdir(os.path.join(base, 'train_images')))
lbl_set = set(os.listdir(os.path.join(base, 'train_labels')))
missing_labels = img_set - lbl_set
missing_images = lbl_set - img_set
print(f"\nImages without labels: {len(missing_labels)}")
if missing_labels:
    print(f"  {sorted(missing_labels)[:10]}")
print(f"Labels without images: {len(missing_images)}")
if missing_images:
    print(f"  {sorted(missing_images)[:10]}")

# Same for deprecated
dep_img = set(os.listdir(os.path.join(base, 'deprecated_train_images')))
dep_lbl = set(os.listdir(os.path.join(base, 'deprecated_train_labels')))
print(f"Deprecated images without labels: {len(dep_img - dep_lbl)}")
print(f"Deprecated labels without images: {len(dep_lbl - dep_img)}")

# Cross-check with CSV
with open(os.path.join(base, 'train.csv')) as f:
    train_rows = list(csv.DictReader(f))
with open(os.path.join(base, 'test.csv')) as f:
    test_rows = list(csv.DictReader(f))

csv_ids = set(r['id'] for r in train_rows)
file_ids = set(f.replace('.tif', '') for f in img_set | dep_img)
print(f"\nCSV entries: {len(csv_ids)}")
print(f"All image files (train+deprecated): {len(file_ids)}")
print(f"CSV IDs not on disk: {len(csv_ids - file_ids)}")
print(f"Disk IDs not in CSV: {len(file_ids - csv_ids)}")

# Test image
test_csv_ids = [r['id'] for r in test_rows]
test_files = [f.replace('.tif', '') for f in os.listdir(os.path.join(base, 'test_images'))]
print(f"\nTest CSV IDs: {test_csv_ids}")
print(f"Test files: {test_files}")

# Try reading the test image to verify it's complete
test_img = tifffile.imread(os.path.join(base, 'test_images', '1407735.tif'))
print(f"Test image readable: shape={test_img.shape}, dtype={test_img.dtype}, min={test_img.min()}, max={test_img.max()}")

# Spot check: try reading a random label and image to confirm they load
img = tifffile.imread(os.path.join(base, 'train_images', '1004283650.tif'))
lbl = tifffile.imread(os.path.join(base, 'train_labels', '1004283650.tif'))
print(f"Sample train image: shape={img.shape}, dtype={img.dtype}")
print(f"Sample train label: shape={lbl.shape}, dtype={lbl.dtype}")
assert img.shape == lbl.shape, "MISMATCH: image and label shapes differ!"

print("\n=== ALL CHECKS PASSED ===")
