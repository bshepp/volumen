# Submitting V1 as a Pipeline Test (Deadline: Feb 27, 2026)

## Where the details live

| What | Location |
|------|----------|
| **Rules, format, metric** | **Project:** `COMPETITION_NOTES.md` (deadline, submission format, metric formula, label semantics) |
| **Kaggle competition** | https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection |
| **Kaggle data** | https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/data |
| **Metric demo** | https://www.kaggle.com/code/sohier/vesuvius-2025-metric-demo |
| **Pipeline / code** | `PIPELINES.md`, `notebooks/submission.ipynb`, `AWS_TRAINING.md` (after-training steps) |

So: **rules and pertinent info are in the project folder** (`COMPETITION_NOTES.md`) and on Kaggle (links above). Use the project for format/rules, Kaggle for submitting and scoring.

---

## What you need to submit V1

- **Checkpoint:** V1 best model (e.g. `outputs_aws/run1_best_model.pth`).
- **Notebook:** `notebooks/submission.ipynb` (self-contained: features, V1 UNet3D, sliding window, TTA, post-process, `submission.zip`).
- **Submission format:** `submission.zip` with one `.tif` per test image, `[image_id].tif`, uint8, same shape as source. Code competition: notebook runs on Kaggle, produces the zip.

---

## Steps to submit V1 (pipeline test)

1. **Upload the V1 weights to Kaggle as a Dataset**
   - Go to [Kaggle Datasets](https://www.kaggle.com/datasets), **New Dataset**.
   - Upload `outputs_aws/run1_best_model.pth`.
   - **Important:** Kaggle will show a single file. The notebook expects the weight file as `best_model.pth` inside the dataset. So either:
     - **Option A:** When creating the dataset, name the file `best_model.pth` (or add a single file and name it `best_model.pth`), **or**
     - **Option B:** In the notebook config cell, set `MODEL_PATH = os.path.join(MODEL_DIR, 'run1_best_model.pth')` and add the file as `run1_best_model.pth`.
   - Note the dataset name (e.g. `vesuvius-model-weights` or `vesuvius-v1-run1`).

2. **Create a new Kaggle Notebook for the competition**
   - Competition: [Vesuvius Challenge - Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection).
   - **Add datasets:** competition data + your model-weights dataset (the one from step 1).
   - Copy the contents of `notebooks/submission.ipynb` into the Kaggle notebook (or upload the notebook and adapt paths).

3. **Set paths in the notebook**
   - `DATA_DIR = '/kaggle/input/vesuvius-challenge-surface-detection'` (default).
   - `MODEL_DIR = '/kaggle/input/<your-dataset-slug>'` (e.g. `'/kaggle/input/vesuvius-v1-run1'`).
   - `MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')` — or `'run1_best_model.pth'` if you kept that name.

4. **Match V1 architecture (already set in the notebook)**
   - `IN_CHANNELS = 6`, `NUM_CLASSES = 3`, `BASE_FILTERS = 32`, `DEPTH = 4`. Do not change these for the run1 checkpoint.

5. **Run the notebook**
   - Enable **GPU** (faster inference).
   - Run all cells. It will load the model, run test-volume inference (with TTA if enabled), post-process, write `.tif` files, then create `submission.zip` in `/kaggle/working/`.

6. **Submit**
   - **Submit** the notebook. Kaggle will use the `submission.zip` it produces.
   - Limits: max 3 submissions/day; you can select up to 2 final submissions before the deadline.

---

## Quick checklist

- [ ] `run1_best_model.pth` (or `best_model.pth`) in a Kaggle dataset.
- [ ] Notebook added to competition; competition data + model dataset attached.
- [ ] `MODEL_DIR` and `MODEL_PATH` point to that dataset and file.
- [ ] Config: `BASE_FILTERS=32`, `DEPTH=4` (no change).
- [ ] GPU on, run all → `submission.zip` in `/kaggle/working/`.
- [ ] Submit before **Feb 27, 2026**.

That’s all you need to do to submit V1 as a pipeline test.
