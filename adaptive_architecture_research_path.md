# Learnable Architecture Parameters for Volumetric Scroll Segmentation

**A Future Research Roadmap: From Fixed to Adaptive 3D U-Net Architectures**

Brian Sheppard | February 2026

> **Status:** This document is a research roadmap for future work. None of these experiments are currently in progress. V2 completed 200 epochs (best val_loss=0.6728) and scored 0.390/0.409 (public/private) on Kaggle. V3 paused at ~53 epochs due to NaN divergence (best val_loss=0.7016). All AWS instances terminated Feb 27, 2026. A sibling project (`../vesuvius_sinogram/`) explores sinogram-domain detection as an alternative research direction. See `PIPELINES.md` for full pipeline status and `CLAUDE.md` for checkpoint locations.

---

## 1. Motivation and Context

The three pipeline versions (V1–V3) built for the Kaggle Vesuvius Challenge demonstrate that architectural choices have outsized impact on segmentation quality. The jump from V1 (standard 3D U-Net with CE+Dice loss) to V2 (deep supervision with Focal+Dice and SkeletonRecall) yielded a 2x improvement in surface Dice. V3's multi-scale fusion approach achieved V1's 200-epoch performance in just 20 epochs.

These results confirm that scroll volume data -- with its heterogeneous structure across spatial scales, variable ink density, and depth-dependent characteristics -- requires architectures that can adapt their behavior regionally. Currently, architectural parameters (network depth, kernel sizes, attention patterns, loss weights) are set statically by the researcher. This document outlines a future path toward making them learnable.

### 1.1 Baseline Performance (as of February 9, 2026)

All pipelines use the same validation split (scroll ID `26002`, 82 samples) and identical evaluation metrics. See `PIPELINES.md` for full architecture details and `docs/archive/AWS_RUN1_REPORT.md` for V1 run history.

| | V1 (`src/`) | V2 (`src_v2/`) | V3 (`src_v3/`) |
|---|---|---|---|
| Architecture | Standard 3D U-Net | U-Net + Deep Supervision | Multi-Scale Fusion (3x UNet) |
| Parameters | 27M | ~27M | ~15M |
| Loss Function | CE+Dice, clDice, Boundary | Focal+Dice, SkeletonRecall, Boundary (4 scales) | Focal+Dice, SkeletonRecall, Boundary |
| Post-processing | CC filter, bridges, holes, spacing | **1st place pipeline** (closing, patching, LUT, fill) | CC filter, bridges, holes, spacing |
| GPU | T4 (g4dn.xlarge) | T4 (g4dn.xlarge) | A10G (g5.xlarge) |
| Status | **Completed** (200 epochs, frozen) | **Completed** (200 epochs) | **Paused** (~53 epochs, NaN at ~55) |
| Best Val Loss | 1.3639 | **0.6728** | 0.7016 |
| Best Surface Dice | 0.1162 | **0.2538** | 0.2258 |
| Kaggle Score | — | **0.390 / 0.409** (pub/priv) | — |
| Epoch Time | ~60 min | ~57 min | ~134 min |

> **Key observation:** V3 with 15M parameters learns faster per epoch than V2 with 27M parameters, suggesting the multi-scale fusion architecture provides a better inductive bias for scroll data. This motivates the question: what other architectural decisions could be made learnable to further improve this bias?

---

## 2. Taxonomy of Learnable Architectural Parameters

Each parameter below is fixed at design time in V1–V3. This section categorizes them by what they control, how they could be made learnable, and their expected impact on scroll segmentation.

### 2.1 Parameter Overview

| Parameter | Controls | Learning Method | Param Cost | Train Cost | Scroll-Specific Rationale |
|---|---|---|---|---|---|
| Network Depth | Encoder/decoder levels | Stochastic depth, early exit, progressive growing | Low | Medium | Different regions need different abstraction levels |
| Channel Attention | Feature map importance | SE blocks, ECA-Net | Very Low | Very Low | Ink vs papyrus vs void activate different channels |
| Spatial Attention | Where to focus per patch | CBAM, self-attention, deformable attention | Low–Med | Medium | Scroll regions vary enormously in information density |
| Kernel Size | Receptive field per layer | SKNet (selective kernel) | Medium | Medium | Ink strokes vs sheet boundaries need different RF |
| Skip Connection Weight | Low-level vs high-level feature balance | Learned gating (sigmoid gates on skip paths) | Very Low | Very Low | Texture detail vs semantic structure tradeoff |
| Loss Component Weights | Training objective emphasis | Uncertainty weighting (Kendall et al.), GradNorm | Very Low | Low | Focal vs Dice vs Skeleton importance varies by region |
| Feature Resolution Routing | Which scale branch to prioritize | Attention-based fusion (extends V3's design) | Low | Low | Direct extension of V3 multi-scale; per-region routing |
| Normalization Strategy | Feature distribution handling | Switchable Norm, NAS-style search | Low | High (NAS) | Batch statistics vary with scroll region heterogeneity |
| Activation Functions | Nonlinearity shape | PReLU, adaptive piecewise linear, KAN-style | Very Low | Very Low | Low expected scroll-specific impact |
| Fusion Decoder Complexity | How multi-scale features combine | Replace 1×1×1 with 3D conv stack or cross-attention | Medium | Medium | V3's 1×1×1 can't model spatial cross-scale interactions |

---

## 3. Proposed Methodology: Vector Ablation

The proposed approach treats each learnable parameter as an independent vector. Each modification would be tested in isolation against a common baseline (a completed V2 or V3 checkpoint), with performance measured via validation loss and surface Dice on scroll 26002. Only modifications that demonstrate improvement would be combined in subsequent experiments. This is ablation study in reverse -- constructive rather than deconstructive.

> **Methodological advantage:** Serialized experimentation forces interpretability. Each modification's contribution is understood individually before combination. This produces genuine architectural intuition about what matters for volumetric scroll segmentation -- knowledge that brute-force combinatorial search at scale does not yield.

### 3.1 Validation Protocol

All experiments must use the same validation split (scroll 26002) and evaluation metrics (val loss, surface Dice) as the current V1–V3 pipelines. Changing the validation split mid-experiment would invalidate comparisons. Any future cross-validation or holdout changes should be documented as a deliberate protocol revision, not mixed into ongoing comparisons.

**Known limitation:** Single-scroll validation means all architectural decisions are optimized for scroll 26002's characteristics (its ink density, damage patterns, sheet geometry). Improvements on 26002 may not generalize to the hidden Kaggle test scrolls if those scrolls differ substantially. This is an accepted tradeoff -- consistent evaluation is more valuable during development than premature diversification -- but it means final Kaggle submission scores could diverge from validation scores. If experiments plateau or Kaggle scores disappoint despite strong validation, revisiting the validation protocol (e.g., k-fold across scrolls, or rotating the holdout scroll) should be the first diagnostic step.

---

## 4. Prioritized Experiment Plan

Experiments are ordered by expected impact-to-cost ratio. All experiments would fine-tune from completed V2 or V3 checkpoints (approximately 20–30 epochs needed to assess impact, not a full 200-epoch run).

> **Prerequisite:** V2 has completed 200 epochs. V3 paused at ~53 epochs (NaN divergence) — its checkpoint can serve as a partial baseline. These experiments require stable, fully-trained baselines.

### Tier 1: High Impact, Low Cost

These modifications add minimal parameters and training overhead while addressing known limitations in the current architectures.

#### Experiment 1A: Skip Connection Gating

**Baseline:** Completed V2 checkpoint. **Modification:** Add a sigmoid gate (single conv + sigmoid) on each skip connection. The gate learns a per-spatial-location weight between 0 and 1, controlling how much low-level detail passes through versus how much the decoder relies on deep semantic features.

**Rationale:** Standard U-Net skip connections pass all features equally. In scroll volumes, some regions (clear ink on papyrus) benefit from fine texture detail while others (damaged or ambiguous areas) need the decoder to rely on high-level context. Cost is near zero -- one additional convolution per skip path.

**Expected training overhead:** <1% additional time per epoch.
**Success criterion:** Improvement in surface Dice within 30 fine-tuning epochs.
**Pipeline note:** Would be implemented as a new pipeline (V4 or similar) following the isolation pattern in `PIPELINES.md`.

#### Experiment 1B: Channel Attention (SE Blocks)

**Baseline:** Completed V2 checkpoint. **Modification:** Insert Squeeze-and-Excitation blocks after each encoder and decoder conv block. SE blocks perform global average pooling across spatial dimensions, pass through a two-layer MLP bottleneck, and produce per-channel scaling factors.

**Rationale:** Not all feature maps are equally relevant at every level. Channel attention lets the model suppress noise channels and amplify discriminative ones. Well-established across 2D segmentation; 3D extension is straightforward.

**Expected training overhead:** <2% additional time per epoch. Negligible parameter increase with reduction ratio of 16.

#### Experiment 1C: Learned Loss Weighting

**Baseline:** Completed V2 or V3 checkpoint. **Modification:** Replace fixed loss component weights with learnable log-variance parameters following Kendall et al. (2018) multi-task uncertainty weighting. Each loss component (Focal, Dice, SkeletonRecall, Boundary) gets a learned weight that adapts during training based on each task's homoscedastic uncertainty.

**Rationale:** The jump from V1's loss to V2/V3's loss was the single biggest performance driver. The current fixed weights (0.3/0.3/0.2 in `CompositeLossV3`, similar in `DeepSupCompositeLoss`) are hand-tuned. Letting the model learn optimal weighting removes this manual step and allows the balance to shift as training progresses (e.g., emphasizing boundary early, skeleton recall later).

**Expected training overhead:** Effectively zero. Adds only 4 scalar parameters.

> **Note:** Of the three Tier 1 experiments, this one has the highest expected value. The loss function change from V1 to V2 was the single biggest performance driver in the project so far. Making the loss weights adaptive is a natural next step.

### Tier 2: Moderate Cost, High Potential

#### Experiment 2A: Spatial Attention (CBAM or Lightweight Self-Attention)

**Modification:** Add spatial attention modules that produce per-voxel importance maps. CBAM is the lower-cost option (channel + spatial attention in sequence). Lightweight self-attention (e.g., linear attention or windowed attention) is more powerful but costlier in 3D.

**Rationale:** Scroll patches contain highly variable information density. Large void regions, uniform papyrus, and ink-bearing surfaces coexist within single patches. Spatial attention allows the model to allocate computational focus where it matters.

**Expected training overhead:** 5–15% depending on attention variant. Memory is the primary constraint in 3D.

#### Experiment 2B: Improved Fusion Decoder (V3-Specific)

**Modification:** Replace V3's `Conv3d(9, 3, 1)` fusion layer (see `src_v3/model.py`) with a small 3D convolutional decoder (two 3x3x3 conv layers with residual connection) or a cross-attention module that attends across scale branches.

**Rationale:** The 1x1x1 fusion is a learned weighted average -- it cannot capture how features from different scales relate spatially. A richer decoder lets the model learn context-dependent fusion (e.g., trusting fine-scale features near ink boundaries, coarse-scale features in ambiguous regions).

**Expected training overhead:** 10–20% additional epoch time. Parameter increase moderate (few hundred K).

> **Caveat:** V3 trained ~53 epochs before hitting NaN. The fusion layer's limitations may not be the binding constraint until the branch UNets have matured. If V3 is resumed and trained to completion, revisit this experiment.

#### Experiment 2C: Feature Resolution Routing

**Modification:** Extend V3's multi-scale design with a lightweight routing network that produces per-voxel soft weights over the three resolution branches before fusion. This is distinct from the fusion decoder—it controls which branch is prioritized, while the decoder controls how they combine.

**Rationale:** Currently V3 fuses all three scales uniformly everywhere. Some regions are best served by the 128-resolution branch (fine detail), others by the 32-resolution branch (large-scale structure). Per-voxel routing lets the model specialize.

### Tier 3: Higher Cost, Fundamental (Future Work)

#### Experiment 3A: Learnable Network Depth

**Modification:** Implement early-exit mechanisms or stochastic depth where the decoder can produce predictions at multiple encoder levels. During inference, a learned confidence gate determines when sufficient depth has been reached. Alternative: progressive growing where encoder/decoder levels are added during training.

**Rationale:** Synergizes with V2's deep supervision, which already produces predictions at multiple scales. Making depth adaptive rather than fixed could reduce inference cost on simple regions while allocating full depth to complex ones.

#### Experiment 3B: Selective Kernel Networks

**Modification:** Replace fixed 3×3×3 convolutions with SKNet modules that dynamically select between 3×3×3, 5×5×5, and 7×7×7 kernels (or equivalent dilated convolutions) at each layer based on input content.

**Rationale:** Fine ink strokes need small receptive fields; sheet boundaries and large structural features need larger ones. Currently this is partially addressed by V3's multi-scale branches, but selective kernels operate at per-layer granularity rather than per-branch.

#### Experiment 3C: Normalization and Activation Search

**Modification:** Implement switchable normalization (batch norm, group norm, instance norm selectable per layer) and learnable activation functions (PReLU or adaptive piecewise linear). These are best explored via NAS-style search or manual hyperparameter sweeps rather than end-to-end gradient learning.

**Rationale:** Expected impact is lower than other parameters for this specific domain, but normalization choice can interact with batch size constraints (relevant on T4 and A10G GPUs). Lower priority unless other experiments plateau.

---

## 5. Combination Strategy

After individual vector testing, winning modifications are combined following these principles:

**Orthogonal combinations first.** Modifications that control different aspects of the architecture (e.g., skip gating + channel attention + loss weighting) are most likely to provide additive benefit. Combine these before trying modifications that overlap (e.g., spatial attention + feature routing, which both control spatial focus).

**Measure interaction effects.** When combining two winning modifications, the combined improvement may be more or less than the sum of individual improvements. Track whether combinations are superadditive (synergistic) or subadditive (redundant).

**Respect the compute budget.** Each combination adds training overhead. On current hardware (T4 on g4dn.xlarge at ~$0.53/hr, A10G on g5.xlarge at ~$1.01/hr), total epoch time should not exceed 3x the baseline. If it does, prune the least impactful modification.

### 5.1 Recommended Combination Order

| Round | Combination | Rationale | Max Overhead |
|---|---|---|---|
| R1 | Best of {1A, 1B, 1C} | Single best low-cost mod | <5% |
| R2 | R1 winner + second-best of {1A, 1B, 1C} | Stack orthogonal Tier 1 mods | <10% |
| R3 | R2 winner + all three Tier 1 | Full Tier 1 stack | <15% |
| R4 | R3 + best Tier 2 modification | Add structure-changing mod | <30% |
| R5 | R4 + second Tier 2 (if budget allows) | Explore interaction effects | <50% |

---

## 6. Ensemble and Submission Strategy

The project now has three architecturally distinct trained models (V1 complete, V2 complete, V3 partial). Regardless of individual performance, these models likely make different types of errors due to their structural differences. Ensembling strategies to explore:

**Simple averaging** of softmax predictions. Baseline ensemble approach -- often surprisingly competitive. All three pipelines already include sliding window inference with TTA (`src/inference.py`, `src_v2/inference.py`, `src_v3/inference.py`), so the per-model predictions are already available.

**Learned weighted averaging** where per-model weights are optimized on the validation set. Simple logistic regression on the three models' predictions.

**Region-dependent ensembling** where model weights vary spatially. This aligns with the adaptive architecture thesis -- different models may excel in different scroll regions.

**Test-time augmentation (TTA)** with flips and rotations averaged per model before ensembling. Already implemented in all three inference scripts.

> **Practical note:** Ensembling V1+V2+V3 is likely the fastest path to a competition score improvement and could be attempted before any of the architecture experiments in this document. It requires no new training -- just running inference with each model and averaging.

---

## 7. Long-Term Vision: Fully Adaptive Segmentation

The ultimate direction of this research is a segmentation architecture where no structural decisions are fixed at design time. Network depth, receptive field, attention patterns, loss emphasis, and feature routing all adapt per-region based on the input characteristics. This is not NAS (which searches once then fixes), but continuous architectural adaptation at inference time.

The Vesuvius scroll data is an ideal testbed for this concept because it exhibits extreme heterogeneity within a single dataset: damaged vs preserved regions, clear vs faint ink, single vs overlapping sheets, surface vs interior layers. A fixed architecture is always a compromise across these conditions. An adaptive architecture can specialize.

Each experiment in this document is a step toward that vision, testing whether a specific architectural degree of freedom benefits from being learned rather than prescribed. The vector ablation methodology ensures each step is interpretable and that the final architecture is understood, not just optimized.

---

## References

All project paths are relative to the repository root. If files have been moved or renamed since this document was written, check the repo root for current locations. `PIPELINES.md` is the canonical reference for pipeline structure and should be updated first if the project layout changes.

- **Project documentation:** `PIPELINES.md`, `CLAUDE.md`, `AWS_TRAINING.md`, `docs/archive/AWS_RUN1_REPORT.md`
- **Pipeline source:** `src/` (V1, frozen), `src_v2/` (V2, completed), `src_v3/` (V3, paused at ~53 epochs)
- **Kendall et al.** "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (CVPR 2018)
- **Hu et al.** "Squeeze-and-Excitation Networks" (CVPR 2018)
- **Woo et al.** "CBAM: Convolutional Block Attention Module" (ECCV 2018)
- **Li et al.** "Selective Kernel Networks" (CVPR 2019)

---

*Future research roadmap for the Volumen project. Not currently in progress. Last updated: February 28, 2026.*
