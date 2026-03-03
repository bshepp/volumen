# Sinogram-Domain Approach to Touching Sheet Separation

## Problem Statement

The Vesuvius Challenge CT reconstruction pipeline converts raw X-ray detector sinograms into a discrete voxel volume via filtered back-projection (or iterative reconstruction). This reconstruction step makes assumptions about the underlying continuous density field — typically nearest-neighbor or trilinear interpolation onto a regular grid. When two papyrus sheets are in contact or separated by less than the effective voxel pitch, the reconstruction smears their signals together. The resulting voxel representation destroys the angular information that originally distinguished the two sheets in the raw projection data.

Every downstream ML approach — nnU-Net segmentation, ink detection models, post-processing — inherits this information loss. The touching sheet problem is not a model problem. It is a reconstruction artifact.

## Core Insight

In the sinogram domain (raw detector output as a function of detector position and rotation angle), two sheets at slightly different radial positions trace distinct trajectories through projection space. Their attenuation contributions arrive at different detector elements at different angles. This separability exists in the raw data even when the sheets are in physical contact, because the X-ray source illuminates them from different directions as it rotates.

Once reconstructed into a voxel grid, this angular separability is collapsed into a single spatial neighborhood and lost. The approach described here operates upstream of that collapse.

## Approach

### 1. Identify Regions of Interest

Not every region of the scroll requires sinogram-domain processing. Use the existing reconstructed volume and current segmentation outputs to identify specifically where sheets are touching, ambiguous, or poorly resolved. These are the only regions that need this treatment. This keeps computational cost bounded and avoids reprocessing regions where conventional reconstruction already works.

Candidate identification methods:
- Segmentation confidence maps: low-confidence regions from nnU-Net or similar
- Surface mesh proximity: where adjacent segment meshes approach within 1-3 voxels
- Ink detection disagreement zones: where multiple models disagree on ink presence
- Manual annotation of known problematic regions

### 2. Continuous Basis Function Testing (Synthetic Testbed)

Before touching real sinogram data, build a synthetic evaluation framework:

**a. Ground truth construction**

Take known touching sheet regions from the reconstructed volume where we have reasonable (if imperfect) ground truth segmentations. Define two closely spaced surfaces with known geometry — parallel sheets, sheets converging to contact, sheets crossing.

**b. Forward projection with variable basis functions**

Represent the density field using different continuous interpolation schemes and simulate synthetic sinograms via forward projection (ray tracing through the continuous field):

- Trilinear interpolation (baseline — what reconstruction typically assumes)
- Tricubic B-spline
- Gaussian radial basis functions (variable width)
- Thin-plate splines
- Wendland compactly-supported RBFs
- Neural implicit representations (small MLP mapping position → density)

For each basis function, forward-project to generate a synthetic sinogram, then measure **inter-sheet discriminability** in the projection domain:

- Can you identify two distinct attenuation profiles at angles where the sheets are geometrically separable?
- At what sheet separation does each basis function lose the ability to distinguish the two contributions?
- How does the angular sampling rate interact with basis function choice?

**c. Metrics**

- Minimum resolvable sheet separation per basis function (analogous to radar range resolution)
- Signal-to-interference ratio between sheet contributions in projection space
- Reconstruction fidelity when projecting back from the synthetic sinogram

This gives a principled selection criterion for which continuous representation to use before committing to processing real data.

### 3. Sinogram-Domain Sheet Separation

With the best-performing basis function identified:

**a. Extract local sinogram patches**

For each identified region of interest, extract the corresponding subset of the raw sinogram data — only the detector elements and rotation angles whose ray paths intersect the problematic region.

**b. Continuous point reconstruction**

Instead of reconstructing onto a regular voxel grid, reconstruct onto a continuous point representation:

- Define evaluation points along candidate sheet surfaces (seeded from existing segmentations)
- For each point, solve for the local density by back-projecting only the relevant sinogram contributions
- Allow the point positions to move (optimize) to find the configuration that best explains the observed sinogram data as two distinct sheets

This is essentially fitting a continuous surface model directly to the raw projection data in the regions where the discrete reconstruction failed.

**c. Iterative refinement**

- Start with initial surface estimates from existing segmentations
- Refine point positions and density values to minimize reprojection error against the actual sinogram
- Enforce physical priors: sheets have roughly constant thickness, smooth surfaces, bounded density values
- The optimization naturally separates touching sheets because two surfaces explain the angular variation in the sinogram better than one thick surface

### 4. Integration with Existing Pipeline

The sinogram-domain processing only replaces the reconstruction in identified problem regions. The output is a refined local density field and updated surface mesh positions that can be:

- Stitched back into the full volume for downstream ink detection
- Used directly to extract texture images along the corrected surfaces
- Fed back into the segmentation pipeline as improved priors

## Why This Works (Signal Chain Perspective)

The CT scanning hardware captures a continuous analog signal: X-ray attenuation as a function of position and angle. The detector digitizes this (ADC), the reconstruction algorithm interpolates and grids it, and the ML pipeline further processes the discrete output. Information is lost at every stage of this chain.

The touching sheet problem occurs because the reconstruction stage's spatial discretization (voxel grid) cannot represent two surfaces closer than ~1 voxel apart. But the angular information that distinguishes them was present in the analog signal and survives digitization — the detector has sufficient angular sampling. The information loss happens specifically at the reconstruction-to-grid step.

By going back to the sinogram and reconstructing in continuous space only where needed, we recover information that the standard pipeline provably destroys.

## RESIDUALS Integration (Point Cloud Analysis)

### Adapting RESIDUALS for CT Point Clouds

RESIDUALS was designed to detect subtle features in LiDAR point clouds by extracting residuals after removing dominant surface signals. The same principle applies to CT-derived point clouds: fit the dominant sheet surface, remove it, and look for coherent structure in the residuals that indicates a second sheet.

However, CT point clouds differ from LiDAR in a critical way: LiDAR point clouds represent surfaces viewed from above, with generally one surface per XY position (plus canopy returns). CT-derived point clouds from touching sheet zones have points stacked directly on top of each other — two legitimate surfaces occupying nearly the same spatial position with minimal separation. RESIDUALS likely assumes that a local point neighborhood represents a single surface with noise, not two superimposed surfaces.

### Required Modifications

The core modification is teaching RESIDUALS that a bimodal local distribution is signal, not noise. Key areas:

- **Local neighborhood definition**: Standard radius or k-nearest-neighbor queries will mix points from both sheets. The neighborhood model needs to accommodate the possibility of two surfaces in any local region.
- **Surface fitting**: Instead of fitting one surface and treating the rest as residual error, the fitter needs to detect when residuals have coherent spatial structure indicating a second surface rather than scatter.
- **Threshold/parameter tuning**: Noise characteristics of CT data differ from LiDAR — different density ranges, different point spacing, different error profiles.

The residual extraction itself may already work: if RESIDUALS fits the dominant sheet and subtracts it, the second sheet should appear as a spatially coherent residual signal. The question is whether the initial surface fitting contaminates itself by trying to fit both sheets as one.

## The 2D Manifold Prior

### Topological Constraint

The papyrus was originally a flat (2D) sheet. No matter how it was rolled, crushed, or compressed over two millennia, it remains a 2D manifold embedded in 3D space. It can bend, curl, and fold, but it cannot self-intersect. It has roughly constant thickness. Adjacent points on the original sheet are adjacent on the surface.

This is a powerful and underutilized constraint. At a touching point, two surfaces are close in 3D Euclidean space, but they are *distant* in the original 2D parameterization of the flat sheet. They are separate regions of the original papyrus that were rolled into spatial proximity. The problem transforms from "separate two surfaces stacked in 3D" to "determine which points belong to which region of the original flat sheet."

### Existing Work: Interactive Unrolling Tools

Tools already exist that parameterize the sheet as a connected manifold and allow interactive editing — moving a local region of the surface up or down while propagating changes along the sheet connectivity graph. This demonstrates that the 2D prior can be operationalized: the manifold connectivity is known (at least approximately) and perturbations propagate coherently along the sheet.

### Perturbation-Based Sheet Discrimination

The key insight: the same manifold connectivity that enables interactive editing can be used as an **active probe** to discriminate between sheets at touching points.

**Method:**

1. At a touching sheet region, take the fitted surface (or candidate surface from RESIDUALS) and apply a small perturbation — a local displacement of a point or patch.
2. Propagate that perturbation along the manifold connectivity graph, using the 2D sheet prior (adjacent points on the sheet should respond coherently to the perturbation).
3. Observe which surrounding points move together with the perturbed point and which do not.
4. Points that respond coherently (smooth, continuous displacement consistent with a connected surface) belong to the **same sheet**.
5. Points that show no response, discontinuous response, or incoherent response belong to a **different sheet**.

This is essentially a connectivity test implemented as a physical simulation. Instead of trying to separate sheets by static geometric analysis (which fails when they're in contact), you probe the dynamic response of the manifold. Two sheets that are touching in 3D are topologically disconnected — a perturbation on one does not propagate to the other.

**Integration with RESIDUALS:**

- Run RESIDUALS to fit the dominant surface and extract residuals
- Use perturbation propagation to verify that the fitted surface is a single coherent sheet (not accidentally straddling two sheets)
- Apply perturbation testing to the residual point set to determine if it forms a second coherent sheet
- Where the residual points respond coherently to perturbation, classify them as belonging to the secondary sheet
- Where they do not, classify as noise or reconstruction artifact

**Integration with sinogram-domain processing:**

- The perturbation test identifies *which* points belong to which sheet
- The sinogram-domain continuous reconstruction refines *where* those sheets actually are
- The 2D manifold prior constrains *how* the surfaces can be shaped (smooth, constant thickness, non-self-intersecting)
- These three sources of information (connectivity, raw projection data, physical priors) are complementary and can be combined in the iterative refinement step

**Implementation: Algorithmic, Learned, or Hybrid**

The perturbation-based discrimination can be implemented at three levels:

*Algorithmic approach:* Define a perturbation magnitude, propagate along the connectivity graph using a stiffness/elasticity model, threshold the response, classify. This is deterministic, interpretable, and requires no training data. Good for establishing baselines and understanding the data.

*ML approach:* Treat the perturbation response pattern as a feature vector. Instead of hand-coding what "coherent response" means, feed the response signatures into a model and let it learn the decision boundary between same-sheet and different-sheet. The model may capture subtleties that are difficult to encode by hand — response differences driven by papyrus fiber orientation, local damage, ink presence, compression history, or other factors not obvious from geometry alone.

*Hybrid approach:* Start with the algorithmic method to generate initial sheet assignments on clear-cut cases. Use those assignments as training labels for an ML model that then handles the ambiguous cases the algorithm cannot cleanly resolve. The algorithm bootstraps the training data for the learned model. As the ML model improves, its confident predictions can be fed back to retrain on harder examples — a self-improving loop.

The hybrid approach is likely the most practical path: the algorithmic version gets you working results quickly and generates the labeled data you need, while the ML version captures the edge cases and scales to the full scroll.

**Advantages:**

- Does not require the sheets to be geometrically separable in any single view or cross-section
- Works even when sheets are in full contact (zero separation) because the test is topological, not geometric
- Leverages existing manifold parameterization infrastructure (the interactive unrolling tools)
- Provides a binary classification (same sheet / different sheet) that is robust to noise in the density values
- Can be applied iteratively: once initial sheet assignments are made, refine the surfaces and re-test

## Computational Notes

- This is NOT full re-reconstruction of the entire volume. It targets only identified problem regions.
- Forward projection for the synthetic testbed can use standard ray-tracing libraries (ASTRA Toolbox, TomoPy, or custom CUDA kernels).
- The continuous point optimization is a local problem — small number of points, small sinogram patches — and can run on a single GPU per region.
- Parallelism is trivial: each region of interest is independent.

## Relationship to Existing Competition Findings

Top-scoring teams in the Vesuvius Challenge reported:
- Touching sheets remained unsolved; nnU-Net was relied upon to minimize their occurrence
- Post-processing (hole filling, threshold tuning) contributed disproportionately to final scores
- Logits fusion outperformed probability fusion on private LB, suggesting information was being lost in the sigmoid squash
- Higher detection thresholds (0.35-0.4) generalized better than lower ones tuned on training data

All of these are symptoms of working downstream of information loss. The logits/probability fusion finding is directly analogous: just as fusing before the sigmoid preserves more information than fusing after, reconstructing in continuous space before gridding preserves more information than processing after.

## Ink-Aware Hole Reconstruction

### Problem

Current hole filling techniques use geometric methods — detect a hole in the prediction mask, fill it. This creates a discontinuity at the fill boundary that manifests as a ring artifact. In an ink detection task, ring artifacts can mimic or obscure actual letter forms, injecting false signal or destroying real detections. Naive smoothing to blend the fill boundary risks smearing legitimate ink edges nearby.

### Approach: Learned Hole Filling

Replace geometric hole filling with a small ML model trained specifically to reconstruct missing regions in an ink-aware manner. The model sees the local context around a hole — surrounding ink predictions, confidence values, density characteristics — and learns what should be there based on the structure of actual ink.

Unlike geometric filling, a learned model understands that:
- Ink forms thin curved lines, not filled blobs
- Letters have characteristic structure and stroke patterns
- Ink has specific density signatures in the underlying CT data
- Confident predictions nearby constrain what the missing region can contain

### Self-Supervised Training with Synthetic Holes

Training data is essentially unlimited. The process:

1. Take any region of the scroll where ink predictions are high-confidence
2. Punch a synthetic hole — mask out a patch
3. Train the model to reconstruct the masked region from surrounding context
4. Ground truth is known because you created the hole

The synthetic holes can be engineered to match real hole characteristics:
- Sample from the actual distribution of hole sizes and shapes in the prediction masks
- Place holes in realistic locations (near sheet boundaries, low-confidence zones, areas of surface damage)
- Vary hole density and clustering to match observed patterns

This gives the model realistic training examples without requiring any new manual labels.

### Architecture

This is a small, targeted model:
- Operates only on identified hole regions — small patches, limited context window
- Input: local prediction map, confidence values, and optionally the underlying CT density
- Output: reconstructed ink prediction for the masked region, blended seamlessly with surrounding context
- Lightweight enough to run as a post-processing step without significant computational overhead

### Advantages Over Geometric Hole Filling

- No ring artifacts — the model learns to produce predictions consistent with surrounding context
- Ink-aware — fills are constrained by what ink actually looks like, not just spatial interpolation
- Confidence-calibrated — the model can output low confidence for ambiguous fills rather than committing to a hard binary
- Self-supervised — no additional labeling effort, unlimited training data
- Domain-specific — trained on the actual ink characteristics of carbonized papyrus, not generic inpainting

## Next Steps

1. Acquire raw sinogram data (from scanning facility or challenge organizers)
2. Build synthetic testbed with known touching sheet geometry
3. Benchmark basis functions on synthetic data
4. Convert touching sheet regions to point cloud and run through RESIDUALS — assess what modifications are needed for stacked point distributions
5. Implement perturbation-based sheet discrimination on known touching regions using existing manifold connectivity
6. Implement local sinogram patch extraction for real data
7. Prototype continuous point reconstruction on a single known-difficult region
8. Combine all three information sources: sinogram-domain separation, RESIDUALS surface detection, and perturbation-based manifold connectivity testing
9. Build synthetic hole training set from confident prediction regions; train ink-aware hole reconstruction model
10. Evaluate: does the refined surface produce better ink detection in previously ambiguous zones?
