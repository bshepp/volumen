# Social Post Draft (HF Profile)

Scored 0.405/0.426 on the Vesuvius Challenge Surface Detection (up from 0.390/0.409 after integrating 1st-place post-processing) — the 1st place team hit 0.614 with nnU-Net. That gap sent us down a rabbit hole.

We built four independent 3D segmentation pipelines to understand what actually matters: standard U-Net, deep supervision + focal loss, multi-scale fusion, and finally nnU-Net itself. Each pipeline taught us something — V1 showed us our GPU was idle 90% of the time (CPU feature bottleneck), V2 proved skeleton recall beats clDice, V3 NaN'd at epoch 55 from FP16 overflow, and nnU-Net is training on HF Jobs right now.

The 1st place post-processing pipeline (binary closing, height-map patching, LUT hole plugging) turned out to be a bigger lever than we expected. We only had access after the competition closed and writeups were published.

Code: github.com/bshepp/volumen
