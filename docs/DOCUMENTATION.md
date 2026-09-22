# Project Documentation - Coronary Artery Enhancement

## 1. Problem
Improve visibility of contrast-filled coronary arteries in 16-bit cardiac X-ray/cine frames by suppressing
ribs, spine, lungs and noise, within **36 ms/frame**, and show Raw vs Processed vs Enhanced.

## 2. Approach and why
Deep-learning segmentation needs labelled clinical data (not available) and adds GPU/latency risk.
A physics-motivated classical pipeline needs no training and has a tight, predictable latency.

**Key observation.** X-ray intensity follows `I = I0 * exp(-sum(mu*t))`. In the log domain
`L = -log(I)` every structure adds its own thickness term. Anatomical clutter (ribs 15-25 mm, spine, lung
fields, heart silhouette) is *spatially wide / smooth*, whereas coronary arteries are *thin (2-5 mm) tubes*.
A scale separation therefore separates them:

1. **Background estimate** - grey opening with a structuring element wider than the widest vessel removes
   thin bright structures; what remains (`B`) is bones + lungs + soft tissue.
2. **Top-hat residual** `T = L - B` keeps thin structures only.
3. **Temporal filtering** (cine) - recursive average with a per-pixel motion gate, so static noise averages
   out but moving vessels are not ghosted.
4. **Ridge (vesselness) filter** - eigenvalues of the Hessian of `T` at several scales. Across a bright tube the
   smaller eigenvalue is strongly negative and the larger one ~0; blobs (both large) and noise (both small)
   are rejected. Scale-normalised, max over scales.
5. **Enhancement** - `enhanced = T * (floor + (1-floor)*V)`, adaptive local contrast + gamma, quantised to 16-bit.

## 3. 16-bit handling
Input `uint16` is converted to float32 only internally; all outputs are `uint16`
(`processed`, `enhanced`, `vesselness`). Percentile normalisation avoids assuming 12- vs 16-bit range.
`io.to_uint16` converts 8-bit / float / DICOM (MONOCHROME1 inverted) sources.

## 4. Latency engineering
* Background at half resolution; large ridge scales at half resolution; frames >512 px analysed at 512 px.
* Separable derivative-of-Gaussian kernels (`sepFilter2D`) instead of full 2-D convolutions.
* Percentiles from a 1/36 pixel sub-sample; in-place NumPy/OpenCV ops; kernels/SE created once in `__init__`.
* OpenCV 16-bit CLAHE (~14 ms per call, 65536-bin histograms) was replaced by a float local-gain map (~2 ms);
  it stays available via `use_clahe=True`.
* Benchmark methodology: warm-up 30 frames, then `perf_counter` around `process()` for N frames;
  reports avg/min/max/P95/P99/FPS and count of frames above 36 ms.

## 5. Parameters (`PipelineConfig`)
`vessel_max_width` (px of the widest coronary at analysis resolution - the most important knob),
`sigmas`, `temporal_alpha`, `motion_thr`, `vessel_floor`, `context`, `local_strength`, `vessels_dark`, `use_clahe`.

## 6. Validation
`scripts/evaluate.py` compares each output with the synthetic ground-truth mask (ROC-AUC, best Dice, CNR,
background clutter). See README section 7. `tests/` checks 16-bit I/O, vessel AUC, and the 36 ms budget.

## 7. Known limitations / future work
Synthetic-only quantitative validation; rib cortical edges and catheters can leak through on real data;
possible extensions: temporal static-structure subtraction, U-Net refinement, GPU (CUDA/OpenCL) port,
DICOM windowing metadata, motion compensation.
