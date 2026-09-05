# Model card — MDDF anomaly detectors

## Overview

Per-category unsupervised anomaly detectors for the 15 [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
object/texture categories. One model is trained per `(method, category)` pair for
three methods:

| Method | Paradigm | Backbone | Train time (RTX 4060) |
|---|---|---|---|
| PaDiM | per-patch multivariate Gaussian | ResNet-18 | ~30 s / category |
| PatchCore | coreset memory bank (10 %) | WideResNet-50 | ~1.5 min / category |
| EfficientAD-S | student–teacher + autoencoder | custom PDN | step-budgeted (~24k steps) |

Implementation: [Anomalib](https://github.com/open-edge-platform/anomalib) 2.6.

## Intended use

Visual quality inspection where defect-free parts are plentiful and defects are
rare / open-ended. Returns an image-level anomaly score, a pixel-level anomaly
map, a binary mask and defect bounding boxes. **Not** a safety-certified system;
a human reviews flagged parts.

## Training data

MVTec AD `train/good` only (defect-free). No defect images or masks are used for
fitting — thresholds are chosen post-hoc on the test split (see limitations).
Input: RGB resized to 256 (PatchCore additionally centre-crops to 224), ImageNet
normalisation (PaDiM/PatchCore) baked into the exported ONNX.

## Evaluation

MVTec AD `test/` split. Metrics: image AUROC, pixel AUROC, AUPRO, image F1,
and an operating-point analysis (threshold / precision / false-alarm rate /
miss rate at target recalls). Baseline for comparison: the image AUROC reported
in the PatchCore paper (Roth et al., 2022). See `docs/RESULTS.md`.

## Serving

Exported to ONNX (opset 17), optionally dynamic-INT8 quantized. Inference is
ONNXRuntime CPU, single-thread, no PyTorch. Artifacts (~per-category `model.onnx`
+ JSON) are hosted in a Hugging Face model repo and pulled lazily.

## Limitations & risks

- **Threshold selection uses the test split.** The adaptive F1-optimal threshold
  and the operating-point curves are computed on MVTec AD `test/`; in a real
  deployment these must be set on a held-out validation set of the target line.
- **Per-category models.** A model only makes sense for images of its category;
  out-of-distribution input yields undefined scores. The service requires the
  caller to name the category.
- **PaDiM with ResNet-18** trades accuracy for memory/speed and is expected to
  trail PatchCore on the harder categories — it is included as the
  distribution-based reference point, not the recommended production model.
- MVTec AD is a controlled benchmark (fixed pose, lighting). Real lines have more
  nuisance variation; expect a domain gap.
- Licence: MVTec AD is CC BY-NC-SA 4.0 (non-commercial).
