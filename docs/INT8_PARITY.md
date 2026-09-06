# INT8 parity

Each fp32 / INT8 ONNX pair re-scored over a stratified 25 images/class/category sample.
Gate: |delta AUROC| <= 0.03. **Passed: True** (30/30).

| Model | Category | AUROC fp32 | AUROC int8 | delta |
|---|---|---:|---:|---:|
| padim | bottle | 0.9920 | 0.9920 | +0.0000 |
| patchcore | bottle | 1.0000 | 1.0000 | +0.0000 |
| padim | cable | 0.7472 | 0.7680 | +0.0208 |
| patchcore | cable | 0.9792 | 0.9792 | +0.0000 |
| padim | capsule | 0.8765 | 0.8783 | +0.0017 |
| patchcore | capsule | 0.8643 | 0.8626 | -0.0017 |
| padim | carpet | 0.9936 | 0.9936 | +0.0000 |
| patchcore | carpet | 0.9968 | 0.9968 | +0.0000 |
| padim | grid | 0.8305 | 0.8095 | -0.0210 |
| patchcore | grid | 0.9562 | 0.9562 | +0.0000 |
| padim | hazelnut | 0.8784 | 0.8784 | +0.0000 |
| patchcore | hazelnut | 1.0000 | 1.0000 | +0.0000 |
| padim | leather | 1.0000 | 1.0000 | +0.0000 |
| patchcore | leather | 1.0000 | 1.0000 | +0.0000 |
| padim | metal_nut | 0.9709 | 0.9709 | +0.0000 |
| patchcore | metal_nut | 0.9709 | 0.9709 | +0.0000 |
| padim | pill | 0.8336 | 0.8352 | +0.0016 |
| patchcore | pill | 0.8528 | 0.8416 | -0.0112 |
| padim | screw | 0.7488 | 0.7664 | +0.0176 |
| patchcore | screw | 0.6320 | 0.6336 | +0.0016 |
| padim | tile | 0.9696 | 0.9648 | -0.0048 |
| patchcore | tile | 1.0000 | 1.0000 | +0.0000 |
| padim | toothbrush | 0.8433 | 0.8433 | +0.0000 |
| patchcore | toothbrush | 0.8233 | 0.8267 | +0.0033 |
| padim | transistor | 0.9184 | 0.9232 | +0.0048 |
| patchcore | transistor | 0.9968 | 0.9968 | +0.0000 |
| padim | wood | 0.9768 | 0.9768 | +0.0000 |
| patchcore | wood | 0.9958 | 0.9958 | +0.0000 |
| padim | zipper | 0.8528 | 0.8544 | +0.0016 |
| patchcore | zipper | 0.9904 | 0.9904 | +0.0000 |
