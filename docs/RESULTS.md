# Results

_Populated by `mddf benchmark --latency` after training + export. This file is
regenerated, not hand-edited._

Until the full run lands, single-category smoke checks (RTX 4060, image size 256):

| Category | Model | Image AUROC | Published (PatchCore) | Pixel AUROC | AUPRO | Train s |
|---|---|---:|---:|---:|---:|---:|
| leather | padim | 1.000 | 1.000 | 0.990 | 0.974 | 33 |
| leather | patchcore | 1.000 | 1.000 | 0.990 | — | 91 |
| bottle | padim | 0.997 | 1.000 | 0.981 | 0.931 | 27 |
