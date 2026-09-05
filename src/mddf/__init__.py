"""MDDF — Manufacturing Defect Detection System.

Unsupervised visual anomaly detection on the MVTec AD dataset. Two models are
trained per category on defect-free images only (PatchCore for accuracy,
EfficientAD for latency) and served CPU-only through a FastAPI service.
"""

__version__ = "0.1.0"
