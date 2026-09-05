"""Decode an uploaded image into the tensor the exported ONNX expects.

Anomalib bakes resize + normalise into the ONNX graph, so at runtime we only need
to produce a fixed-size ``[1, 3, H, W]`` float32 array in ``[0, 1]`` (channels-first,
RGB). If a spec ever says preprocessing is *not* baked in, we also apply the
ImageNet normalisation here.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

from mddf.training.preprocess_spec import PreprocessSpec


class DecodedImage:
    """The original RGB image plus the network-ready tensor."""

    __slots__ = ("array", "original_size", "tensor")

    def __init__(
        self, tensor: np.ndarray, array: np.ndarray, original_size: tuple[int, int]
    ) -> None:
        self.tensor = tensor  # (1, 3, H, W) float32
        self.array = array  # (H0, W0, 3) uint8, original resolution, RGB
        self.original_size = original_size  # (width, height)


def load_rgb(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    with Image.fromarray(arr) as im:
        return np.asarray(im.resize((w, h), Image.BILINEAR), dtype=np.uint8)


def _center_crop(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    H, W = arr.shape[:2]
    top = max((H - h) // 2, 0)
    left = max((W - w) // 2, 0)
    return arr[top : top + h, left : left + w]


def preprocess(data: bytes, spec: PreprocessSpec) -> DecodedImage:
    rgb = load_rgb(data)
    original_size = (rgb.shape[1], rgb.shape[0])

    work = _resize(rgb, spec.image_size)
    if spec.center_crop is not None:
        work = _center_crop(work, spec.center_crop)

    x = work.astype(np.float32) / 255.0  # HWC [0,1]
    if spec.normalize and not spec.baked_into_onnx:
        mean = np.asarray(spec.mean, dtype=np.float32)
        std = np.asarray(spec.std, dtype=np.float32)
        x = (x - mean) / std

    tensor = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)
    return DecodedImage(tensor=tensor, array=rgb, original_size=original_size)


__all__ = ["DecodedImage", "load_rgb", "preprocess"]
