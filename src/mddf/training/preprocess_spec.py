"""Serialisable description of a model's input preprocessing.

Anomalib bakes the ``PreProcessor`` transform into the exported ONNX graph, but the
inference service still needs to know the target size (to letterbox / report
coordinates) and whether it must normalise itself. This spec is written next to
every exported artifact and re-read by :mod:`mddf.inference.preprocess`.
"""

from __future__ import annotations

from pydantic import BaseModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PreprocessSpec(BaseModel):
    image_size: tuple[int, int]
    center_crop: tuple[int, int] | None = None
    normalize: bool = True
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    # True when the ONNX graph already contains the resize+normalize (Anomalib 2.x).
    baked_into_onnx: bool = True

    @property
    def network_input(self) -> tuple[int, int]:
        return self.center_crop or self.image_size


def padim_spec(image_size: int = 256) -> PreprocessSpec:
    return PreprocessSpec(
        image_size=(image_size, image_size),
        center_crop=None,
        normalize=True,
    )


def patchcore_spec(image_size: int = 256, center_crop: int = 224) -> PreprocessSpec:
    return PreprocessSpec(
        image_size=(image_size, image_size),
        center_crop=(center_crop, center_crop),
        normalize=True,
    )


def efficient_ad_spec(image_size: int = 256) -> PreprocessSpec:
    # EfficientAD normalises internally; its PreProcessor is resize-only.
    return PreprocessSpec(
        image_size=(image_size, image_size),
        center_crop=None,
        normalize=False,
    )


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PreprocessSpec",
    "efficient_ad_spec",
    "padim_spec",
    "patchcore_spec",
]
