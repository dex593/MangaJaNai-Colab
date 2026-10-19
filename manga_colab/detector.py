from __future__ import annotations

import numpy as np

from .upstream_logic import cv_image_is_grayscale


def detect(image: np.ndarray, threshold: float = 12, mode: str = "upstream") -> bool:
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
        raise ValueError(f"Ảnh có số kênh không hỗ trợ: {image.shape}")
    if mode == "upstream":
        return bool(cv_image_is_grayscale(image, threshold))
    if mode != "corrected":
        raise ValueError(f"Detector không hợp lệ: {mode}")
    if image.ndim == 2 or image.shape[2] == 1:
        return True
    rgb = image[:, :, :3].astype(np.float64)
    keep = ~((rgb == 0).all(axis=2) | (rgb == 255).all(axis=2))
    if not keep.any():
        return True
    # Explicit opt-in: avoid upstream uint8 sum overflow and classify pure B/W as gray.
    difference = sum(
        np.maximum(np.abs(rgb[:, :, a] - rgb[:, :, b]) - threshold, 0)
        for a, b in ((0, 1), (0, 2), (1, 2))
    )
    return bool(difference[keep].sum() / (keep.sum() * 3) <= threshold / 12)


def read_image(path: str) -> np.ndarray:
    import pyvips

    # Exactly the upstream reader, including ICC conversion and strict decode.
    return (
        pyvips.Image.new_from_file(path, access="sequential", fail=True)
        .icc_transform("srgb")
        .numpy()
    )
