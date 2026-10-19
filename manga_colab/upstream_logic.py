# SPDX-License-Identifier: GPL-3.0-only
# Functions copied verbatim from the-database/MangaJaNaiConverterGui
# e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222, run_upscale.py.
# Inference is NOT vendored; these pure functions allow lightweight planning/tests.
from typing import Any
import cv2
import numpy as np


def get_h_w_c(image: np.ndarray) -> tuple[int, int, int]:
    return image.shape[0], image.shape[1], 1 if image.ndim == 2 else image.shape[2]


def cv_image_is_grayscale(image: np.ndarray, user_threshold: float) -> bool:
    _, _, c = get_h_w_c(image)

    if c == 1:
        return True

    b, g, r = cv2.split(image[:, :, :3])

    ignore_threshold = user_threshold

    # getting differences between (b,g), (r,g), (b,r) channel pixels
    r_g = cv2.subtract(cv2.absdiff(r, g), ignore_threshold)  # type: ignore
    r_b = cv2.subtract(cv2.absdiff(r, b), ignore_threshold)  # type: ignore
    g_b = cv2.subtract(cv2.absdiff(g, b), ignore_threshold)  # type: ignore

    # create masks to identify pure black and pure white pixels
    pure_black_mask = np.logical_and.reduce((r == 0, g == 0, b == 0))
    pure_white_mask = np.logical_and.reduce((r == 255, g == 255, b == 255))

    # combine masks to exclude both pure black and pure white pixels
    exclude_mask = np.logical_or(pure_black_mask, pure_white_mask)

    # exclude pure black and pure white pixels from diff_sum and image size calculation
    diff_sum = np.sum(np.where(exclude_mask, 0, r_g + r_b + g_b))
    size_without_black_and_white = np.sum(~exclude_mask) * 3

    # if the entire image is pure black or pure white, return False
    if size_without_black_and_white == 0:
        return False

    # finding ratio of diff_sum with respect to size of image without pure black and pure white pixels
    ratio = diff_sum / size_without_black_and_white

    return ratio <= user_threshold / 12

def should_chain_activate_for_image(
    original_width: int,
    original_height: int,
    is_grayscale: bool,
    target_scale: float,
    chain: dict[str, Any],
) -> bool:
    min_width, min_height = (int(x) for x in chain["MinResolution"].split("x"))
    max_width, max_height = (int(x) for x in chain["MaxResolution"].split("x"))

    # resolution tests
    if min_width != 0 and min_width > original_width:
        return False
    if min_height != 0 and min_height > original_height:
        return False
    if max_width != 0 and max_width < original_width:
        return False
    if max_height != 0 and max_height < original_height:
        return False

    # color / grayscale tests
    if is_grayscale and not chain["IsGrayscale"]:
        return False
    if not is_grayscale and not chain["IsColor"]:
        return False

    # scale tests
    if chain["MaxScaleFactor"] != 0 and target_scale > chain["MaxScaleFactor"]:
        return False
    if chain["MinScaleFactor"] != 0 and target_scale < chain["MinScaleFactor"]:
        return False

    return True
