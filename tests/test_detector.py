import numpy as np
import pytest

from manga_colab.detector import detect
from manga_colab.upstream_logic import cv_image_is_grayscale


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_gray(channels):
    image = np.full((8, 9, channels), 128, dtype=np.uint8)
    assert detect(image)


def test_color_and_threshold():
    image = np.full((8, 9, 3), (50, 53, 50), dtype=np.uint8)
    assert detect(image, 12)
    assert not detect(image, 0)
    assert not detect(np.full((8, 9, 3), (255, 0, 0), dtype=np.uint8))


@pytest.mark.parametrize("value", [0, 255])
def test_upstream_pure_rgb_bw_quirk(value):
    image = np.full((8, 9, 3), value, dtype=np.uint8)
    assert not detect(image)
    assert detect(image, mode="corrected")
    assert detect(image[:, :, 0])


def test_overflow_opt_in_is_explicit():
    image = np.full((8, 9, 3), (0, 0, 140), dtype=np.uint8)
    assert detect(image)  # (128 + 128) wraps to zero in upstream uint8.
    assert not detect(image, mode="corrected")


def test_same_as_upstream_random_inputs():
    rng = np.random.default_rng(7)
    for threshold in (0, 6, 12, 24):
        for _ in range(8):
            image = rng.integers(0, 256, (15, 13, 3), dtype=np.uint8)
            assert detect(image, threshold) == bool(cv_image_is_grayscale(image, threshold))


def test_bad_channels():
    with pytest.raises(ValueError):
        detect(np.zeros((8, 8, 2), dtype=np.uint8))
