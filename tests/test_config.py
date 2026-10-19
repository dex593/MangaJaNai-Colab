from dataclasses import replace

import pytest


@pytest.mark.parametrize(
    "change",
    [
        {"scale": "2"},
        {"scale": float("nan")},
        {"scale": 0},
        {"overwrite": "false"},
        {"tile": -1},
        {"cuda_device": -1},
        {"quality": 101},
        {"threshold": True},
        {"detector": "fake"},
        {"format": "jpg"},
    ],
)
def test_reject_invalid_types(config, change):
    with pytest.raises(ValueError):
        replace(config, **change)


def test_reject_output_equal_input(config):
    with pytest.raises(ValueError):
        replace(config, output=config.input)


def test_reject_null_input(config):
    with pytest.raises(ValueError, match="input"):
        replace(config, input=None)
