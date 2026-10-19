from types import SimpleNamespace

import numpy as np
import pytest

from manga_colab.backend import Backend, TileOOM


def test_oom_exits_original_split_before_it_can_reuse_partial_rows():
    class Split:
        pass

    backend = object.__new__(Backend)
    backend.callback = lambda *args: None
    backend.split_module = SimpleNamespace(Split=Split)
    tiler = SimpleNamespace(starting_tile_size=lambda *args: (256, 256))
    invoked = []

    def original(image, callback, tiler, overlap):
        invoked.append(True)
        callback(image, None)
        raise AssertionError("Must not continue with partial output after OOM")

    backend.original_split = original
    with pytest.raises(TileOOM) as caught:
        backend._tracked_split(np.zeros((400, 400, 3)), lambda *args: Split(), tiler)
    assert caught.value.tile == 128
    assert len(invoked) == 1


def test_non_oom_cuda_error_is_not_masked():
    backend = object.__new__(Backend)
    backend.callback = lambda *args: None
    backend.split_module = SimpleNamespace(Split=type("Split", (), {}))
    backend.original_split = lambda image, callback, *args: callback(image, None)
    tiler = SimpleNamespace(starting_tile_size=lambda *args: (64, 64))

    def infer(*args):
        raise RuntimeError("CUDA illegal memory access")

    with pytest.raises(RuntimeError, match="illegal memory access"):
        backend._tracked_split(np.zeros((64, 64, 3)), infer, tiler)
