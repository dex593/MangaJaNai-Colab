import sys
from collections import OrderedDict
from io import StringIO
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


@pytest.mark.parametrize("requested", [0, 512])
@pytest.mark.parametrize("free_gib,cached_gib", [(15, 0), (0.5, 8)])
def test_reduced_tile_is_reused_only_for_same_model(
    tmp_path, monkeypatch, requested, free_gib, cached_gib
):
    """Fake inference isolates OOM scheduling; this is not a GPU quality test."""
    backend = object.__new__(Backend)
    backend.config = SimpleNamespace(
        models=tmp_path, model="manga-ora", tile=requested, cuda_device=0,
        scale=2, threshold=12, format="png", quality=95, lossless=True,
    )
    backend.log = StringIO()
    backend._context = lambda: None
    backend.cache = OrderedDict()
    backend.tile_limits = {}
    backend.oom_retries = 0
    image = np.zeros((64, 64, 3), dtype=np.float32)
    model = SimpleNamespace(dtype="torch.float32", to=lambda *args: None)
    tiles = []

    def preprocess(queue, source, destination, enabled, scale, width, height, chains, cache, threshold):
        chain = chains[0]
        cache[str(tmp_path / chain["ModelFilePath"])] = model
        queue.put((image, None, None, False, 64, 64, int(chain["ModelTileSize"]), model))

    def infer(image, tile, model):
        tiles.append(tile)
        if len(tiles) == 1:
            raise TileOOM(256)
        return image

    backend.module = SimpleNamespace(
        preprocess_worker_image=preprocess, ai_upscale_image=infer,
        get_tile_size=int, save_image=lambda *args: None,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        mem_get_info=lambda index: (int(free_gib * 1024**3), 16 * 1024**3),
        memory_reserved=lambda index: int(cached_gib * 1024**3),
        memory_allocated=lambda index: 0,
        empty_cache=lambda: None,
    )))
    rows = []
    for name in ("ora.pth", "ora.pth", "other.pth"):
        (tmp_path / name).touch()
        rows.append(backend.process(tmp_path / "in.png", tmp_path / "out.png",
                                    {"ModelFilePath": name}, lambda *args: None))
    assert tiles == [512, 256, 256, 512]
    assert [row["oom_retries"] for row in rows] == [1, 0, 0]
    assert [row["model_cache_hit"] for row in rows] == [False, True, False]
    assert rows[0]["precision"] == "fp32"
