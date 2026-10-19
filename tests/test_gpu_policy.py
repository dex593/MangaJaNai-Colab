import pytest

from manga_colab.gpu import initial_tile_size, use_reduced_precision


@pytest.mark.parametrize(
    "model,precision,fp16,bf16,expected",
    [
        ("manga-ora", "auto", True, False, False),  # T4: avoid emulated BF16
        ("manga-ora", "auto", True, True, True),  # L4/A100: native BF16
        ("manga-ora", "auto", False, False, False),
        ("manga-ora", "fp32", True, True, False),
        ("auto", "auto", True, False, True),
        ("eula-digimanga", "auto", True, False, True),
        ("mangascale-gfisrv2", "auto", True, False, True),
        ("auto", "fp32", True, True, False),
    ],
)
def test_precision_matches_model_and_native_gpu_support(model, precision, fp16, bf16, expected):
    assert use_reduced_precision(precision, model, {"fp16": fp16, "bf16": bf16}) is expected


@pytest.mark.parametrize("free_gib,expected", [(1, 128), (2, 320), (4, 320), (6, 512), (15, 512)])
def test_ora_auto_tiles_use_free_vram(free_gib, expected):
    assert initial_tile_size("manga-ora", 0, free_gib * 1024**3) == expected


def test_manual_tiles_and_other_models_keep_existing_behavior():
    assert initial_tile_size("manga-ora", 320, 16 * 1024**3) == 320
    assert initial_tile_size("auto", 0, 16 * 1024**3) == 0
    assert initial_tile_size("eula-digimanga", 0, 16 * 1024**3) == 0


@pytest.mark.parametrize("free_gib,expected", [(1, 256), (2, 512), (4, 512), (6, 1024)])
def test_gfisrv2_auto_tiles_use_free_vram(free_gib, expected):
    assert initial_tile_size("mangascale-gfisrv2", 0, free_gib * 1024**3) == expected
