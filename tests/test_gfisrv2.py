"""Architecture checks need backend dependencies; weights are not downloaded by pytest."""

import hashlib
from pathlib import Path

import pytest


def test_vendored_gfisrv2_matches_pinned_source():
    source = Path(__file__).parents[1] / "manga_colab/_vendor/gfisrv2.py"
    content = source.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(content).hexdigest() == (
        "441368e7ebab3b40ca31ac66200c2deb7aad7ae366b07408853b2f27fe78f8a6"
    )


def test_descriptor_matches_original_forward_and_rejects_other_variant():
    torch = pytest.importorskip("torch")
    spandrel = pytest.importorskip("spandrel")
    from manga_colab._vendor.gfisrv2 import GFISRV2
    from manga_colab.extra_arches import GFISRV2Arch, install

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        original = GFISRV2().eval()
        state = original.state_dict()
        install()
        install()
        descriptor = spandrel.ModelLoader().load_from_state_dict(state)
        assert descriptor.architecture.id == "GFISRV2"
        assert (descriptor.scale, descriptor.input_channels, descriptor.output_channels) == (4, 3, 3)
        image = torch.rand(1, 3, 17, 23)
        with torch.inference_mode():
            expected = original(image).clamp(0, 1)
            result = descriptor(image)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert result.shape == (1, 3, 68, 92)
        assert torch.isfinite(result).all()
        bad = dict(state)
        bad["upscale.MetaUpsample"] = torch.tensor([3, 1, 2, 48, 3, 32, 4])
        with pytest.raises(ValueError, match="4x_wtp_MangaScale"):
            GFISRV2Arch().load(bad)
        missing = dict(state)
        del missing["in_to_dim.weight"]
        with pytest.raises(RuntimeError, match="Missing key"):
            GFISRV2Arch().load(missing)
    finally:
        torch.set_num_threads(previous_threads)
