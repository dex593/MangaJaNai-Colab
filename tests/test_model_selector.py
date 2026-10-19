import copy

import pytest

from manga_colab.model_selector import load_workflow, select_chain


@pytest.mark.parametrize(
    "height,family",
    [
        (1250, 1200),
        (1251, 1300),
        (1350, 1300),
        (1351, 1400),
        (1450, 1400),
        (1451, 1500),
        (1550, 1500),
        (1551, 1600),
        (1760, 1600),
        (1761, 1920),
        (1984, 1920),
        (1985, 2048),
    ],
)
@pytest.mark.parametrize("scale,multiplier", [(1, 2), (1.5, 2), (2, 2), (2.01, 4), (4, 4)])
def test_boundaries(height, family, scale, multiplier):
    chain = select_chain(900, height, True, scale, load_workflow())
    assert chain["ModelFilePath"].startswith(f"{multiplier}x_MangaJaNai_{family}p_V1_")


@pytest.mark.parametrize("scale,multiplier", [(2, 2), (2.1, 4)])
def test_color_uses_current_v3_fdat(scale, multiplier):
    name = select_chain(900, 1600, False, scale, load_workflow())["ModelFilePath"]
    assert name.startswith(f"{multiplier}x_IllustrationJaNai_V3denoise_FDAT_")
    assert name.endswith(".safetensors")


def test_first_matching_chain_wins_at_scale_two():
    workflow = load_workflow()
    chains = workflow["Chains"]["$values"]
    chains[2:4] = list(reversed(chains[2:4]))
    assert select_chain(800, 1200, True, 2, workflow)["ModelFilePath"].startswith("4x_")


def test_min_width_and_no_match():
    workflow = copy.deepcopy(load_workflow())
    workflow["Chains"]["$values"] = [workflow["Chains"]["$values"][2]]
    workflow["Chains"]["$values"][0]["MinResolution"] = "1000x0"
    with pytest.raises(ValueError, match="chain"):
        select_chain(900, 1200, True, 2, workflow)
