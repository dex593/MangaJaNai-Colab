import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from manga_colab.backend import make_settings
from manga_colab.cli import from_args, parser
from manga_colab.model_manager import ModelManager
from manga_colab.model_selector import DATA, EXTRA_MODELS, load_workflow, select_chain
from manga_colab.scanner import build_plan, scan_files


def test_original_workflow_remains_unchanged():
    assert load_workflow() == json.loads((DATA / "workflow.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("model", list(EXTRA_MODELS))
@pytest.mark.parametrize("height", [500, 1251, 1985, 4000])
@pytest.mark.parametrize("scale", [1, 1.5, 2])
def test_extra_models_do_not_use_mangajanai_height_bands(model, height, scale):
    workflow = load_workflow(model=model)
    chain = select_chain(800, height, True, scale, workflow)
    assert chain["ModelFilePath"] == EXTRA_MODELS[model]["filename"]
    assert chain["AutoAdjustLevels"] is False
    assert chain["ResizeFactorBeforeUpscale"] == 100
    assert len(workflow["Chains"]["$values"]) == 1


def test_native_scales_and_color_policy(config):
    eula = load_workflow(model="eula-digimanga")
    ora = load_workflow(model="manga-ora")
    assert select_chain(800, 1200, True, 4, eula)["ModelFilePath"].startswith("4x-")
    with pytest.raises(ValueError, match="chỉ hỗ trợ manga đen trắng"):
        select_chain(800, 1200, False, 2, eula)
    assert select_chain(800, 1200, False, 2, ora)["ModelFilePath"] == "2x_Manga_Ora.pth"
    with pytest.raises(ValueError, match="chain"):
        select_chain(800, 1200, True, 4, ora)
    with pytest.raises(ValueError, match="giảm số lần"):
        replace(config, model="manga-ora", scale=4)
    with pytest.raises(ValueError, match="model"):
        replace(config, model="unverified")
    with pytest.raises(ValueError, match="đồng thời"):
        replace(config, model="manga-ora", workflow=DATA / "workflow.json")


def test_eula_detects_pure_black_white_and_rejects_color(config, write_image):
    config = replace(config, model="eula-digimanga")
    array = np.zeros((12, 8, 3), dtype=np.uint8)
    array[::2] = 255
    Image.fromarray(array).save(config.input / "Đen trắng.png")
    write_image(config.input / "Màu.png", color=True)
    plan = {item.source: item for item in build_plan(config, load_workflow(model=config.model))}
    assert plan["Đen trắng.png"].grayscale is True
    assert not plan["Đen trắng.png"].error
    assert "có màu" in plan["Màu.png"].error
    assert replace(config, model="auto").effective_detector == "upstream"


@pytest.mark.parametrize("model", list(EXTRA_MODELS))
def test_settings_and_cli_for_each_extra_model(config, model):
    parsed = parser().parse_args(["run", "--input", str(config.input), "--model", model])
    config = from_args(parsed)
    workflow = load_workflow(model=model)
    settings = json.loads(json.dumps(make_settings(config, workflow, 1, False)))
    actual = settings["Workflows"]["$values"][0]
    assert actual["Chains"]["$values"][0]["ModelFilePath"] == EXTRA_MODELS[model]["filename"]
    assert config.model == model
    parsed = parser().parse_args(
        ["run", "--input", str(config.input), "--model", model, "--auto-model"]
    )
    with pytest.raises(ValueError, match="--auto-model"):
        from_args(parsed)


def test_model_change_changes_resume_fingerprint(config):
    manager = ModelManager(config.models)
    fingerprints = []
    for model in ("auto", *EXTRA_MODELS):
        workflow = load_workflow(model=model)
        chain = select_chain(800, 1200, True, 2, workflow)
        fingerprints.append(
            config.fingerprint(workflow, manager.entry(chain["ModelFilePath"])["sha256"])
        )
    assert len(set(fingerprints)) == 3


def test_extra_catalog_has_published_checksums_and_sources():
    assert EXTRA_MODELS["eula-digimanga"]["size"] == 67010885
    assert EXTRA_MODELS["manga-ora"]["size"] == 45723706
    for entry in EXTRA_MODELS.values():
        assert entry["license"] == "CC-BY-NC-SA-4.0"
        assert entry["source"].startswith("https://openmodeldb.info/models/")
        assert len(bytes.fromhex(entry["sha256"])) == 32


def test_scanning_excludes_other_models_previous_outputs(config, write_image):
    write_image(config.input / "001.jpg")
    previous = config.input / "Output/Eula Digimanga"
    write_image(previous / "001.png")
    (previous / ".manga-colab").mkdir()
    current = replace(config, model="manga-ora", output=config.input / "Output/Manga Ora")
    assert [str(path) for path in scan_files(current)] == ["001.jpg"]
