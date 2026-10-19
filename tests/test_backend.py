import json

import pytest

from manga_colab.backend import make_settings
from manga_colab.gpu import map_device
from manga_colab.model_selector import load_workflow


def test_gpu_indices_not_interchangeable():
    devices = [
        {"type": "cpu", "index": 0},
        {"type": "cuda", "index": 0},
        {"type": "cuda", "index": 1},
    ]
    assert map_device(devices, 0) == (1, 0)
    assert map_device(devices, 1) == (2, 1)
    with pytest.raises(RuntimeError):
        map_device(devices, 2)


def test_settings_json(config):
    value = json.loads(json.dumps(make_settings(config, load_workflow(), 1, True)))
    assert value["SelectedDeviceIndex"] == 1
    assert value["ModelsDirectory"] == str(config.models)
    workflow = value["Workflows"]["$values"][value["SelectedWorkflowIndex"]]
    assert workflow["PngSelected"] and not workflow["WebpSelected"]
    assert workflow["UpscaleScaleFactor"] == 2
    assert workflow["GrayscaleDetectionThreshold"] == 12
    assert len(workflow["Chains"]["$values"]) == 16
