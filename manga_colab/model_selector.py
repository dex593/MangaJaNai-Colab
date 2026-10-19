from __future__ import annotations

import json
import math
from pathlib import Path

from .upstream_logic import should_chain_activate_for_image

DATA = Path(__file__).with_name("data")
EXTRA_MODELS = json.loads((DATA / "extra_models.json").read_text(encoding="utf-8"))


def load_workflow(path: Path | None = None, model: str = "auto") -> dict:
    if model != "auto" and model not in EXTRA_MODELS:
        raise ValueError(f"Model không được hỗ trợ: {model}")
    if path and model != "auto":
        raise ValueError("Chọn --workflow hoặc --model, không dùng đồng thời cả hai")
    with (path or DATA / "workflow.json").open(encoding="utf-8") as stream:
        workflow = json.load(stream)
    if not isinstance(workflow, dict):
        raise ValueError("Workflow JSON phải là object")
    chains = workflow.get("Chains", {}).get("$values", [])
    if not chains:
        raise ValueError("Workflow phải có Chains.$values")
    allowed = {
        c["ModelFilePath"]
        for c in json.loads((DATA / "workflow.json").read_text())["Chains"]["$values"]
    }
    if model != "auto":
        entry = EXTRA_MODELS[model]
        allowed.add(entry["filename"])
        chain = chains[0].copy()
        chain.update(
            MinResolution="0x0",
            MaxResolution="0x0",
            IsGrayscale=True,
            IsColor=entry["input_channels"] == 3,
            MinScaleFactor=0,
            MaxScaleFactor=entry["scale"],
            ModelFilePath=entry["filename"],
            AutoAdjustLevels=False,
        )
        workflow["WorkflowName"] = entry["label"]
        chains = workflow["Chains"]["$values"] = [chain]
    for chain in chains:
        if chain.get("ModelFilePath") not in allowed:
            raise ValueError(
                f"Chỉ hỗ trợ model chính thức đã kiểm tra: {chain.get('ModelFilePath')}"
            )
        for key in ("MinResolution", "MaxResolution"):
            value = chain.get(key, "").split("x")
            if len(value) != 2 or any(not v.isdecimal() for v in value):
                raise ValueError(f"Chain {key}: cần WIDTHxHEIGHT, dùng 0 để không giới hạn")
        for key in (
            "MinScaleFactor",
            "MaxScaleFactor",
            "ResizeHeightBeforeUpscale",
            "ResizeWidthBeforeUpscale",
            "ResizeFactorBeforeUpscale",
        ):
            if (
                type(chain.get(key)) not in (int, float)
                or not math.isfinite(chain[key])
                or chain[key] < 0
            ):
                raise ValueError(f"Chain {key}: cần số >= 0")
        for key in ("IsGrayscale", "IsColor", "AutoAdjustLevels"):
            if type(chain.get(key)) is not bool:
                raise ValueError(f"Chain {key}: cần boolean")
    return workflow


def select_chain(width: int, height: int, grayscale: bool, scale: float, workflow: dict) -> dict:
    for chain in workflow["Chains"]["$values"]:
        if should_chain_activate_for_image(width, height, grayscale, scale, chain):
            return chain.copy()
    if workflow.get("WorkflowName") == EXTRA_MODELS["eula-digimanga"]["label"] and not grayscale:
        raise ValueError(
            "Eula Digimanga BW chỉ hỗ trợ manga đen trắng. "
            "Ảnh này có màu; hãy dùng chế độ MangaJaNai cho ảnh màu."
        )
    raise ValueError(
        f"Không có chain phù hợp: {width}x{height}, grayscale={grayscale}, scale={scale}"
    )
