from __future__ import annotations

from typing import Any


def map_device(devices: list[dict], cuda_index: int) -> tuple[int, int]:
    """Return (GUI index including CPU, settings index excluding CPU)."""
    accelerated = [d for d in devices if d["type"] != "cpu"]
    selected = next((d for d in devices if d["type"] == "cuda" and d["index"] == cuda_index), None)
    if selected is None:
        raise RuntimeError(f"Không tìm thấy CUDA device {cuda_index}; bật GPU trong Colab")
    return devices.index(selected), accelerated.index(selected)


def gpu_info(index: int = 0, required: bool = False) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "available": torch.cuda.is_available(),
    }
    if not result["available"]:
        if required:
            raise RuntimeError(
                "Colab chưa có GPU CUDA. Chọn Runtime → Change runtime type → GPU rồi chạy lại."
            )
        return result
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA index {index} vượt số GPU {torch.cuda.device_count()}")
    torch.cuda.set_device(index)
    props = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    result.update(
        name=props.name,
        free=free,
        total=total,
        compute=f"{props.major}.{props.minor}",
        fp16=props.major >= 7,
        bf16=props.major >= 8 and torch.cuda.is_bf16_supported(),
        index=index,
    )
    return result
