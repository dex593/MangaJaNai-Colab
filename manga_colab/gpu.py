from __future__ import annotations

from typing import Any


def use_reduced_precision(precision: str, model: str, info: dict[str, Any]) -> bool:
    # DAT rejects FP16; only use its BF16 path when the GPU supports it natively.
    return precision == "auto" and info["fp16"] and (model != "manga-ora" or info["bf16"])


def initial_tile_size(model: str, requested: int, free_bytes: int) -> int:
    if requested or model not in ("manga-ora", "mangascale-gfisrv2"):
        return requested
    if model == "mangascale-gfisrv2":
        return 1024 if free_bytes >= 6 * 1024**3 else (512 if free_bytes >= 2 * 1024**3 else 256)
    # Ora attention uses more working memory than the upstream parameter-count estimate.
    if free_bytes >= 6 * 1024**3:
        return 512
    return 320 if free_bytes >= 2 * 1024**3 else 128


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
