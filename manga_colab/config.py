from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

UPSTREAM_COMMIT = "e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222"
UPSTREAM_REPO = "https://github.com/the-database/MangaJaNaiConverterGui.git"
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    input: Path
    output: Path
    models: Path
    scale: float = 2.0
    format: str = "png"
    threshold: float = 12.0
    detector: str = "upstream"
    resume: bool = True
    overwrite: bool = False
    retry_failed: bool = False
    tile: int = 0
    cuda_device: int = 0
    precision: str = "auto"
    quality: int = 95
    lossless: bool = True
    stage: Path | None = None
    log_level: str = "normal"
    backend: Path = ROOT / ".upstream" / "MangaJaNaiConverterGui"
    workflow: Path | None = None

    def __post_init__(self) -> None:
        for key in ("input", "output", "models", "backend", "stage", "workflow"):
            value = getattr(self, key)
            if value is None and key in ("input", "output", "models", "backend"):
                raise ValueError(f"{key}: cần đường dẫn không rỗng")
            if value is not None:
                if not isinstance(value, (str, Path)) or not str(value).strip():
                    raise ValueError(f"{key}: cần đường dẫn không rỗng")
                object.__setattr__(self, key, Path(value).expanduser().resolve())
        for key in ("resume", "overwrite", "retry_failed", "lossless"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key}: cần true/false")
        for key in ("tile", "cuda_device", "quality"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key}: cần số nguyên")
        for key in ("scale", "threshold"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key}: cần số hữu hạn")
        if not 1 <= self.scale <= 4:
            raise ValueError("scale: hỗ trợ từ 1 đến 4 (kể cả 1.5, 2.5)")
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold: phải từ 0 đến 255")
        if self.tile != 0 and self.tile < 32:
            raise ValueError("tile: dùng 0 (auto) hoặc ít nhất 32 px")
        if self.cuda_device < 0 or not 1 <= self.quality <= 100:
            raise ValueError("cuda_device phải >= 0; quality phải từ 1 đến 100")
        for key, choices in {
            "format": ("png", "jpeg", "webp", "avif"),
            "detector": ("upstream", "corrected"),
            "precision": ("auto", "fp32"),
            "log_level": ("quiet", "normal", "verbose", "debug"),
        }.items():
            if getattr(self, key) not in choices:
                raise ValueError(f"{key}: chọn một trong {choices}")
        if self.output == self.input or self.input.is_relative_to(self.output):
            raise ValueError("Output phải khác input và không được là thư mục cha của input")
        if self.stage and (self.stage == self.input or self.input.is_relative_to(self.stage)):
            raise ValueError("Staging không được chứa input")
        if self.models == self.input or self.input.is_relative_to(self.models):
            raise ValueError("Model cache không được chứa input")

    def fingerprint(self, workflow: dict, model_digest: str) -> str:
        data = {
            k: getattr(self, k)
            for k in (
                "scale",
                "format",
                "threshold",
                "detector",
                "precision",
                "quality",
                "lossless",
                "tile",
            )
        }
        data.update(upstream=UPSTREAM_COMMIT, adapter=1, workflow=workflow, model=model_digest)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def as_json(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise ValueError("Cấu hình YAML phải là mapping")
    unknown = result.keys() - Config.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(f"Cấu hình không được hỗ trợ: {', '.join(sorted(unknown))}")
    return result
