from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config
from .detector import detect, read_image
from .model_selector import select_chain
from .storage import sha256

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}


@dataclass
class PlanItem:
    source: str
    output: str
    size: int = 0
    source_hash: str = ""
    width: int = 0
    height: int = 0
    grayscale: bool = False
    model: str = ""
    chain: dict | None = None
    error: str = ""

    def as_json(self) -> dict:
        return asdict(self)


def natural_key(path: Path) -> tuple:
    return tuple(
        (0, int(s)) if s.isdigit() else (1, s.casefold())
        for s in re.split(r"(\d+)", path.as_posix())
    ) + ((2, path.as_posix()),)


def scan_files(config: Config) -> list[Path]:
    excluded = [config.output, config.models, config.backend]
    if config.stage:
        excluded.append(config.stage)
    paths = []

    def failed(error: OSError) -> None:
        raise error

    for root, directories, files in os.walk(config.input, onerror=failed, followlinks=False):
        directories[:] = [
            name
            for name in directories
            if name != ".manga-colab"
            and not (Path(root) / name).is_symlink()
            and not (Path(root) / name / ".manga-colab").is_dir()
            and not any((Path(root) / name).resolve().is_relative_to(p) for p in excluded)
        ]
        for name in files:
            path = Path(root) / name
            if not path.is_symlink() and path.suffix.lower() in EXTENSIONS:
                paths.append(path.relative_to(config.input))
    return sorted(paths, key=natural_key)


def output_names(paths: list[Path], extension: str) -> dict[Path, str]:
    names = {p: p.with_suffix("." + extension).as_posix() for p in paths}
    counts: dict[str, int] = {}
    for value in names.values():
        counts[value.casefold()] = counts.get(value.casefold(), 0) + 1
    for path, value in list(names.items()):
        if counts[value.casefold()] > 1:
            digest = hashlib.sha256(path.as_posix().encode()).hexdigest()[:12]
            names[path] = path.with_name(
                f"{path.stem}--{path.suffix[1:]}-{digest}.{extension}"
            ).as_posix()
    if len({v.casefold() for v in names.values()}) != len(names):
        raise ValueError("Tên output xung đột sau khi thêm hậu tố; đổi tên input gây xung đột")
    return names


def build_plan(config: Config, workflow: dict, progress=None) -> list[PlanItem]:
    paths = scan_files(config)
    names = output_names(paths, config.format)
    items = []
    for index, relative in enumerate(paths):
        path = config.input / relative
        item = PlanItem(relative.as_posix(), names[relative])
        try:
            before = path.stat()
            image = read_image(str(path))
            item.source_hash = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise OSError("Input thay đổi trong khi quét; hãy chạy lại khi copy hoàn tất")
            item.size = before.st_size
            item.height, item.width = image.shape[:2]
            item.grayscale = detect(image, config.threshold, config.effective_detector)
            item.chain = select_chain(
                item.width, item.height, item.grayscale, config.scale, workflow
            )
            item.model = item.chain["ModelFilePath"]
        except OSError:
            raise
        except Exception as exc:
            item.error = f"{type(exc).__name__}: {exc}"
        items.append(item)
        if progress:
            progress(index + 1, len(paths), item.source)
    return items
