from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def require_space(path: Path, amount: int) -> None:
    existing = path
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < amount:
        raise OSError(f"Không đủ dung lượng tại {path}: cần {amount:,} bytes, còn {free:,}")


def probe_directory(path: Path, write: bool = True) -> None:
    if write:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".manga-probe-") as stream:
            stream.write(b"ok")
            stream.flush()
            os.fsync(stream.fileno())
    elif not path.is_dir() or not os.access(path, os.R_OK | os.X_OK):
        raise OSError(f"Không đọc được thư mục: {path}")


def verify_image(path: Path, expected: tuple[int, int] | None = None) -> tuple[int, int]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Output thiếu/rỗng: {path}")
    import pyvips

    image = pyvips.Image.new_from_file(str(path), access="sequential", fail=True)
    image.avg()  # Force decoding every pixel; header-only checks miss truncated images.
    dimensions = (image.width, image.height)
    if expected and dimensions != expected:
        raise ValueError(f"Sai kích thước output: {dimensions}, cần {expected}")
    return dimensions
