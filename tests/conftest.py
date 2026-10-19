from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from manga_colab.config import Config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    source = tmp_path / "input"
    source.mkdir()
    return Config(source, tmp_path / "output", tmp_path / "models", log_level="quiet")


@pytest.fixture
def write_image():
    def write(path: Path, color: bool = False, width: int = 24, height: int = 32) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.arange(width * height, dtype=np.uint8).reshape(height, width)
        image = (
            np.stack((array, np.full_like(array, 40), np.full_like(array, 180)), axis=-1)
            if color
            else array
        )
        Image.fromarray(image).save(path)
        return path

    return write
