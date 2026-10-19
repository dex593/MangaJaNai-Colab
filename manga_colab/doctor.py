from __future__ import annotations

import importlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import UPSTREAM_COMMIT
from .gpu import gpu_info
from .model_manager import ModelManager
from .storage import probe_directory


def doctor(
    backend: Path,
    input_path: Path | None = None,
    output: Path | None = None,
    models: Path | None = None,
    check_models: bool = False,
    cuda_device: int = 0,
) -> int:
    console = Console(highlight=False)
    table = Table(title="MangaJaNai environment doctor")
    table.add_column("Check")
    table.add_column("Result")
    errors = 0

    def check(name: str, operation) -> None:
        nonlocal errors
        try:
            value = operation()
            table.add_row(name, str(value))
        except Exception as exc:
            errors += 1
            table.add_row(name, f"ERROR: {type(exc).__name__}: {exc}", style="red")

    check("Python", lambda: f"{sys.version.split()[0]} / {platform.platform()} / {sys.executable}")
    check("CUDA", lambda: gpu_info(cuda_device, required=True))
    for module in (
        "torch",
        "torchvision",
        "numpy",
        "cv2",
        "pyvips",
        "chainner_ext",
        "spandrel",
        "safetensors",
        "rich",
    ):
        check(
            module,
            lambda module=module: getattr(
                importlib.import_module(module), "__version__", "import OK"
            ),
        )

    def native() -> str:
        import pyvips
        from PIL import ImageCms

        image = pyvips.Image.black(4, 4).cast("uchar")
        encoded = image.pngsave_buffer()
        pyvips.Image.new_from_buffer(encoded, "").icc_transform("srgb").avg()
        profiles = backend / "MangaJaNaiConverterGui/backend/ImageMagick"
        src = ImageCms.getOpenProfile(str(profiles / "Dot Gain 20%.icc"))
        dst = ImageCms.getOpenProfile(str(profiles / "Custom Gray Gamma 1.0.icc"))
        ImageCms.buildTransformFromOpenProfiles(src, dst, "L", "L")
        formats = []
        for suffix in (".png", ".jpg", ".webp", ".avif"):
            try:
                image.write_to_buffer(suffix)
                formats.append(suffix)
            except pyvips.Error:
                continue
        return f"libvips {pyvips.version(0)}.{pyvips.version(1)}.{pyvips.version(2)}; ICC OK; {formats}"

    check("Native codecs / ICC", native)

    def upstream() -> str:
        actual = subprocess.check_output(
            ["git", "-C", str(backend), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != UPSTREAM_COMMIT:
            raise ValueError(f"Cần {UPSTREAM_COMMIT}, nhận {actual}; chạy setup_colab.sh")
        source = backend / "MangaJaNaiConverterGui/backend/src"
        sys.path.insert(0, str(source))
        import spandrel_custom

        spandrel_custom.install(ignore_duplicates=True)
        from spandrel import MAIN_REGISTRY

        return (
            f"{actual}; custom FDAT registered ({len(MAIN_REGISTRY.architectures())} architectures)"
        )

    check("Upstream / FDAT", upstream)
    for label, path in (("Input", input_path), ("Output", output), ("Models", models)):
        if path:

            def directory(path=path, label=label):
                probe_directory(path, write=label != "Input")
                return f"{path}; free {shutil.disk_usage(path).free / 2**30:.2f} GiB"

            check(label, directory)
            if str(path).startswith("/content/drive/"):
                check(
                    "Drive mount",
                    lambda: "mounted"
                    if Path("/content/drive").is_mount()
                    else (_ for _ in ()).throw(OSError("Drive chưa mount; chạy cell Mount Drive")),
                )
    if models and check_models:
        manager = ModelManager(models)

        def validate_models() -> str:
            from spandrel import ModelLoader

            count = 0
            for name in manager.catalog["models"]:
                if (models / name).exists():
                    if not manager.valid(name):
                        raise ValueError(f"Checksum không hợp lệ: {name}")
                    descriptor = ModelLoader("cpu").load_from_file(models / name)
                    if descriptor.scale != int(name[0]):
                        raise ValueError(f"Model scale không khớp: {name}")
                    count += 1
            return f"{count} model load bằng Spandrel thành công"

        check("Model compatibility", validate_models)
    console.print(table)
    return 2 if errors else 0
