from __future__ import annotations

import importlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from rich import box
from rich.table import Table
from rich.text import Text

from .config import UPSTREAM_COMMIT
from .display import make_console
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
    console = make_console()
    table = Table(
        title="MANGA JANAI  /  KIỂM TRA MÔI TRƯỜNG",
        title_justify="left",
        box=box.SIMPLE,
        padding=(0, 1),
    )
    table.add_column("Hạng mục", style="cyan", no_wrap=True)
    table.add_column("Trạng thái", no_wrap=True)
    table.add_column("Chi tiết", overflow="fold")
    errors = 0

    def check(name: str, operation) -> None:
        nonlocal errors
        try:
            value = operation()
            table.add_row(name, Text("✓ OK", style="green"), Text(str(value)))
        except Exception as exc:
            errors += 1
            table.add_row(name, Text("✗ LỖI", style="red"), Text(f"{type(exc).__name__}: {exc}"))

    check("Python", lambda: f"{sys.version.split()[0]} · {platform.system()}\n{sys.executable}")

    def gpu_details() -> str:
        info = gpu_info(cuda_device, required=True)
        return (
            f"{info['name']} · CUDA {info['cuda']} · PyTorch {info['torch']}\n"
            f"VRAM {info['free']/2**30:.2f} / {info['total']/2**30:.2f} GiB trống\n"
            f"Compute {info['compute']} · FP16 {info['fp16']} · BF16 {info['bf16']}"
        )

    check("GPU / CUDA", gpu_details)
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
        return f"libvips {pyvips.version(0)}.{pyvips.version(1)}.{pyvips.version(2)} · ICC OK\n{', '.join(formats)}"

    check("Đọc / lưu ảnh", native)

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

        return f"{actual[:12]} · FDAT sẵn sàng ({len(MAIN_REGISTRY.architectures())} kiến trúc)"

    check("Bộ xử lý ảnh", upstream)
    if any(
        path and str(path).startswith("/content/drive/") for path in (input_path, output, models)
    ):
        check(
            "Google Drive",
            lambda: "Đã kết nối"
            if Path("/content/drive").is_mount()
            else (_ for _ in ()).throw(OSError("Chưa kết nối Drive; chạy lại Bước 2")),
        )
    for label, path in (("Input", input_path), ("Output", output), ("Models", models)):
        if path:

            def directory(path=path, label=label):
                probe_directory(path, write=label != "Input")
                return f"{path}\nDung lượng trống: {shutil.disk_usage(path).free / 2**30:.2f} GiB"

            check(label, directory)
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

        check("Model đã tải", validate_models)
    console.print(table)
    console.print(
        "Môi trường sẵn sàng." if not errors else f"Cần sửa {errors} mục lỗi ở trên.",
        style="green" if not errors else "red",
    )
    return 2 if errors else 0
