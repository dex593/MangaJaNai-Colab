from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config, ROOT, load_yaml
from .display import make_console, model_label


def add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="File YAML; CLI flags ghi đè YAML")
    for name, description in (
        ("input", "Thư mục ảnh nguồn (recursive)"),
        ("output", "Thư mục ảnh kết quả"),
        ("models", "Cache model trên Drive"),
        ("backend", "Checkout upstream đã pin"),
        ("workflow", "Workflow JSON có Chains.$values"),
        ("stage", "Thư mục local staging, ví dụ /content/manga-stage"),
    ):
        parser.add_argument("--" + name, type=Path, default=argparse.SUPPRESS, help=description)
    for name, kind, description in (
        ("scale", float, "Target scale từ 1 đến 4"),
        ("threshold", float, "Grayscale threshold, mặc định 12"),
        ("tile", int, "0=upstream auto estimate; hoặc px >=32"),
        ("cuda-device", int, "CUDA index, mặc định 0; KHÔNG phải GUI backend index"),
        ("quality", int, "JPEG/WebP/AVIF quality 1–100"),
    ):
        parser.add_argument("--" + name, type=kind, default=argparse.SUPPRESS, help=description)
    for name, choices, description in (
        ("format", ["png", "jpeg", "webp", "avif"], "Định dạng output"),
        ("detector", ["upstream", "corrected"], "Giữ upstream hoặc sửa pure B/W + uint8 overflow"),
        ("precision", ["auto", "fp32"], "Auto FP16 nếu GPU/model hỗ trợ"),
        ("log-level", ["quiet", "normal", "verbose", "debug"], "Độ chi tiết log"),
    ):
        parser.add_argument(
            "--" + name, choices=choices, default=argparse.SUPPRESS, help=description
        )
    for name, description in (
        ("resume", "Xác minh và bỏ qua output đã hoàn tất"),
        ("overwrite", "Cho phép thay thế output"),
        ("retry-failed", "Thử lại lỗi lần trước"),
        ("lossless", "WebP lossless; PNG luôn lossless"),
    ):
        parser.add_argument(
            "--" + name,
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
            help=description,
        )
    parser.add_argument(
        "--auto-model", action="store_true", help="Tự chọn model cho từng ảnh (luôn bật)"
    )


def from_args(args: argparse.Namespace) -> Config:
    values = load_yaml(args.config) if args.config else {}
    values.update({k: v for k, v in vars(args).items() if k in Config.__dataclass_fields__})
    if "input" not in values:
        raise ValueError("Cần --input hoặc input trong YAML")
    if "output" not in values:
        values["output"] = Path(values["input"]).parent / (Path(values["input"]).name + "-upscaled")
    if "models" not in values:
        values["models"] = Path.home() / ".cache/mangajanai/models"
    return Config(**values)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="MangaJaNai-Colab — AI backend gốc, batch an toàn, tự chọn model"
    )
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="Upscale hoặc lập plan; lỗi trả exit code 2, Ctrl+C trả 130"
    )
    add_config(run)
    run.add_argument(
        "--dry-run", action="store_true", help="Chỉ quét/lập plan; không download hay inference"
    )
    scan = commands.add_parser("scan", help="Đọc nội dung ảnh, chọn model và ghi plan.json")
    add_config(scan)
    report = commands.add_parser("report", help="Hiển thị báo cáo của lượt chạy đã lưu")
    report.add_argument("--output", type=Path, required=True, help="Thư mục output của lượt chạy")
    doctor = commands.add_parser(
        "doctor", help="Kiểm tra GPU, native codecs, upstream, model và đường dẫn"
    )
    doctor.add_argument("--backend", type=Path, default=ROOT / ".upstream/MangaJaNaiConverterGui")
    for path in ("input", "output", "models"):
        doctor.add_argument("--" + path, type=Path, help="Đường dẫn cần kiểm tra")
    doctor.add_argument("--cuda-device", type=int, default=0, help="CUDA GPU index")
    doctor.add_argument(
        "--check-models",
        action="store_true",
        help="Xác minh SHA-256 và thử load model cached trên CPU",
    )
    models = commands.add_parser(
        "models", help="Catalog model chính thức và selective ZIP download"
    )
    operations = models.add_subparsers(dest="operation", required=True)
    listing = operations.add_parser("list", help="Liệt kê filename, size và cache status")
    listing.add_argument("--models", type=Path, default=Path.home() / ".cache/mangajanai/models")
    download = operations.add_parser("download", help="Tải những model thực sự cần cho input")
    add_config(download)
    return root


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    console = make_console()
    try:
        if args.command == "report":
            from .report import display_report

            path = args.output / ".manga-colab/report.json"
            if not path.is_file():
                raise ValueError(f"Chưa có báo cáo: {path}. Xem log Cell 5 để biết lỗi.")
            display_report(console, json.loads(path.read_text(encoding="utf-8")))
            console.print(f"JSON / CSV / log: {path.parent}", style="dim")
            return 0
        if args.command == "doctor":
            from .doctor import doctor

            return doctor(
                args.backend,
                args.input,
                args.output,
                args.models,
                args.check_models,
                args.cuda_device,
            )
        if args.command == "models":
            from .model_manager import ModelManager

            if args.operation == "list":
                manager = ModelManager(args.models)
                for name, entry in manager.catalog["models"].items():
                    console.print(
                        f"{'HIT ' if manager.valid(name) else 'MISS'} {entry['size']/2**20:7.2f} MiB  {name}"
                    )
                return 0
            from .logger import download_progress
            from .model_selector import load_workflow
            from .scanner import build_plan

            config = from_args(args)
            plan = build_plan(config, load_workflow(config.workflow))
            if not plan:
                raise ValueError("Input rỗng hoặc không tồn tại")
            with download_progress(console) as progress:

                def factory(name: str, size: int):
                    task = progress.add_task(model_label(name), total=size)
                    return lambda done, _: progress.update(
                        task, completed=done, total=max(done, size)
                    )

                ModelManager(config.models).ensure(
                    [item.model for item in plan if item.model], factory
                )
            for item in plan:
                if item.error:
                    console.print(f"{item.source}: {item.error}", style="red", markup=False)
            return 2 if any(item.error for item in plan) else 0
        from .pipeline import run

        return run(from_args(args), dry_run=args.command == "scan" or args.dry_run)
    except KeyboardInterrupt:
        console.print("Đã dừng; chạy lại --resume.", style="yellow")
        return 130
    except Exception as exc:
        console.print(f"{type(exc).__name__}: {exc}", style="red", markup=False)
        if getattr(args, "log_level", "normal") == "debug":
            import traceback

            traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
