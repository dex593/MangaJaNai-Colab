from __future__ import annotations

import csv
import io
import os
import time
from collections import Counter
from pathlib import Path

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .display import duration_label, model_label, size_label
from .storage import atomic_json


def write_report(
    directory: Path, rows: list[dict], total: int, started: float, config: dict, fatal: str = ""
) -> dict:
    seconds = time.monotonic() - started
    counts = Counter(row["status"] for row in rows)
    success = [row for row in rows if row["status"] == "success"]
    summary = dict(
        total=total,
        successful=counts["success"],
        failed=counts["failed"],
        skipped=counts["skipped"],
        pending=total - len(rows),
        seconds=seconds,
        images_per_second=len(success) / max(seconds, 0.001),
        input_bytes=sum(row.get("input_bytes", 0) for row in rows),
        output_bytes=sum(row.get("output_bytes", 0) for row in rows),
        models=dict(Counter(row["model"] for row in success)),
        oom_retries=sum(row.get("oom_retries", 0) for row in rows),
        fatal=fatal,
    )
    result = {"summary": summary, "config": config, "images": rows}
    atomic_json(directory / "report.json", result)
    fields = [
        "source",
        "output",
        "status",
        "model",
        "error",
        "seconds",
        "input_bytes",
        "output_bytes",
        "dimensions",
        "oom_retries",
        "reason",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    path = directory / "report.csv"
    temp = path.with_suffix(".csv.part")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        stream.write(buffer.getvalue())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return result


def display_report(console, report: dict) -> None:
    summary = report["summary"]
    table = Table(box=box.SIMPLE, expand=True, show_edge=False)
    table.add_column("Tổng ảnh", justify="right")
    table.add_column("Thành công", justify="right", style="green")
    table.add_column("Bỏ qua", justify="right", style="cyan")
    table.add_column("Lỗi", justify="right", style="red")
    table.add_column("Chưa xử lý", justify="right", style="yellow")
    table.add_row(
        *(str(summary[key]) for key in ("total", "successful", "skipped", "failed", "pending"))
    )
    details = Table.grid(padding=(0, 3))
    details.add_column(style="dim")
    details.add_column()
    for label, value in (
        ("Thời gian", duration_label(summary["seconds"])),
        ("Tốc độ trung bình", f"{summary['images_per_second']:.3f} ảnh thành công/s"),
        (
            "Dung lượng ảnh gốc / kết quả",
            f"{size_label(summary['input_bytes'])} / {size_label(summary['output_bytes'])}",
        ),
        ("Thử lại do thiếu bộ nhớ card", str(summary["oom_retries"])),
        ("Nơi lưu kết quả", report["config"].get("output", "—")),
    ):
        details.add_row(label, Text(value))
    errors = [row for row in report["images"] if row["status"] == "failed" or row.get("error")]
    interrupted = bool(summary["fatal"]) or summary["pending"] > 0
    title = "LƯỢT CHẠY ĐÃ DỪNG" if interrupted else ("HOÀN TẤT CÓ LỖI" if errors else "HOÀN TẤT")
    console.print(
        Panel(
            Group(table, "", details),
            title=title,
            title_align="left",
            border_style="yellow" if errors or interrupted else "green",
            padding=(1, 2),
        )
    )
    if summary["models"]:
        models = Table(box=box.SIMPLE, title="Model đã dùng trong lượt này", title_justify="left")
        models.add_column("Model", style="cyan")
        models.add_column("Ảnh thành công", justify="right")
        for name, count in summary["models"].items():
            models.add_row(model_label(name), str(count))
        console.print(models)
    if summary["fatal"]:
        console.print(Text(summary["fatal"], style="red"))
    if errors:
        failures = Table(box=box.SIMPLE, title="File cần kiểm tra", title_justify="left")
        failures.add_column("Ảnh", overflow="fold")
        failures.add_column("Nguyên nhân", style="red", overflow="fold")
        for row in errors:
            failures.add_row(Text(row["source"]), Text(row.get("error", row.get("reason", ""))))
        console.print(failures)
