from __future__ import annotations

import csv
import io
import os
import time
from collections import Counter
from pathlib import Path

from .display import Card, duration_label, model_label, size_label
from .logger import user_message
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
    errors = [row for row in report["images"] if row["status"] == "failed" or row.get("error")]
    kept = sum(row["status"] == "skipped" and not row.get("error") for row in report["images"])
    interrupted = bool(summary["fatal"]) or summary["pending"] > 0
    title = (
        "Đã dừng · Kết quả hoàn tất được giữ lại"
        if interrupted
        else ("Đã kết thúc · Có ảnh cần kiểm tra" if errors else "Hoàn tất")
    )
    rows = [
        ("Nơi lưu kết quả", report["config"].get("output", "—")),
        ("Thời gian", duration_label(summary["seconds"])),
        ("Tốc độ trung bình", f"{summary['images_per_second'] * 60:.1f} ảnh/phút"),
    ]
    if summary["fatal"]:
        rows.append(("Lý do dừng", user_message(summary["fatal"])))
    details = [
        (
            "Dung lượng ảnh gốc / kết quả",
            f"{size_label(summary['input_bytes'])} / {size_label(summary['output_bytes'])}",
        ),
        ("Thử lại do thiếu bộ nhớ card", str(summary["oom_retries"])),
        (
            "Báo cáo JSON, CSV và log",
            str(Path(report["config"].get("output", ".")) / ".manga-colab"),
        ),
    ]
    details.extend(
        (model_label(name), f"{count} ảnh đã lưu") for name, count in summary["models"].items()
    )
    if summary["fatal"]:
        details.append(("Chi tiết lỗi", summary["fatal"]))
    note = (
        "Xem nguyên nhân từng ảnh bên dưới. Sửa lỗi rồi bật “Thử lại ảnh bị lỗi” ở Bước 1."
        if errors
        else (
            "Chạy lại Bước 5 để tiếp tục những ảnh chưa hoàn tất."
            if interrupted
            else "Mở thư mục lưu kết quả trên Google Drive để xem ảnh. Thêm ảnh mới rồi chạy lại Bước 5 khi cần."
        )
    )
    console.print(
        Card(
            "4 · " + title,
            metrics=[
                ("Tổng ảnh", summary["total"]),
                ("Đã lưu", summary["successful"]),
                ("Đã làm từ trước", kept),
                ("Cần kiểm tra", len(errors)),
                ("Chưa xử lý", summary["pending"]),
            ],
            rows=rows,
            details=details,
            note=note,
            tone="yellow" if errors or interrupted else "green",
        )
    )
    if errors:
        console.print(
            Card(
                "Ảnh cần kiểm tra",
                rows=[
                    (row["source"], user_message(row.get("error", row.get("reason", ""))))
                    for row in errors
                ],
                details=[
                    (row["source"], row.get("error", row.get("reason", ""))) for row in errors
                ],
                tone="red",
            )
        )
