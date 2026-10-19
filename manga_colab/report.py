from __future__ import annotations

import csv
import io
import os
import time
from collections import Counter
from pathlib import Path

from rich.table import Table

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
    table = Table(title="Kết quả")
    table.add_column("Thống kê", style="cyan")
    table.add_column("Giá trị")
    for key, value in report["summary"].items():
        table.add_row(key, f"{value:.3f}" if isinstance(value, float) else str(value))
    console.print(table)
    for row in report["images"]:
        if row["status"] == "failed" or row.get("error"):
            console.print(
                f"{row['source']}: {row.get('error', row.get('reason', ''))}",
                style="red",
                markup=False,
            )
