from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

from rich.console import Console
from rich.console import Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.text import Text

from .display import AppProgress, NotebookConsole, make_console, model_label


def error_message(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = exc.__cause__
    while cause is not None:
        parts.append(f"{type(cause).__name__}: {cause}")
        cause = cause.__cause__
    return " ← ".join(parts)


class AppLogHandler(logging.Handler):
    def __init__(self, console: Console) -> None:
        super().__init__()
        self.console = console
        self.recent: deque[Text] = deque(maxlen=8)

    def emit(self, record: logging.LogRecord) -> None:
        style = (
            "red"
            if record.levelno >= logging.ERROR
            else ("yellow" if record.levelno >= logging.WARNING else "default")
        )
        text = Text(record.getMessage(), style=style)
        if isinstance(self.console, NotebookConsole):
            self.recent.append(text)
            self.console.frame(
                "activity",
                Panel(
                    Group(*self.recent),
                    title="Hoạt động gần đây",
                    title_align="left",
                    border_style="dim",
                ),
            )
        else:
            self.console.print(text)


def configure(path: Path, level: str) -> tuple[Console, logging.Logger]:
    console = make_console()
    logger = logging.getLogger("manga_colab")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for previous in logger.handlers:
        previous.close()
    logger.handlers.clear()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    display = AppLogHandler(console)
    display.setLevel(
        {
            "quiet": logging.ERROR,
            "normal": logging.INFO,
            "verbose": logging.DEBUG,
            "debug": logging.DEBUG,
        }[level]
    )
    logger.addHandler(display)
    return console, logger


def dashboard(console: Console, values: dict) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(overflow="fold")
    for key, value in values.items():
        table.add_row(key, Text(str(value)))
    console.print(
        Panel(
            table,
            title="MANGA JANAI  /  COLAB",
            subtitle="UPSCALE · TỰ CHỌN MODEL AI",
            title_align="left",
            subtitle_align="right",
            padding=(1, 2),
            border_style="cyan",
        )
    )


def download_progress(console: Console, quiet: bool = False) -> Progress:
    return AppProgress(
        TextColumn("{task.description}", markup=False),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=quiet,
        refresh_per_second=2,
        title="Tải model chính thức",
    )


def scan_progress(console: Console, quiet: bool = False) -> AppProgress:
    return AppProgress(
        TextColumn("Quét ảnh", style="cyan"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("ảnh"),
        console=console,
        disable=quiet,
        refresh_per_second=2,
        title="Đọc ảnh và kiểm tra danh sách",
    )


class BatchProgress:
    def __init__(self, console: Console, total: int, quiet: bool = False):
        self.progress = AppProgress(
            TextColumn("{task.description}", markup=False),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("còn"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[detail]}", markup=False),
            console=console,
            disable=quiet or total == 0,
            refresh_per_second=2,
            title="Tiến độ Upscale",
        )
        self.overall = self.progress.add_task("Toàn lượt", total=total, detail=f"0/{total} ảnh")
        self.current = self.progress.add_task("Ảnh hiện tại", total=1, detail="Chờ xử lý")
        self.started = time.monotonic()
        self.done = 0
        self.successful = 0
        self.total = total

    def image(
        self, index: int, source: str, width: int, height: int, grayscale: bool, model: str
    ) -> None:
        details = Table.grid(padding=(0, 2))
        details.add_column(style="dim", no_wrap=True)
        details.add_column(overflow="fold")
        details.add_row("Ảnh", Text(f"{index}/{self.total}  {source}", style="bold"))
        details.add_row(
            "Ảnh gốc", f"{width:,} × {height:,} px · {'Đen trắng' if grayscale else 'Màu'}"
        )
        details.add_row("Model", model_label(model))
        details.add_row("Tệp AI", "Đã tải và kiểm tra")
        self.progress.heading = details
        self.progress.reset(self.current, total=1, detail="Chuẩn bị")

    def update(self, done: int, total: int, stage: str) -> None:
        self.progress.update(self.current, completed=done, total=total, detail=stage)

    def advance(self, success: bool = True) -> None:
        self.done += 1
        self.successful += int(success)
        rate = self.successful / max(time.monotonic() - self.started, 0.001)
        self.progress.update(
            self.current,
            completed=1,
            total=1,
            detail="Đã lưu và kiểm tra kết quả" if success else "Lỗi — xem chi tiết",
        )
        self.progress.update(
            self.overall, advance=1, detail=f"{self.done}/{self.total} ảnh · {rate:.3f} ảnh/s"
        )
