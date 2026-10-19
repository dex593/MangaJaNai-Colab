from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table


def error_message(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = exc.__cause__
    while cause is not None:
        parts.append(f"{type(cause).__name__}: {cause}")
        cause = cause.__cause__
    return " ← ".join(parts)


def configure(path: Path, level: str) -> tuple[Console, logging.Logger]:
    console = Console(
        file=sys.stdout,
        highlight=False,
        force_terminal=True if os.environ.get("MANGA_COLAB_LIVE") == "1" else None,
    )
    logger = logging.getLogger("manga_colab")
    logger.setLevel(logging.DEBUG)
    for previous in logger.handlers:
        previous.close()
    logger.handlers.clear()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    display = RichHandler(console=console, show_time=False, show_path=False, markup=False)
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
    for key, value in values.items():
        table.add_row(f"[cyan]{key}[/]", str(value))
    console.print(Panel(table, title="MANGA JANAI COLAB UPSCALER", border_style="cyan"))


def download_progress(console: Console, quiet: bool = False) -> Progress:
    return Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=quiet,
        refresh_per_second=2,
    )


class BatchProgress:
    def __init__(self, console: Console, total: int, quiet: bool = False):
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[detail]}"),
            console=console,
            disable=quiet,
            refresh_per_second=2,
        )
        self.overall = self.progress.add_task("Total", total=total, detail="")
        self.current = self.progress.add_task("Image", total=1, detail="")
        self.started = time.monotonic()
        self.done = 0

    def update(self, done: int, total: int, stage: str) -> None:
        self.progress.update(self.current, completed=done, total=total, detail=stage)

    def advance(self) -> None:
        self.done += 1
        rate = self.done / max(time.monotonic() - self.started, 0.001)
        self.progress.update(self.overall, advance=1, detail=f"{rate:.3f} images/s")
