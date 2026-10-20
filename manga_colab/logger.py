from __future__ import annotations

import logging
import re
import time
from collections import deque
from pathlib import Path

from rich.console import Console
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
from rich.text import Text

from .display import AppProgress, Card, NotebookConsole, make_console, model_label


def error_message(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = exc.__cause__
    while cause is not None:
        parts.append(f"{type(cause).__name__}: {cause}")
        cause = cause.__cause__
    return " ← ".join(parts)


def user_message(message: str) -> str:
    lowered = message.lower()
    for match, help_text in (
        (
            "cuda out of memory",
            "Card đồ họa không đủ bộ nhớ. Dừng các tác vụ khác đang dùng GPU rồi chạy lại.",
        ),
        (
            "cuda oom",
            "Card đồ họa không đủ bộ nhớ để xử lý ảnh này. Thử lại trên GPU có nhiều bộ nhớ hơn.",
        ),
        (
            "out of memory",
            "Không đủ bộ nhớ để xử lý ảnh này. Xem chi tiết lỗi bên dưới để biết bộ nhớ nào đã hết.",
        ),
        ("no space left", "Đã hết dung lượng lưu trữ. Dọn bớt tệp ở nơi lưu kết quả rồi chạy lại."),
        ("transport endpoint", "Kết nối Google Drive bị gián đoạn. Chạy lại Bước 2 rồi Bước 5."),
        (
            "permission denied",
            "Không có quyền đọc hoặc lưu tệp. Kiểm tra quyền của thư mục trên Drive.",
        ),
        (
            "not a known file format",
            "Không đọc được ảnh. Tệp có thể bị hỏng hoặc định dạng chưa được hỗ trợ. Thử mở ảnh gốc rồi tải lại lên Drive.",
        ),
        (
            "cannot identify image file",
            "Không đọc được ảnh. Thử mở ảnh gốc để kiểm tra và tải lại lên Drive.",
        ),
    ):
        if match in lowered:
            return help_text
    return re.sub(r"\b(?:[A-Za-z_]\w*)?(?:Error|Exception):\s*", "", message)


class AppLogHandler(logging.Handler):
    def __init__(self, console: Console) -> None:
        super().__init__()
        self.console = console
        self.recent: deque[tuple[str, str]] = deque(maxlen=5)

    def emit(self, record: logging.LogRecord) -> None:
        style = (
            "red"
            if record.levelno >= logging.ERROR
            else ("yellow" if record.levelno >= logging.WARNING else "default")
        )
        message = user_message(record.getMessage())
        text = Text(message, style=style)
        if isinstance(self.console, NotebookConsole):
            label = (
                "Cần kiểm tra"
                if record.levelno >= logging.ERROR
                else ("Lưu ý" if record.levelno >= logging.WARNING else "Thông tin")
            )
            self.recent.append((label, message))
            self.console.frame(
                "activity",
                Card(
                    "Thông báo",
                    rows=list(self.recent),
                    tone="red"
                    if record.levelno >= logging.ERROR
                    else ("yellow" if record.levelno >= logging.WARNING else "blue"),
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
            "normal": logging.WARNING,
            "verbose": logging.INFO,
            "debug": logging.DEBUG,
        }[level]
    )
    logger.addHandler(display)
    return console, logger


def dashboard(console: Console, values: dict, counts: dict, technical: dict) -> None:
    console.print(
        Card(
            "2 · Chuẩn bị Upscale",
            rows=list(values.items()),
            metrics=list(counts.items()),
            details=list(technical.items()),
            note="Ảnh đã làm xong và còn hợp lệ sẽ được giữ lại khi chọn chỉ làm ảnh mới.",
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
        title="Tải model còn thiếu · Lần sau sẽ dùng lại",
    )


def scan_progress(console: Console, quiet: bool = False) -> AppProgress:
    return AppProgress(
        TextColumn("Kiểm tra ảnh", style="cyan"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("ảnh"),
        console=console,
        disable=quiet,
        refresh_per_second=2,
        title="1 · Tìm ảnh trong thư mục",
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
            title="3 · Đang Upscale",
        )
        self.overall = self.progress.add_task(
            "Tiến độ toàn bộ", total=total, detail=f"0/{total} ảnh", eta=True
        )
        self.current = self.progress.add_task("AI xử lý ảnh hiện tại", total=1, detail="Chờ xử lý")
        self.started = time.monotonic()
        self.done = 0
        self.successful = 0
        self.total = total

    def image(
        self,
        index: int,
        source: str,
        width: int,
        height: int,
        grayscale: bool,
        model: str,
        scale: float = 2,
    ) -> None:
        target_height = round(height * scale)
        target_width = round(width * target_height / height)
        self.progress.heading = Card(
            "Ảnh đang xử lý",
            rows=[
                ("Ảnh đang làm", f"{index}/{self.total} · {source}"),
                ("Kích thước", f"{width} × {height} → {target_width} × {target_height} px"),
                ("Loại ảnh", "Đen trắng" if grayscale else "Ảnh màu"),
                ("Model", model_label(model)),
            ],
        )
        self.progress.reset(self.current, total=1, detail="Chuẩn bị", failed=False)

    def update(self, done: int, total: int, stage: str) -> None:
        if stage.startswith(("Hoàn thiện", "Đã lưu")):
            self.progress.update(self.current, detail="Đang lưu và kiểm tra ảnh kết quả")
        else:
            self.progress.update(self.current, completed=done, total=total, detail=stage)

    def advance(self, success: bool = True) -> None:
        self.done += 1
        self.successful += int(success)
        rate = self.successful / max(time.monotonic() - self.started, 0.001)
        self.progress.update(
            self.current,
            **({"completed": 1, "total": 1} if success else {}),
            failed=not success,
            detail="Đã lưu và kiểm tra kết quả"
            if success
            else "Ảnh chưa hoàn tất · xem nguyên nhân",
        )
        self.progress.update(
            self.overall,
            advance=1,
            detail=f"{self.done}/{self.total} ảnh · Đã lưu {self.successful} · "
            f"Lỗi {self.done - self.successful} · {rate * 60:.1f} ảnh/phút",
        )
