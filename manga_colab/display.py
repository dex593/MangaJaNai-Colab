"""Rich renderables shared by CLI and the subprocess-to-notebook transport."""

from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from html import escape
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress
from rich.terminal_theme import TerminalTheme
from rich.text import Text
from rich.table import Table

from .notebook import EVENT_PREFIX, html_card
from .model_selector import EXTRA_MODELS

THEME = TerminalTheme(
    (15, 23, 42),
    (226, 232, 240),
    [
        (15, 23, 42),
        (248, 113, 113),
        (52, 211, 153),
        (251, 191, 36),
        (96, 165, 250),
        (192, 132, 252),
        (34, 211, 238),
        (226, 232, 240),
    ],
)
HTML_FORMAT = (
    '<div style="background:{background};color:{foreground};padding:18px 22px;'
    'border:1px solid #334155;border-radius:12px;margin:8px 0;overflow-x:auto">'
    '<pre style="margin:0;font:14px/1.6 ui-monospace,monospace;white-space:pre;'
    'color:inherit;background:transparent">{code}</pre></div>'
)


def html_rows(rows: list[tuple[str, Any]]) -> str:
    return (
        '<dl style="margin:0">'
        + "".join(
            '<div style="display:grid;grid-template-columns:minmax(105px,25%) minmax(0,1fr);'
            'gap:12px;padding:7px 0;border-bottom:1px solid #eef2f6">'
            f'<dt style="color:#596579">{escape(str(key))}</dt>'
            f'<dd style="margin:0;white-space:pre-wrap">{escape(str(value))}</dd></div>'
            for key, value in rows
        )
        + "</dl>"
    )


@dataclass
class Card:
    title: str
    rows: list[tuple[str, Any]] = field(default_factory=list)
    metrics: list[tuple[str, Any]] = field(default_factory=list)
    note: str = ""
    details: list[tuple[str, Any]] = field(default_factory=list)
    tone: str = "blue"

    def __rich__(self) -> Panel:
        table = Table.grid(padding=(0, 2), expand=True)
        table.add_column(style="dim", ratio=1)
        table.add_column(ratio=3, overflow="fold")
        for key, value in self.metrics + self.rows:
            table.add_row(Text(str(key)), Text(str(value)))
        contents: list[Any] = [table]
        if self.note:
            contents.extend(["", Text(self.note)])
        if self.details:
            details = Table.grid(padding=(0, 2))
            for key, value in self.details:
                details.add_row(Text(str(key), style="dim"), Text(str(value), style="dim"))
            contents.extend(["", details])
        return Panel(
            Group(*contents),
            title=self.title,
            title_align="left",
            border_style=self.tone,
            padding=(1, 2),
        )

    def html(self) -> str:
        metrics = (
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));'
            'gap:10px;margin:0 0 18px">'
            + "".join(
                '<div style="padding:14px;background:#f3f6fb;border-radius:10px">'
                f'<div style="font-size:27px;font-weight:650;line-height:1.2">{escape(str(value))}</div>'
                f'<div style="color:#596579;margin-top:6px">{escape(str(key))}</div></div>'
                for key, value in self.metrics
            )
            + "</div>"
            if self.metrics
            else ""
        )
        note = (
            f'<p style="margin:16px 0 0;color:#46546a">{escape(self.note)}</p>' if self.note else ""
        )
        details = (
            '<details style="margin-top:16px"><summary style="cursor:pointer;color:#596579">'
            "Thông tin kỹ thuật và báo cáo</summary>" + html_rows(self.details) + "</details>"
            if self.details
            else ""
        )
        return html_card(self.title, metrics + html_rows(self.rows) + note + details, self.tone)


class NotebookConsole(Console):
    def __init__(self) -> None:
        super().__init__(
            file=io.StringIO(),
            width=100,
            highlight=False,
            markup=False,
            force_terminal=False,
            force_jupyter=False,
            legacy_windows=False,
        )
        # Capture before upstream redirects sys.stdout to its debug log.
        self.transport = sys.stdout

    def frame(self, key: str | None, *objects: Any, html: str | None = None, **kwargs: Any) -> None:
        rendered = Console(
            file=io.StringIO(),
            record=True,
            width=self.width,
            highlight=False,
            markup=False,
            force_jupyter=False,
            legacy_windows=False,
        )
        # The HTML card supplies the border; keep terminal box characters out of Colab.
        if len(objects) == 1 and isinstance(objects[0], Card):
            html = objects[0].html()
        if len(objects) == 1 and isinstance(objects[0], Panel):
            panel = objects[0]
            title = Text(str(panel.title or ""), style=panel.border_style)
            contents = [title, "", panel.renderable]
            if panel.subtitle:
                contents.extend(["", Text(str(panel.subtitle), style="dim")])
            objects = (Group(*contents),)
        rendered.print(*objects, **kwargs)
        plain = rendered.export_text(clear=False)
        if html is None:
            html = rendered.export_html(theme=THEME, inline_styles=True, code_format=HTML_FORMAT)
        event = dict(id=key, text=plain, html=html)
        self.transport.write(EVENT_PREFIX + json.dumps(event, ensure_ascii=False) + "\n")
        self.transport.flush()

    def print(self, *objects: Any, **kwargs: Any) -> None:
        self.frame(None, *objects, **kwargs)


def make_console() -> Console:
    if os.environ.get("MANGA_COLAB_NOTEBOOK") == "1":
        return NotebookConsole()
    return Console(file=sys.stdout, highlight=False, markup=False)


class AppProgress(Progress):
    """Terminal Live or explicit HTML updates; no terminal cursor codes in Colab."""

    def __init__(self, *columns: Any, title: str, **kwargs: Any) -> None:
        self.title = title
        self.heading: Any = None
        self.frame_id = "progress-" + uuid.uuid4().hex
        self.active = False
        self.last_render = 0.0
        notebook = isinstance(kwargs.get("console"), NotebookConsole)
        super().__init__(
            *columns,
            auto_refresh=not notebook,
            redirect_stdout=False,
            redirect_stderr=False,
            **kwargs,
        )

    def get_renderable(self) -> Panel:
        contents = super().get_renderable()
        if self.heading is not None:
            heading = (
                self.heading.__rich__().renderable
                if isinstance(self.heading, Card)
                else self.heading
            )
            contents = Group(heading, "", contents)
        return Panel(
            contents, title=self.title, title_align="left", border_style="cyan", padding=(1, 2)
        )

    def start(self) -> None:
        self.active = True
        if isinstance(self.console, NotebookConsole):
            self.last_render = 0
            self.refresh()
        else:
            super().start()

    def stop(self) -> None:
        if isinstance(self.console, NotebookConsole):
            self.last_render = 0
            self.refresh()
        else:
            super().stop()
        self.active = False

    def html(self) -> str:
        body = html_rows(self.heading.rows) if isinstance(self.heading, Card) else ""
        for task in self.tasks:
            if not task.visible:
                continue
            known = task.total is not None and task.total > 0
            percent = min(100, max(0, task.percentage)) if known else 0
            label = f"{percent:.0f}%" if known else "Đang kiểm tra…"
            if task.fields.get("failed"):
                label = "Chưa hoàn tất"
            detail = str(task.fields.get("detail", ""))
            if task.fields.get("download"):
                total = size_label(int(task.total)) if known else "chưa biết tổng dung lượng"
                detail = f"{size_label(int(task.completed))} / {total}"
                if task.speed:
                    detail += f" · {size_label(int(task.speed))}/s"
            elif task.fields.get("scan"):
                detail = (
                    f"Đã kiểm tra {int(task.completed)}/{int(task.total)} ảnh"
                    if known
                    else "Đang tìm ảnh trong các thư mục…"
                )
            eta = ""
            if not task.finished and task.fields.get("eta"):
                eta = (
                    f"Còn khoảng {duration_label(task.time_remaining)}"
                    if task.time_remaining is not None
                    else "Đang tính thời gian còn lại"
                )
            body += (
                '<div style="margin-top:18px">'
                '<div style="display:flex;justify-content:space-between;gap:12px;font-weight:600">'
                f"<span>{escape(task.description)}</span><span>{label}</span></div>"
                '<div style="height:9px;background:#e8eef7;border-radius:6px;margin:9px 0">'
                f'<div role="progressbar" aria-label="{escape(task.description, quote=True)}" '
                f'aria-valuenow="{percent:.2f}" aria-valuemin="0" aria-valuemax="100" '
                f'style="height:100%;width:{percent:.2f}%;background:#2563eb;border-radius:6px"></div></div>'
                '<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;'
                f'color:#596579"><span>{escape(detail)}</span><span>{escape(eta)}</span></div></div>'
            )
        return html_card(self.title, body)

    def refresh(self) -> None:
        if isinstance(self.console, NotebookConsole):
            if self.disable or not self.active or time.monotonic() - self.last_render < 0.5:
                return
            self.last_render = time.monotonic()
            self.console.frame(self.frame_id, self.get_renderable(), html=self.html())
        else:
            super().refresh()

    def update(self, task_id: Any, **kwargs: Any) -> None:
        super().update(task_id, **kwargs)
        if isinstance(self.console, NotebookConsole):
            self.refresh()


def model_label(filename: str) -> str:
    for entry in EXTRA_MODELS.values():
        if filename == entry["filename"]:
            return entry["label"]
    parts = filename.rsplit(".", 1)[0].split("_")
    if len(parts) >= 4 and parts[1] == "MangaJaNai":
        return f"MangaJaNai {parts[2]} · {parts[0]} · {parts[3]}"
    if len(parts) >= 4 and parts[1] == "IllustrationJaNai":
        return f"IllustrationJaNai {parts[2]} · {parts[0]} · {parts[3]}"
    return filename


def size_label(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def duration_label(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"
