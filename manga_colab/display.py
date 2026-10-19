"""Rich renderables shared by CLI and the subprocess-to-notebook transport."""

from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress
from rich.terminal_theme import TerminalTheme
from rich.text import Text

from .notebook import EVENT_PREFIX
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

    def frame(self, key: str | None, *objects: Any, **kwargs: Any) -> None:
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
        if len(objects) == 1 and isinstance(objects[0], Panel):
            panel = objects[0]
            title = Text(str(panel.title or ""), style=panel.border_style)
            contents = [title, "", panel.renderable]
            if panel.subtitle:
                contents.extend(["", Text(str(panel.subtitle), style="dim")])
            objects = (Group(*contents),)
        rendered.print(*objects, **kwargs)
        plain = rendered.export_text(clear=False)
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
            contents = Group(self.heading, "", contents)
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

    def refresh(self) -> None:
        if isinstance(self.console, NotebookConsole):
            if self.disable or not self.active or time.monotonic() - self.last_render < 0.5:
                return
            self.last_render = time.monotonic()
            self.console.frame(self.frame_id, self.get_renderable())
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
