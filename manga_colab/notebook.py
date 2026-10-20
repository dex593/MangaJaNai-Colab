"""Stdlib-only subprocess bridge, loaded by the notebook before dependencies exist."""

from __future__ import annotations

import json
import os
import posixpath
import signal
import subprocess
import sys
from collections import deque
from html import escape
from pathlib import Path
from typing import Any

EVENT_PREFIX = "@@MANGA_COLAB_UI@@"
STEP_PREFIX = "@@MANGA_COLAB_STEP@@"


def html_card(title: str, body: str, tone: str = "blue") -> str:
    accent = {"blue": "#2563eb", "green": "#15803d", "yellow": "#a16207", "red": "#b91c1c"}[tone]
    return (
        '<section style="box-sizing:border-box;max-width:960px;margin:12px 0;padding:22px;'
        "background:#fff;color:#172033;border:1px solid #dce3ed;border-radius:14px;"
        'font:14px/1.6 system-ui,-apple-system,Segoe UI,sans-serif;overflow-wrap:anywhere">'
        f'<h3 style="font:600 19px/1.4 system-ui;margin:0 0 16px;color:{accent}">'
        f"{escape(title)}</h3>{body}</section>"
    )


def notice(title: str, message: str, displays: dict, key: str, tone: str = "blue") -> None:
    show_event(
        dict(
            id=key,
            text=f"{title}\n{message}\n",
            html=html_card(
                title, f'<p style="margin:0;white-space:pre-wrap">{escape(message)}</p>', tone
            ),
        ),
        displays,
    )


def refresh_drive(input_dir: str) -> None:
    """Refresh the mounted Drive from the notebook kernel before listing images."""
    path = posixpath.normpath(input_dir)
    if path != "/content/drive" and not path.startswith("/content/drive/"):
        return
    from google.colab import drive

    displays: dict[str, Any] = {}
    notice("Kết nối Google Drive", "Đang lưu dữ liệu còn chờ lên Google Drive…", displays, "drive")
    try:
        drive.flush_and_unmount(timeout_ms=60_000)
    except Exception as exc:
        raise RuntimeError(
            "Chưa lưu xong dữ liệu lên Drive. Đã dừng để bảo vệ kết quả; "
            "chờ Drive ổn định rồi chạy lại Bước 5. "
            "Không ngắt kết nối hoặc đặt lại Colab lúc này."
        ) from exc
    notice("Kết nối Google Drive", "Đang kết nối lại Drive để tìm ảnh mới…", displays, "drive")
    try:
        drive.mount("/content/drive", force_remount=True)
    except Exception as exc:
        raise RuntimeError(
            "Chưa kết nối lại được Google Drive. Chạy Bước 2 để kết nối, " "rồi chạy lại Bước 5."
        ) from exc
    if not Path(input_dir).is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh gốc: {input_dir}. "
            "Kiểm tra tài khoản Drive và đường dẫn ở Bước 1."
        )
    notice(
        "Google Drive đã sẵn sàng",
        "Đã kết nối lại Drive. Bắt đầu kiểm tra ảnh.",
        displays,
        "drive",
        "green",
    )


def show_event(event: dict, displays: dict) -> None:
    """Update a notebook display in place; retain readable output outside IPython."""
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except ImportError:
        get_ipython = None
    if get_ipython is not None and getattr(get_ipython(), "kernel", None) is not None:
        content = HTML(event["html"])
        key = event.get("id")
        if key and key in displays:
            displays[key].update(content)
        elif key:
            displays[key] = display(content, display_id=True)
        else:
            display(content)
    else:
        print(event["text"], end="", flush=True)


def run_live(
    args: list[str],
    *,
    cwd: str | Path,
    env: dict[str, str] | None = None,
    check: bool = False,
    log_path: str | Path | None = None,
    quiet: bool = False,
) -> subprocess.CompletedProcess:
    """Forward child output to the kernel, never relying on inherited OS stdout."""
    environment = {
        **os.environ,
        **(env or {}),
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "MANGA_COLAB_NOTEBOOK": "1",
    }
    displays: dict[str, Any] = {}
    tail: deque[str] = deque(maxlen=30)
    log = None
    has_steps = False
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        log = path.open("w", encoding="utf-8", buffering=1)
    try:
        with subprocess.Popen(
            args,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=os.name == "posix",
        ) as process:
            assert process.stdout is not None
            interrupted = False
            while True:
                try:
                    line = process.stdout.readline()
                    if not line:
                        returncode = process.wait()
                        break
                    event = None
                    if line.startswith(EVENT_PREFIX):
                        event = json.loads(line[len(EVENT_PREFIX) :])
                        text = event["text"]
                    else:
                        text = line
                    tail.append(text)
                    if log:
                        log.write(text)
                    if line.startswith(STEP_PREFIX):
                        has_steps = True
                        notice(
                            "Đang chuẩn bị chương trình",
                            line[len(STEP_PREFIX) :].strip(),
                            displays,
                            "setup",
                        )
                    elif not quiet:
                        if event:
                            show_event(event, displays)
                        else:
                            print(text, end="", flush=True)
                except KeyboardInterrupt:
                    if process.poll() is not None:
                        continue
                    if interrupted:
                        print(
                            "Dừng ngay theo yêu cầu. Lượt sau sẽ kiểm tra lại ảnh kết quả.",
                            flush=True,
                        )
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                    else:
                        interrupted = True
                        print("Đang dừng và lưu tiến độ; chờ tiến trình kết thúc…", flush=True)
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGINT)
                        else:
                            process.terminate()
            result = subprocess.CompletedProcess(args, returncode)
    finally:
        if log:
            log.close()
    if result.returncode and quiet:
        print("\nKhông hoàn tất. Chi tiết cuối log:\n" + "".join(tail), flush=True)
    if has_steps:
        notice(
            "Chuẩn bị xong" if not result.returncode else "Chưa chuẩn bị xong",
            "Đã hoàn tất các bước cài đặt."
            if not result.returncode
            else "Xem phần thông báo lỗi rồi chạy lại Bước 4.",
            displays,
            "setup",
            "green" if not result.returncode else "red",
        )
    if log_path and result.returncode:
        print(f"Log chi tiết: {log_path}", flush=True)
    if check:
        result.check_returncode()
    return result


if __name__ == "__main__":
    sys.exit("Module này được notebook nạp bằng runpy.run_path().")
