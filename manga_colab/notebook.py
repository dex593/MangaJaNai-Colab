"""Stdlib-only subprocess bridge, loaded by the notebook before dependencies exist."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

EVENT_PREFIX = "@@MANGA_COLAB_UI@@"
STEP_PREFIX = "@@MANGA_COLAB_STEP@@"


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
                        print("  • " + line[len(STEP_PREFIX) :].strip(), flush=True)
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
    if log_path:
        print(f"Log chi tiết: {log_path}", flush=True)
    if check:
        result.check_returncode()
    return result


if __name__ == "__main__":
    sys.exit("Module này được notebook nạp bằng runpy.run_path().")
