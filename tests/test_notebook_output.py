import io
import json
import os
import subprocess
import sys
import types

import pytest
from rich.text import Text

from manga_colab import notebook
from manga_colab.display import AppProgress, NotebookConsole
from manga_colab.logger import BatchProgress


def test_notebook_updates_existing_html_display(monkeypatch):
    created = []
    updated = []
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: types.SimpleNamespace(kernel=object())
    display_module = types.ModuleType("IPython.display")
    display_module.HTML = lambda html: html

    def display(content, **kwargs):
        created.append(content)
        return types.SimpleNamespace(update=updated.append)

    display_module.display = display
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    handles = {}
    notebook.show_event(dict(id="same", html="first", text="first"), handles)
    notebook.show_event(dict(id="same", html="second", text="second"), handles)
    assert created == ["first"]
    assert updated == ["second"]


def test_child_stdout_stderr_and_unicode_reach_kernel(tmp_path, capsys):
    result = notebook.run_live(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; print('Trang 日本語'); print('lỗi mẫu', file=sys.stderr); sys.exit(7)",
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 7
    output = capsys.readouterr().out
    assert "Trang 日本語" in output
    assert "lỗi mẫu" in output


def test_event_visible_before_child_finishes(tmp_path, monkeypatch):
    events = []
    acknowledgment = tmp_path / "visible"

    def show(event, displays):
        events.append(event)
        acknowledgment.touch()

    monkeypatch.setattr(notebook, "show_event", show)
    code = """
import json, time, sys
from pathlib import Path
print('@@MANGA_COLAB_UI@@' + json.dumps(dict(id='progress', html='<b>ready</b>', text='ready')), flush=True)
deadline = time.monotonic() + 5
while not Path('visible').exists() and time.monotonic() < deadline:
    time.sleep(0.01)
sys.exit(0 if Path('visible').exists() else 8)
"""
    notebook.run_live([sys.executable, "-u", "-c", code], cwd=tmp_path, check=True)
    assert len(events) == 1


def test_setup_quiet_saves_full_log_and_exposes_failure(tmp_path, capsys):
    log = tmp_path / "setup.log"
    with pytest.raises(subprocess.CalledProcessError):
        notebook.run_live(
            [
                sys.executable,
                "-u",
                "-c",
                "print('@@MANGA_COLAB_STEP@@1/4 Kiểm tra'); print('dependency conflict'); exit(2)",
            ],
            cwd=tmp_path,
            log_path=log,
            quiet=True,
            check=True,
        )
    output = capsys.readouterr().out
    assert "1/4 Kiểm tra" in output
    assert "dependency conflict" in output
    assert "dependency conflict" in log.read_text(encoding="utf-8")


def test_html_escapes_filenames_and_keeps_progress_in_one_display(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    console = NotebookConsole()
    console.print(Text("<script>alert(1)</script> [red]日本語.png"))
    with AppProgress(console=console, title="Upscale") as progress:
        task = progress.add_task("Ảnh", total=2)
        progress.update(task, completed=2)
    events = [
        json.loads(line[len(notebook.EVENT_PREFIX) :]) for line in output.getvalue().splitlines()
    ]
    assert "<script>" not in events[0]["html"]
    assert "&lt;script&gt;" in events[0]["html"]
    assert "[red]日本語.png" in events[0]["text"]
    frames = [event for event in events if event["id"]]
    assert len(frames) >= 2
    assert len({event["id"] for event in frames}) == 1
    assert "100%" in frames[-1]["text"]
    assert "\x1b" not in output.getvalue()


def test_rate_counts_successes_only(monkeypatch):
    from manga_colab import logger

    now = [10.0]
    monkeypatch.setattr(logger.time, "monotonic", lambda: now[0])
    progress = BatchProgress(NotebookConsole(), 2, quiet=True)
    now[0] = 12.0
    progress.advance(success=False)
    assert "0.000 ảnh/s" in progress.progress.tasks[0].fields["detail"]
    now[0] = 14.0
    progress.advance(success=True)
    assert "0.250 ảnh/s" in progress.progress.tasks[0].fields["detail"]


@pytest.mark.skipif(os.name != "posix", reason="Colab uses POSIX process-group SIGINT")
def test_stop_forwards_sigint_and_drains_checkpoint_message(tmp_path, monkeypatch, capsys):
    def stop_once(event, displays):
        raise KeyboardInterrupt

    monkeypatch.setattr(notebook, "show_event", stop_once)
    code = """
import json, signal, time
def stop(signum, frame):
    print('checkpoint saved', flush=True)
    raise SystemExit(130)
signal.signal(signal.SIGINT, stop)
print('@@MANGA_COLAB_UI@@' + json.dumps(dict(id=None, html='', text='ready')), flush=True)
time.sleep(20)
"""
    result = notebook.run_live([sys.executable, "-u", "-c", code], cwd=tmp_path)
    assert result.returncode == 130
    assert "checkpoint saved" in capsys.readouterr().out


@pytest.mark.skipif(os.name != "posix", reason="Colab uses POSIX process-group signals")
def test_second_stop_kills_the_process_group(tmp_path, monkeypatch):
    import signal

    signals = []
    killpg = os.killpg

    def track_signal(pid, sig):
        signals.append(sig)
        killpg(pid, sig)

    def stop(event, displays):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "killpg", track_signal)
    monkeypatch.setattr(notebook, "show_event", stop)
    code = """
import json, signal, time
def event(*args):
    print('@@MANGA_COLAB_UI@@' + json.dumps(dict(id=None, html='', text='ready')), flush=True)
signal.signal(signal.SIGINT, event)
event()
time.sleep(20)
"""
    result = notebook.run_live([sys.executable, "-u", "-c", code], cwd=tmp_path)
    assert signals == [signal.SIGINT, signal.SIGKILL]
    assert result.returncode == -signal.SIGKILL
