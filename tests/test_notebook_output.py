import io
import json
import os
import subprocess
import sys
import types

import pytest
from rich.text import Text

from manga_colab import notebook
from manga_colab.display import AppProgress, Card, NotebookConsole
from manga_colab.logger import BatchProgress, configure, scan_progress, user_message


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
    assert "0.0 ảnh/phút" in progress.progress.tasks[0].fields["detail"]
    assert progress.progress.tasks[1].completed == 0
    assert "Chưa hoàn tất" in progress.progress.html()
    now[0] = 14.0
    progress.advance(success=True)
    assert "15.0 ảnh/phút" in progress.progress.tasks[0].fields["detail"]


def test_scan_summary_has_count_without_last_filename(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    with scan_progress(NotebookConsole()) as progress:
        task = progress.add_task("Kiểm tra ảnh", total=3, scan=True)
        progress.update(task, completed=3, detail="017.jpg")
    events = [
        json.loads(line[len(notebook.EVENT_PREFIX) :]) for line in output.getvalue().splitlines()
    ]
    assert "3/3" in events[-1]["text"]
    assert "ảnh" in events[-1]["text"]
    assert ".jpg" not in events[-1]["text"]
    assert ".jpg" not in events[-1]["html"]


def test_card_escapes_all_user_text_and_collapses_technical_details():
    filename = '<script>alert("日本語")</script>'
    html = Card(
        filename,
        rows=[(filename, filename)],
        metrics=[(filename, filename)],
        details=[("Phiên bản", filename)],
        note=filename,
    ).html()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "日本語" in html
    assert "<pre" not in html
    assert "<details " in html and "<details open" not in html


def test_batch_saving_does_not_report_success_before_verification():
    progress = BatchProgress(NotebookConsole(), 1, quiet=True)
    progress.image(1, "Chương 1/日本語.png", 900, 1350, True, "model.pth", scale=4)
    assert "900 × 1350 → 3600 × 5400 px" in progress.progress.html()
    progress.update(3, 4, "Đang Upscale bằng model")
    assert 'aria-valuenow="75.00"' in progress.progress.html()
    progress.update(4, 4, "Đang Upscale bằng model")
    progress.update(0, 1, "Hoàn thiện và lưu ảnh")
    progress.update(1, 1, "Đã lưu tạm · đang kiểm tra")
    html = progress.progress.html()
    assert "Đang lưu và kiểm tra ảnh kết quả" in html
    assert "Đã lưu và kiểm tra kết quả" not in html
    assert progress.progress.tasks[0].completed == 0
    progress.advance()
    assert "Đã lưu và kiểm tra kết quả" in progress.progress.html()
    assert progress.progress.tasks[0].completed == 1


def test_download_html_uses_transferred_bytes_and_measured_speed():
    now = [0.0]
    progress = AppProgress(
        console=NotebookConsole(),
        title="Tải model",
        get_time=lambda: now[0],
    )
    task = progress.add_task("Model", total=4096, download=True, eta=True)
    progress.update(task, completed=1024)
    now[0] = 2.0
    progress.update(task, completed=2048)
    html = progress.html()
    assert 'aria-valuenow="50.00"' in html
    assert "2.0 KiB / 4.0 KiB" in html
    assert "512.0 B/s" in html
    assert "Còn khoảng 00:00:04" in html


def test_normal_log_keeps_per_image_details_in_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MANGA_COLAB_NOTEBOOK", "1")
    path = tmp_path / "application.log"
    _, log = configure(path, "normal")
    try:
        log.info("Ảnh đã lưu: 日本語.png")
        assert not capsys.readouterr().out
        log.error("日本語.png: ValueError: tệp bị hỏng")
        event = json.loads(capsys.readouterr().out.removeprefix(notebook.EVENT_PREFIX))
        assert "tệp bị hỏng" in event["html"]
        assert "日本語.png" in event["html"]
        saved = path.read_text(encoding="utf-8")
        assert "Ảnh đã lưu: 日本語.png" in saved
        assert "ValueError: tệp bị hỏng" in saved
    finally:
        for handler in log.handlers:
            handler.close()
        log.handlers.clear()


def test_memory_error_message_distinguishes_gpu_from_unknown_memory():
    assert "Card đồ họa" in user_message("RuntimeError: CUDA out of memory")
    assert "Card đồ họa" not in user_message("RuntimeError: CPU allocator out of memory")


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
