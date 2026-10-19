"""Drive API test doubles; these do not verify a live Colab mount."""

import sys
import types

import pytest

from manga_colab import notebook


@pytest.fixture
def drive_calls(monkeypatch):
    calls = []
    drive = types.SimpleNamespace(
        flush_and_unmount=lambda **kw: calls.append(("flush", kw)),
        mount=lambda *args, **kw: calls.append(("mount", args, kw)),
    )
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = drive
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(notebook.Path, "is_dir", lambda path: True)
    return drive, calls


def test_every_run_flushes_before_remounting(drive_calls):
    _, calls = drive_calls
    for _ in range(2):
        notebook.refresh_drive("/content/drive/MyDrive/Ảnh 日本語")
    assert calls == [
        ("flush", {"timeout_ms": 60_000}),
        ("mount", ("/content/drive",), {"force_remount": True}),
    ] * 2


@pytest.mark.parametrize("step", ["flush_and_unmount", "mount"])
def test_failed_drive_operation_stops_with_help(step, drive_calls, monkeypatch):
    drive, calls = drive_calls
    error = ValueError("injected Drive failure")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(drive, step, fail)
    with pytest.raises(RuntimeError, match="Bước 5") as caught:
        notebook.refresh_drive("/content/drive/MyDrive/Manga/Input")
    assert caught.value.__cause__ is error
    if step == "flush_and_unmount":
        assert not calls, "Never force remount after a failed flush"
    else:
        assert calls == [("flush", {"timeout_ms": 60_000})]


def test_missing_folder_after_remount_has_actionable_error(drive_calls, monkeypatch):
    monkeypatch.setattr(notebook.Path, "is_dir", lambda path: False)
    with pytest.raises(FileNotFoundError, match="tài khoản Drive.*Bước 1"):
        notebook.refresh_drive("/content/drive/MyDrive/Missing")


@pytest.mark.parametrize("path", ["/content/images", "/content/drive-copy/images", "/content/drive/../images"])
def test_local_input_does_not_touch_drive(path, drive_calls):
    _, calls = drive_calls
    notebook.refresh_drive(path)
    assert not calls
