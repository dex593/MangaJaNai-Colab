import ast
import json
import runpy
import subprocess
from pathlib import Path

import pytest


def test_launcher_syntax_and_no_credentials_or_engine():
    path = Path(__file__).parents[1] / "colab/Run_MangaJaNai.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 6
    for source in code:
        ast.parse(source)
    combined = "\n".join(code)
    for forbidden in (
        "userdata",
        "GITHUB_TOKEN",
        "load_model",
        "import torch",
        "shell=True",
        "safetensors",
    ):
        assert forbidden not in combined
    assert "--ff-only" in combined
    assert "scripts/setup_colab.sh" in combined


def test_run_cell_streams_and_rebuilds_arguments(monkeypatch):
    notebook = json.loads(
        (Path(__file__).parents[1] / "colab/Run_MangaJaNai.ipynb").read_text(encoding="utf-8")
    )
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    calls = []
    refreshed = []

    def live(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(
        runpy, "run_path", lambda path: {"run_live": live, "refresh_drive": refreshed.append}
    )
    scope = {"Path": Path}
    exec(code[0], scope)
    exec(code[4], scope)
    scope["INPUT_DIR"] = "/new/input 日本語"
    exec(code[4], scope)
    assert calls[0][calls[0].index("--input") + 1] != calls[1][calls[1].index("--input") + 1]
    assert "--resume" in calls[1]
    assert "--overwrite" not in calls[1]
    assert refreshed == [calls[0][calls[0].index("--input") + 1], "/new/input 日本語"]


def test_run_cell_stops_if_drive_cannot_refresh(monkeypatch):
    notebook = json.loads(
        (Path(__file__).parents[1] / "colab/Run_MangaJaNai.ipynb").read_text(encoding="utf-8")
    )

    def fail(path):
        raise RuntimeError("Drive chưa lưu xong")

    def forbidden(*args, **kwargs):
        pytest.fail("Must not start scanning or AI after a Drive refresh error")

    monkeypatch.setattr(
        runpy, "run_path", lambda path: {"run_live": forbidden, "refresh_drive": fail}
    )
    scope = {"Path": Path}
    exec("".join(notebook["cells"][1]["source"]), scope)
    with pytest.raises(RuntimeError, match="Drive chưa lưu xong"):
        exec("".join(notebook["cells"][5]["source"]), scope)


def test_processing_choice_switches_cli_flags_without_conflicting_modes(monkeypatch):
    notebook = json.loads(
        (Path(__file__).parents[1] / "colab/Run_MangaJaNai.ipynb").read_text(encoding="utf-8")
    )
    form = "".join(notebook["cells"][1]["source"])
    run = "".join(notebook["cells"][5]["source"])
    calls = []

    def live(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(
        runpy, "run_path", lambda path: {"run_live": live, "refresh_drive": lambda path: None}
    )
    scope = {"Path": Path}
    for mode, expected in (
        ("Chỉ Upscale ảnh chưa làm", (True, False)),
        ("Upscale lại tất cả ảnh", (False, True)),
        ("Chỉ Upscale ảnh chưa làm", (True, False)),
    ):
        source = form.replace('Chạy = "Chỉ Upscale ảnh chưa làm"', f'Chạy = "{mode}"')
        exec(source, scope)
        exec(run, scope)
        assert (scope["RESUME"], scope["OVERWRITE"]) == expected
        assert ("--resume" in calls[-1], "--overwrite" in calls[-1]) == expected
        assert ("--no-resume" in calls[-1]) is expected[1]
    invalid = form.replace('Chạy = "Chỉ Upscale ảnh chưa làm"', 'Chạy = "Không hợp lệ"')
    with pytest.raises(KeyError):
        exec(invalid, scope)


def test_every_form_field_has_help_and_internal_names_are_hidden():
    notebook = json.loads(
        (Path(__file__).parents[1] / "colab/Run_MangaJaNai.ipynb").read_text(encoding="utf-8")
    )
    form = "".join(notebook["cells"][1]["source"])
    assert "phóng to" not in form.casefold()
    for key in ("INPUT_DIR", "OUTPUT_DIR", "RESUME", "OVERWRITE", "PIN_COMMIT"):
        assert not any(key in line and "# @param" in line for line in form.splitlines())
    assert "không nhập link chia sẻ" in form
    fields = [line.split("=", 1)[0].strip() for line in form.splitlines() if "# @param" in line]
    assert len(fields) == 10
    assert all(field.isidentifier() and "_" not in field for field in fields)
    assert "Chạy" in fields
    assert "Bỏ_qua_ảnh_đã_làm_xong" not in form
    assert "Làm_lại_ảnh_cũ" not in form
    for removed in ("Địa chỉ chương trình", "Nhánh chương trình", "Mã phiên bản", "Bo_qua_"):
        assert removed.casefold() not in form.casefold()
    scope = {}
    exec(form, scope)
    assert scope["GITHUB_REPO"] == "https://github.com/dex593/MangaJaNai-Colab.git"
    assert scope["BRANCH"] == "main"
    assert scope["PIN_COMMIT"] == ""
    assert scope["RESUME"] is True and scope["OVERWRITE"] is False


@pytest.mark.parametrize(
    "choice,model,folder",
    [
        ("MangaJaNai (tự chọn)", "auto", "Output"),
        ("Eula Digimanga BW V2 NC1", "eula-digimanga", "Eula Digimanga"),
        ("2x Manga Ora", "manga-ora", "Manga Ora"),
        ("MangaScale GFISRV2 4x", "mangascale-gfisrv2", "MangaScale GFISRV2"),
    ],
)
def test_model_choice_reaches_cli_and_uses_separate_output(monkeypatch, choice, model, folder):
    notebook = json.loads(
        (Path(__file__).parents[1] / "colab/Run_MangaJaNai.ipynb").read_text(encoding="utf-8")
    )
    calls = []

    def live(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(
        runpy, "run_path", lambda path: {"run_live": live, "refresh_drive": lambda path: None}
    )
    scope = {}
    form = "".join(notebook["cells"][1]["source"])
    exec(form.replace('Bộ = "MangaJaNai (tự chọn)"', f'Bộ = "{choice}"'), scope)
    exec("".join(notebook["cells"][5]["source"]), scope)
    assert calls[0][calls[0].index("--model") + 1] == model
    assert Path(calls[0][calls[0].index("--output") + 1]).name == folder
    assert "--auto-model" not in calls[0]
    exec("".join(notebook["cells"][6]["source"]), scope)
    assert calls[1][calls[1].index("--output") + 1] == scope["OUTPUT_DIR"]
