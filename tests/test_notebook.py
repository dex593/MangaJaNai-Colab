import ast
import json
import runpy
import subprocess
from pathlib import Path


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

    def live(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(runpy, "run_path", lambda path: {"run_live": live})
    scope = {"Path": Path}
    exec(code[0], scope)
    exec(code[4], scope)
    scope["INPUT_DIR"] = "/new/input 日本語"
    exec(code[4], scope)
    assert calls[0][calls[0].index("--input") + 1] != calls[1][calls[1].index("--input") + 1]
    assert "--resume" in calls[1]
    assert "--overwrite" not in calls[1]


def test_every_form_field_has_help_and_internal_names_are_hidden():
    notebook = json.loads(
        (Path(__file__).parents[1] / "colab/Run_MangaJaNai.ipynb").read_text(encoding="utf-8")
    )
    form = "".join(notebook["cells"][1]["source"])
    assert "phóng to" not in form.casefold()
    for key in ("INPUT_DIR", "OUTPUT_DIR", "RESUME", "OVERWRITE", "PIN_COMMIT"):
        assert not any(key in line and "# @param" in line for line in form.splitlines())
    assert "không nhập link chia sẻ" in form
