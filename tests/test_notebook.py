import ast
import json
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
