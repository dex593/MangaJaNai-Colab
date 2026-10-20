import json

from manga_colab.cli import main
from manga_colab.logger import error_message, user_message
from manga_colab.notebook import EVENT_PREFIX


def test_model_loader_root_cause_visible():
    try:
        try:
            raise RuntimeError("CUDA out of memory")
        except RuntimeError as cause:
            raise ValueError("Model unsupported") from cause
    except ValueError as exc:
        assert "CUDA out of memory" in error_message(exc)


def test_unreadable_image_has_plain_language_message():
    assert "Không đọc được ảnh" in user_message(
        "Error: unable to load from file b'017.jpg': VipsForeignLoad: not a known file format"
    )
    assert user_message("Error: tệp bị hỏng") == "tệp bị hỏng"


def test_cli_error_uses_card_with_original_details(monkeypatch, capsys):
    monkeypatch.setenv("MANGA_COLAB_NOTEBOOK", "1")
    assert main(["run"]) == 2
    event = json.loads(capsys.readouterr().out.removeprefix(EVENT_PREFIX))
    assert "Chưa thể tiếp tục" in event["html"]
    assert "Cần --input" in event["html"]
    assert "<details " in event["html"]
    assert "ValueError" in event["text"]
