import json
import time

from manga_colab.display import NotebookConsole
from manga_colab.notebook import EVENT_PREFIX
from manga_colab.report import display_report, write_report


def test_previously_failed_image_is_not_displayed_as_completed(tmp_path, capsys):
    error = "RuntimeError: CUDA out of memory"
    report = write_report(
        tmp_path,
        [
            dict(source="001.png", status="skipped"),
            dict(source="017.png", status="skipped", error=error),
        ],
        total=2,
        started=time.monotonic(),
        config={"output": str(tmp_path)},
    )
    display_report(NotebookConsole(), report)
    events = [
        json.loads(line.removeprefix(EVENT_PREFIX)) for line in capsys.readouterr().out.splitlines()
    ]
    assert "Có ảnh cần kiểm tra" in events[0]["text"]
    assert "Đã làm từ trước" in events[0]["text"]
    assert "017.png" in events[1]["html"]
    assert "Card đồ họa không đủ bộ nhớ" in events[1]["html"]
    assert error in events[1]["html"]
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["images"][1]["error"] == error
    assert saved["summary"]["successful"] == 0
    # A previously failed skip must not inflate the count of completed images.
    cards = []
    display_report(type("Console", (), {"print": lambda _, card: cards.append(card)})(), report)
    metrics = dict(cards[0].metrics)
    assert metrics["Đã làm từ trước"] == 1
    assert metrics["Cần kiểm tra"] == 1


def test_stopped_report_keeps_reason_and_pending_count(tmp_path):
    report = write_report(
        tmp_path,
        [],
        total=3,
        started=time.monotonic(),
        config={"output": str(tmp_path)},
        fatal="OSError: No space left on device",
    )
    cards = []
    display_report(type("Console", (), {"print": lambda _, card: cards.append(card)})(), report)
    assert "Đã dừng" in cards[0].title
    assert dict(cards[0].metrics)["Chưa xử lý"] == 3
    assert "hết dung lượng" in dict(cards[0].rows)["Lý do dừng"]
