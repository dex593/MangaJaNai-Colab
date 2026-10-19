"""Test transfer failures with injected bytes, not model inference."""

import hashlib

import gdown
import pytest
import requests
from gdown.exceptions import FileURLRetrievalError

from manga_colab.model_manager import DownloadError, ModelManager


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    body = b"fixture weights" * 40
    entry = dict(
        label="test weights",
        filename="test.pth",
        size=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
        source="https://example.test/model",
        download_url="https://example.test/model.pth",
    )
    manager = ModelManager(tmp_path, {"models": {"test.pth": entry}})
    monkeypatch.setattr("manga_colab.model_manager.time.sleep", lambda _: None)
    return manager, entry, body


@pytest.mark.parametrize("drive", [False, True])
def test_extra_download_progress_checksum_atomic_publish_and_cache(transfer, monkeypatch, drive):
    manager, entry, body = transfer
    if drive:
        entry["google_drive_id"] = "public-fixture-id"
    calls, updates = [], []

    def download(**kw):
        calls.append(kw)
        assert not (manager.directory / "test.pth").exists()
        assert kw["quiet"] is True and kw["use_cookies"] is False
        assert kw["timeout"] == (15, 60)
        for block in (body[:20], body[20:]):
            kw["output"].write(block)
            kw["hasher"].update(block)
            kw["progress"](kw["output"].tell(), len(body))
        return kw["output"]

    monkeypatch.setattr(gdown, "download", download)
    manager.ensure(["test.pth"], lambda name, size: lambda done, total: updates.append(done))
    assert (manager.directory / "test.pth").read_bytes() == body
    assert updates == [0, 20, len(body)]
    assert not (manager.directory / "test.pth.part").exists()
    assert calls[0]["id"] == ("public-fixture-id" if drive else None)
    assert calls[0]["url"] == (None if drive else entry["download_url"])
    manager.ensure(["test.pth"])
    assert len(calls) == 1


@pytest.mark.parametrize(
    "failure", ["network", "quota", "short", "checksum", "oversize", "interrupt"]
)
def test_extra_download_failure_preserves_previous_file_and_cleans_part(
    transfer, monkeypatch, failure
):
    manager, entry, body = transfer
    old = b"previous invalid file"
    (manager.directory / "test.pth").write_bytes(old)
    calls = []

    def download(**kw):
        calls.append(kw)
        data = b"x" * len(body) if failure == "checksum" else body[:20]
        kw["output"].write(data)
        kw["hasher"].update(data)
        if failure == "network":
            raise requests.ConnectionError("interrupted")
        if failure == "quota":
            raise FileURLRetrievalError("Google Drive quota exceeded")
        if failure == "oversize":
            kw["progress"](entry["size"] + 1, None)
        if failure == "interrupt":
            raise KeyboardInterrupt
        return kw["output"]

    monkeypatch.setattr(gdown, "download", download)
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else DownloadError):
        manager.ensure(["test.pth"])
    assert (manager.directory / "test.pth").read_bytes() == old
    assert not (manager.directory / "test.pth.part").exists()
    assert len(calls) == (1 if failure == "interrupt" else 3)


def test_interrupted_extra_download_restarts_and_verifies(transfer, monkeypatch):
    manager, entry, body = transfer
    calls = []

    def download(**kw):
        calls.append(kw)
        assert kw["output"].tell() == 0
        if len(calls) == 1:
            kw["output"].write(body[:10])
            raise requests.ConnectionError("interrupted")
        kw["output"].write(body)
        kw["hasher"].update(body)
        kw["progress"](len(body), len(body))

    monkeypatch.setattr(gdown, "download", download)
    manager.ensure(["test.pth"])
    assert manager.valid("test.pth")
    assert len(calls) == 2
