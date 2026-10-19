import hashlib
import io
import zipfile

import pytest
import requests

from manga_colab.model_manager import DownloadError, ModelManager, RangeReader


class Response:
    def __init__(self, data, start, stop, status=206, truncate=False):
        self.data = data[start : stop - (1 if truncate else 0)]
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start}-{stop-1}/{len(data)}"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def iter_content(self, size):
        yield self.data


class Session:
    def __init__(self, data, failures=0, truncate=False, status=206):
        self.data, self.failures, self.truncate, self.status = data, failures, truncate, status
        self.calls = 0

    def get(self, url, headers, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise requests.ConnectionError("interrupted")
        start, last = map(int, headers["Range"][6:].split("-"))
        return Response(self.data, start, last + 1, self.status, self.truncate)


def fixture_zip():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("models/real-name.pth", b"weights fixture" * 1000)
        archive.writestr("../do-not-extract", b"not selected")
    return stream.getvalue()


def test_remote_zip_range_and_retry(monkeypatch):
    monkeypatch.setattr("manga_colab.model_manager.time.sleep", lambda _: None)
    data = fixture_zip()
    session = Session(data, failures=1)
    with RangeReader("https://example.test", len(data), session) as reader:
        with zipfile.ZipFile(reader) as archive:
            assert archive.read("models/real-name.pth") == b"weights fixture" * 1000
    assert session.calls >= 2


@pytest.mark.parametrize("kwargs", [{"failures": 9}, {"truncate": True}, {"status": 200}])
def test_failed_partial_or_ignored_range(monkeypatch, kwargs):
    monkeypatch.setattr("manga_colab.model_manager.time.sleep", lambda _: None)
    data = fixture_zip()
    with pytest.raises(DownloadError):
        RangeReader("https://example.test", len(data), Session(data, **kwargs)).read(32)


def test_cache_does_not_accept_partial_or_corrupt(tmp_path):
    name = "fixture.pth"
    body = b"complete model fixture"
    catalog = {"models": {name: {"size": len(body), "sha256": hashlib.sha256(body).hexdigest()}}}
    manager = ModelManager(tmp_path, catalog)
    (tmp_path / (name + ".part")).write_bytes(body)
    assert not manager.valid(name)
    (tmp_path / name).write_bytes(b"x" * len(body))
    assert not manager.valid(name)
    (tmp_path / name).write_bytes(body)
    assert manager.valid(name)
    with pytest.raises(DownloadError, match="official|chính thức"):
        manager.entry("unknown.pth")


def test_failed_ensure_never_publishes(tmp_path, monkeypatch):
    catalog = {
        "models": {
            "fixture.pth": {"size": 10, "sha256": "bad", "archive": "bundle", "member": "x"}
        },
        "archives": {"bundle": {"tag": "1", "name": "bundle"}},
    }
    monkeypatch.setattr(
        "manga_colab.model_manager.release_asset",
        lambda *args: (_ for _ in ()).throw(DownloadError("interrupted")),
    )
    with pytest.raises(DownloadError, match="fixture.pth"):
        ModelManager(tmp_path, catalog).ensure(["fixture.pth"])
    assert not (tmp_path / "fixture.pth").exists()
    assert not (tmp_path / "fixture.pth.part").exists()
