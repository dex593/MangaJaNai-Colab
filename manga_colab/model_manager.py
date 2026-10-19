from __future__ import annotations

import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Callable

import requests
from filelock import FileLock

from .model_selector import DATA
from .storage import atomic_json, require_space, sha256

OFFICIAL = "https://github.com/the-database/MangaJaNai/releases"
Progress = Callable[[int, int], None]


class DownloadError(RuntimeError):
    pass


class RangeReader(io.RawIOBase):
    """Seekable HTTP file; ZIP reads only its directory and the requested member."""

    def __init__(
        self,
        url: str,
        size: int,
        session: requests.Session | None = None,
        progress: Progress | None = None,
        retries: int = 3,
    ):
        self.url, self.size, self.position = url, size, 0
        self.session = session or requests.Session()
        self.owns_session = session is None
        self.progress, self.retries = progress, retries
        self.transferred = 0
        self.buffer_start, self.buffer = 0, b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        target = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if target < 0:
            raise ValueError("Negative seek")
        self.position = target
        return target

    def read(self, size: int = -1) -> bytes:
        size = min(self.size - self.position, size if size >= 0 else self.size)
        if size <= 0:
            return b""
        start, end = self.position, self.position + size
        if not self.buffer_start <= start < end <= self.buffer_start + len(self.buffer):
            stop = min(self.size, max(end, start + 1024 * 1024))
            last: Exception | None = None
            for attempt in range(self.retries):
                try:
                    with self.session.get(
                        self.url,
                        headers={
                            "Range": f"bytes={start}-{stop - 1}",
                            "Accept-Encoding": "identity",
                        },
                        timeout=(15, 60),
                        stream=True,
                    ) as response:
                        if response.status_code != 206:
                            raise DownloadError(
                                f"HTTP Range cần 206, nhận {response.status_code}; nguồn: {OFFICIAL}"
                            )
                        expected = f"bytes {start}-{stop - 1}/{self.size}"
                        if response.headers.get("Content-Range") != expected:
                            raise DownloadError(
                                "Content-Range không khớp; từ chối archive thay đổi/chưa đầy đủ"
                            )
                        data = bytearray()
                        for block in response.iter_content(256 * 1024):
                            data.extend(block)
                            self.transferred += len(block)
                            if self.progress:
                                self.progress(self.transferred, self.size)
                            if len(data) > stop - start:
                                raise DownloadError("Server trả dữ liệu vượt phạm vi yêu cầu")
                        if len(data) != stop - start:
                            raise DownloadError("Download bị gián đoạn: số bytes không khớp")
                        self.buffer_start, self.buffer = start, bytes(data)
                        break
                except (requests.RequestException, DownloadError) as exc:
                    last = exc
                    if attempt + 1 == self.retries:
                        raise DownloadError(f"Tải thất bại sau {self.retries} lần: {last}") from exc
                    time.sleep(2**attempt)
        result = self.buffer[start - self.buffer_start : end - self.buffer_start]
        self.position = end
        return result

    def close(self) -> None:
        if self.owns_session:
            self.session.close()
        super().close()


def release_asset(tag: str, asset_name: str, expected: dict, session: requests.Session) -> dict:
    url = f"https://api.github.com/repos/the-database/MangaJaNai/releases/tags/{tag}"
    for attempt in range(3):
        try:
            response = session.get(url, timeout=(15, 45))
            if response.status_code in (403, 429):
                raise DownloadError(
                    f"GitHub API rate limit/quyền truy cập ({response.status_code}). "
                    f"Reset: {response.headers.get('X-RateLimit-Reset', 'unknown')}. "
                    "Thử lại sau hoặc tải thủ công từ " + OFFICIAL
                )
            response.raise_for_status()
            assets = response.json()["assets"]
            asset = next((a for a in assets if a["name"] == asset_name), None)
            if asset is None:
                raise DownloadError(f"Asset không còn tồn tại: {asset_name}; {url}")
            if any(asset.get(k) != expected.get(k) for k in ("id", "size", "digest")):
                raise DownloadError(
                    f"Asset {asset_name} đã thay đổi; cần cập nhật catalog đã kiểm chứng"
                )
            if not asset["browser_download_url"].startswith(OFFICIAL + "/download/"):
                raise DownloadError("GitHub API trả URL ngoài nguồn model chính thức")
            return asset
        except requests.RequestException as exc:
            if attempt == 2:
                raise DownloadError(
                    f"Không đọc được GitHub Releases API: {exc}; {OFFICIAL}"
                ) from exc
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


class ModelManager:
    def __init__(self, directory: Path, catalog: dict | None = None):
        self.directory = directory
        self.catalog = catalog or json.loads((DATA / "models.json").read_text(encoding="utf-8"))
        self.validated: dict[str, str] = {}

    def entry(self, name: str) -> dict:
        if name not in self.catalog["models"]:
            raise DownloadError(f"Model không có trong catalog chính thức: {name}; {OFFICIAL}")
        return self.catalog["models"][name]

    def valid(self, name: str) -> bool:
        entry = self.entry(name)
        path = self.directory / name
        if not path.is_file() or path.stat().st_size != entry["size"]:
            return False
        digest = sha256(path)
        if digest == entry["sha256"]:
            self.validated[name] = digest
            return True
        return False

    def ensure(self, names: list[str], progress_factory=None) -> dict[str, str]:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (
            FileLock(str(self.directory / ".download.lock"), timeout=1),
            requests.Session() as session,
        ):
            assets: dict[str, dict] = {}
            for name in dict.fromkeys(names):
                if self.valid(name):
                    continue
                entry = self.entry(name)
                archive = self.catalog["archives"][entry["archive"]]
                manual = (
                    f"Model: {name}. Nguồn: {OFFICIAL}/tag/{archive['tag']}. "
                    f"Giải nén {entry['member']} từ {archive['name']} vào {self.directory}."
                )
                temp = self.directory / (name + ".part")
                try:
                    require_space(self.directory, entry["size"] + 32 * 1024 * 1024)
                    asset = assets.get(entry["archive"])
                    if asset is None:
                        asset = release_asset(archive["tag"], archive["name"], archive, session)
                        assets[entry["archive"]] = asset
                    progress = (
                        progress_factory(name, entry["compressed_size"])
                        if progress_factory
                        else None
                    )
                    with RangeReader(
                        asset["browser_download_url"], asset["size"], session, progress
                    ) as remote:
                        with zipfile.ZipFile(remote) as zipped:
                            info = zipped.getinfo(entry["member"])
                            if info.file_size != entry["size"] or info.CRC != entry["crc32"]:
                                raise DownloadError("ZIP member không khớp catalog")
                            digest = hashlib.sha256()
                            written = 0
                            with zipped.open(info) as source, temp.open("wb") as output:
                                for block in iter(lambda: source.read(1024 * 1024), b""):
                                    written += len(block)
                                    if written > entry["size"]:
                                        raise DownloadError("ZIP member vượt kích thước dự kiến")
                                    output.write(block)
                                    digest.update(block)
                                output.flush()
                                os.fsync(output.fileno())
                            if written != entry["size"] or digest.hexdigest() != entry["sha256"]:
                                raise DownloadError("Model bị hỏng: size/SHA-256 không khớp")
                    os.replace(temp, self.directory / name)
                    self.validated[name] = digest.hexdigest()
                    atomic_json(
                        self.directory / (name + ".json"),
                        {
                            "sha256": digest.hexdigest(),
                            "source": asset["browser_download_url"],
                            "member": entry["member"],
                        },
                    )
                except (OSError, ValueError, KeyError, zipfile.BadZipFile, DownloadError) as exc:
                    raise DownloadError(f"{exc}\n{manual}") from exc
                finally:
                    temp.unlink(missing_ok=True)
        return {name: self.validated[name] for name in names}
