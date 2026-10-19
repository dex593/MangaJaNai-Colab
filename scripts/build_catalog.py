"""Maintainer-only: verify official release archives and record member SHA-256.

End users use HTTP Range downloads through ModelManager, never this full download.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manga_colab.model_selector import DATA, load_workflow
from manga_colab.storage import atomic_json, sha256


def main() -> None:
    directory = Path(".runtime/catalog")
    directory.mkdir(parents=True, exist_ok=True)
    names = {c["ModelFilePath"] for c in load_workflow()["Chains"]["$values"]}
    result: dict = {"archives": {}, "models": {}}
    for tag, name in (
        ("1.0.0", "MangaJaNai_V1_ModelsOnly.zip"),
        ("3.0.0", "IllustrationJaNai_V3denoise.zip"),
    ):
        response = requests.get(
            f"https://api.github.com/repos/the-database/MangaJaNai/releases/tags/{tag}", timeout=60
        )
        response.raise_for_status()
        asset = next(a for a in response.json()["assets"] if a["name"] == name)
        path = directory / name
        if not path.exists():
            with requests.get(
                asset["browser_download_url"], stream=True, timeout=(15, 60)
            ) as download:
                download.raise_for_status()
                with path.with_suffix(".part").open("wb") as stream:
                    for block in download.iter_content(4 * 1024 * 1024):
                        stream.write(block)
            path.with_suffix(".part").replace(path)
        assert path.stat().st_size == asset["size"], "Archive size mismatch"
        digest = sha256(path)
        if asset.get("digest"):
            assert asset["digest"] == "sha256:" + digest, "Official archive checksum mismatch"
        result["archives"][name] = {
            k: asset[k] for k in ("id", "name", "size", "digest", "browser_download_url")
        }
        result["archives"][name].update(tag=tag, verified_sha256=digest)
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                filename = Path(info.filename).name
                if filename not in names:
                    continue
                model_hash = hashlib.sha256()
                with archive.open(info) as member:
                    for block in iter(lambda: member.read(4 * 1024 * 1024), b""):
                        model_hash.update(block)
                assert filename not in result["models"], "Duplicate model"
                result["models"][filename] = dict(
                    archive=name,
                    member=info.filename,
                    size=info.file_size,
                    compressed_size=info.compress_size,
                    crc32=info.CRC,
                    sha256=model_hash.hexdigest(),
                )
        print(f"Verified {name}: {digest}", flush=True)
    assert set(result["models"]) == names, "Missing model filenames"
    atomic_json(DATA / "models.json", result)
    print(f"Catalog: {len(names)} verified models")


if __name__ == "__main__":
    main()
