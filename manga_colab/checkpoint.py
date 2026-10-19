from __future__ import annotations

import json
import os
from pathlib import Path

from .storage import sha256, verify_image


class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, dict] = {}
        if path.exists():
            with path.open("rb") as stream:
                lines = stream.readlines()
            valid_bytes = 0
            truncated = False
            for index, line in enumerate(lines):
                try:
                    record = json.loads(line)
                    self.records[record["source"]] = record
                except (ValueError, KeyError, TypeError) as exc:
                    if index != len(lines) - 1:
                        raise ValueError(f"Manifest hỏng ở dòng {index + 1}: {path}") from exc
                    # Only a torn last append is recoverable; preserve all complete records.
                    with path.open("r+b") as stream:
                        stream.truncate(valid_bytes)
                    truncated = True
                    break
                valid_bytes += len(line)
            if lines and not truncated and not lines[-1].endswith(b"\n"):
                with path.open("ab") as stream:
                    stream.write(b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records[record["source"]] = record

    def completed(self, source: str, source_hash: str, fingerprint: str, output: Path) -> bool:
        record = self.records.get(source, {})
        if (
            record.get("status") not in ("success", "prepared")
            or record.get("source_hash") != source_hash
            or record.get("fingerprint") != fingerprint
            or record.get("output") != str(output)
        ):
            return False
        try:
            verify_image(output, tuple(record["dimensions"]))
            return sha256(output) == record["output_hash"]
        except OSError:
            raise
        except Exception:
            return False
