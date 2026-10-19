from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

from filelock import FileLock

from . import __version__
from .backend import Backend
from .checkpoint import Checkpoint
from .config import Config, ROOT, UPSTREAM_COMMIT
from .gpu import gpu_info
from .logger import BatchProgress, configure, dashboard, download_progress, error_message
from .model_manager import ModelManager
from .model_selector import load_workflow
from .report import display_report, write_report
from .scanner import build_plan
from .storage import atomic_json, probe_directory, require_space, sha256, verify_image


def run(config: Config, dry_run: bool = False) -> int:
    started = time.monotonic()
    probe_directory(config.input, write=False)
    probe_directory(config.output)
    state = config.output / ".manga-colab"
    state.mkdir(exist_ok=True)
    with FileLock(str(state / "run.lock"), timeout=1):
        return _run(config, state, started, dry_run)


def _run(config: Config, state: Path, started: float, dry_run: bool) -> int:
    console, log = configure(state / "application.log", config.log_level)
    workflow = load_workflow(config.workflow)
    log.info("Đang quét nội dung ảnh, ICC và SHA-256 input...")
    plan = build_plan(config, workflow)
    atomic_json(state / "plan.json", [item.as_json() for item in plan])
    if not plan:
        report = write_report(
            state, [], 0, started, config.as_json(), "Input không có ảnh được hỗ trợ"
        )
        display_report(console, report)
        return 2
    if dry_run:
        for item in plan:
            console.print(
                f"{item.source} → {item.output} | {item.model or item.error}", markup=False
            )
        console.print(f"Plan: {state / 'plan.json'}", markup=False)
        return 2 if any(item.error for item in plan) else 0
    gpu = gpu_info(config.cuda_device, required=True)
    probe_directory(config.models)
    if config.stage:
        probe_directory(config.stage)
    manager = ModelManager(config.models)
    manifest = Checkpoint(state / "manifest.jsonl")
    rows: list[dict] = []
    pending = []
    for item in plan:
        output = config.output / item.output
        fingerprint = (
            config.fingerprint(workflow, manager.entry(item.model)["sha256"]) if item.model else ""
        )
        row = dict(
            source=item.source,
            output=str(output),
            source_hash=item.source_hash,
            fingerprint=fingerprint,
            model=item.model,
            input_bytes=item.size,
        )
        old = manifest.records.get(item.source, {})
        if item.error:
            row.update(status="failed", error=item.error)
        elif (
            not config.overwrite
            and config.resume
            and manifest.completed(item.source, item.source_hash, fingerprint, output)
        ):
            row.update(
                status="skipped",
                reason="Đã xác minh manifest, cấu hình và checksum output",
                output_bytes=output.stat().st_size,
                dimensions=old["dimensions"],
            )
        elif output.exists() and not config.overwrite:
            row.update(
                status="failed",
                error="Output đã tồn tại nhưng không khớp manifest/cấu hình hoặc bị hỏng. "
                "Dùng output mới hoặc --overwrite để thay thế.",
            )
        elif (
            old.get("status") == "failed"
            and old.get("fingerprint") == fingerprint
            and old.get("source_hash") == item.source_hash
            and not config.retry_failed
            and not config.overwrite
        ):
            row.update(
                status="skipped",
                reason="Lần trước lỗi; dùng --retry-failed",
                error=old.get("error", ""),
            )
        else:
            pending.append((item, row))
            continue
        rows.append(row)
    needed = list(dict.fromkeys(item.model for item, _ in pending))
    missing = [name for name in needed if not manager.valid(name)]
    commit = (
        subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "uncommitted"
    )
    if config.log_level != "quiet":
        dashboard(
            console,
            {
                "Version / commit": f"{__version__} / {commit}",
                "Upstream": UPSTREAM_COMMIT[:12],
                "GPU": gpu["name"],
                "CUDA / PyTorch": f"{gpu['cuda']} / {gpu['torch']}",
                "VRAM free / total": f"{gpu['free']/2**30:.2f} / {gpu['total']/2**30:.2f} GiB",
                "Compute / FP16 / BF16": f"{gpu['compute']} / {gpu['fp16']} / {gpu['bf16']}",
                "Input": config.input,
                "Output": config.output,
                "Model cache": config.models,
                "Target": f"{config.scale}x / {config.format}",
                "Mode": f"AUTO / detector={config.detector}",
                "Images": len(plan),
                "Models to download": len(missing),
                "Previously complete": sum(
                    row["status"] == "skipped" and not row.get("error") for row in rows
                ),
            },
        )
    backend = None
    fatal = ""
    code = 0
    try:
        if pending:
            with download_progress(console, config.log_level == "quiet") as downloads:

                def factory(name: str, size: int):
                    task = downloads.add_task(name, total=size)

                    def update(done: int, _: int) -> None:
                        downloads.update(task, completed=done, total=max(done, size))

                    return update

                manager.ensure(needed, factory)
            backend = Backend(config, workflow, state / "backend.log")
        progress = BatchProgress(console, len(plan), config.log_level == "quiet")
        with progress.progress:
            for _ in rows:
                progress.advance()
            for index, (item, row) in enumerate(pending):
                image_start = time.monotonic()
                output = config.output / item.output
                source = config.input / item.source
                if not output.resolve().is_relative_to(
                    config.output
                ) or not source.resolve().is_relative_to(config.input):
                    raise OSError("Đường dẫn symlink thoát khỏi thư mục input/output đã chọn")
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.resolve().is_relative_to(
                    config.input
                ) and not config.output.is_relative_to(config.input):
                    raise OSError("Output symlink trỏ vào input; từ chối ghi")
                log.info(
                    "[%d/%d] %s | %dx%d | %s | %s | disk cache HIT",
                    index + 1,
                    len(pending),
                    item.source,
                    item.width,
                    item.height,
                    "GRAYSCALE" if item.grayscale else "COLOR",
                    item.model,
                )
                temp: Path | None = None
                try:
                    budget = int(
                        item.width * item.height * config.scale**2 * 8 + item.size + 32 * 2**20
                    )
                    require_space(config.output, budget)
                    if config.stage:
                        require_space(config.stage, budget)
                    if not config.input.is_dir():
                        raise OSError("Input/Google Drive đã bị ngắt mount")
                    if sha256(source) != item.source_hash:
                        raise ValueError("Input thay đổi sau khi scan; chạy lại để lập plan mới")
                    fd, temp_name = tempfile.mkstemp(
                        prefix=".manga-", suffix="." + config.format, dir=output.parent
                    )
                    os.close(fd)
                    temp = Path(temp_name)
                    assert backend is not None
                    if config.stage:
                        with tempfile.TemporaryDirectory(
                            prefix="manga-", dir=config.stage
                        ) as local:
                            local_source = Path(local) / ("input" + source.suffix)
                            local_output = Path(local) / ("output." + config.format)
                            shutil.copyfile(source, local_source)
                            row.update(
                                backend.process(
                                    local_source, local_output, item.chain, progress.update
                                )
                            )
                            verify_image(local_output)
                            shutil.copyfile(local_output, temp)
                    else:
                        row.update(backend.process(source, temp, item.chain, progress.update))
                    # Same rounding as upstream's final_target_resize, including fractional scale.
                    target_height = round(item.height * config.scale)
                    expected = (round(item.width * target_height / item.height), target_height)
                    row["dimensions"] = list(verify_image(temp, expected))
                    row["output_hash"] = sha256(temp)
                    row["output_bytes"] = temp.stat().st_size
                    with temp.open("r+b") as stream:
                        os.fsync(stream.fileno())
                    if output.exists() and not config.overwrite:
                        raise FileExistsError("Output xuất hiện trong khi xử lý; không ghi đè")
                    row.update(status="prepared", seconds=time.monotonic() - image_start)
                    manifest.append(row)
                    os.replace(temp, output)
                    # Drive FUSE can fail after writes: reopen the final path before success.
                    verify_image(output, expected)
                    if sha256(output) != row["output_hash"]:
                        raise OSError("Checksum output thay đổi sau khi ghi Drive")
                    row["status"] = "success"
                    manifest.append(row)
                    log.info(
                        "✓ %s → %s | %.2fs | %s",
                        item.source,
                        item.output,
                        row["seconds"],
                        row["dimensions"],
                    )
                except OSError as exc:
                    row.update(
                        status="failed",
                        error=error_message(exc),
                        seconds=time.monotonic() - image_start,
                    )
                    rows.append(row.copy())
                    raise
                except Exception as exc:
                    row.update(
                        status="failed",
                        error=error_message(exc),
                        seconds=time.monotonic() - image_start,
                    )
                    if backend and hasattr(backend, "log"):
                        traceback.print_exception(exc, file=backend.log)
                    if any(
                        message in str(exc).lower()
                        for message in (
                            "no space left",
                            "input/output error",
                            "read-only file system",
                            "permission denied",
                            "transport endpoint",
                        )
                    ):
                        rows.append(row.copy())
                        raise OSError(row["error"]) from exc
                    manifest.append(row)
                    log.error(
                        "%s: %s", item.source, row["error"], exc_info=config.log_level == "debug"
                    )
                finally:
                    if temp:
                        temp.unlink(missing_ok=True)
                rows.append(row.copy())
                progress.advance()
                write_report(state, rows, len(plan), started, config.as_json())
    except KeyboardInterrupt:
        fatal, code = "Đã dừng an toàn bằng Ctrl+C; chạy lại --resume", 130
        log.warning(fatal)
    except Exception as exc:
        fatal, code = error_message(exc), 2
        log.error(fatal, exc_info=config.log_level == "debug")
    finally:
        if backend:
            # Include retries from images that failed before process() could return metadata.
            reported = sum(row.get("oom_retries", 0) for row in rows)
            if rows and getattr(backend, "oom_retries", 0) > reported:
                rows[-1]["oom_retries"] = (
                    rows[-1].get("oom_retries", 0) + backend.oom_retries - reported
                )
            backend.close()
        report = write_report(state, rows, len(plan), started, config.as_json(), fatal)
        display_report(console, report)
    return code or (2 if any(row["status"] == "failed" or row.get("error") for row in rows) else 0)
