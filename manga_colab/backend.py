from __future__ import annotations

import contextlib
import copy
import importlib
import importlib.util
import json
import logging
import math
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from queue import Queue
from typing import Callable

import numpy as np

from .config import Config, UPSTREAM_COMMIT
from .detector import detect
from .gpu import gpu_info, initial_tile_size, map_device, use_reduced_precision


def make_settings(config: Config, workflow: dict, gui_device: int, fp16: bool) -> dict:
    workflow = copy.deepcopy(workflow)
    workflow.update(
        SelectedTabIndex=0,
        InputFilePath=str(config.input),
        InputFolderPath=str(config.input),
        OutputFolderPath=str(config.output),
        OutputFilename="%filename%",
        UpscaleImages=True,
        UpscaleArchives=False,
        OverwriteExistingFiles=config.overwrite,
        ModeScaleSelected=True,
        ModeWidthSelected=False,
        ModeHeightSelected=False,
        ModeFitToDisplaySelected=False,
        UpscaleScaleFactor=config.scale,
        GrayscaleDetectionThreshold=config.threshold,
        LossyCompressionQuality=config.quality,
        UseLosslessCompression=config.lossless,
    )
    for key, value in (
        ("PngSelected", "png"),
        ("JpegSelected", "jpeg"),
        ("WebpSelected", "webp"),
        ("AvifSelected", "avif"),
    ):
        workflow[key] = config.format == value
    return dict(
        SelectedWorkflowIndex=0,
        SelectedDeviceIndex=gui_device,
        UseCpu=False,
        UseFp16=fp16,
        ModelsDirectory=str(config.models),
        Workflows={"$values": [workflow]},
    )


class TileOOM(RuntimeError):
    def __init__(self, tile: int):
        super().__init__(f"CUDA OOM; retry tile={tile}")
        self.tile = tile


class Backend:
    def __init__(self, config: Config, workflow: dict, log_path: Path):
        self.config, self.workflow = config, workflow
        self.info = gpu_info(config.cuda_device, required=True)
        self.cache: OrderedDict[str, object] = OrderedDict()
        self.tile_limits: dict[str, int] = {}
        self.oom_retries = 0
        self.callback: Callable[[int, int, str], None] = lambda done, total, stage: None
        self.log = log_path.open("a", encoding="utf-8", buffering=1)
        actual = subprocess.check_output(
            ["git", "-C", str(config.backend), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != UPSTREAM_COMMIT:
            raise RuntimeError(f"Sai upstream commit: {actual}; chạy scripts/setup_colab.sh")
        source = config.backend / "MangaJaNaiConverterGui/backend/src"
        sys.path.insert(0, str(source))
        with contextlib.redirect_stdout(self.log), contextlib.redirect_stderr(self.log):
            from accelerator_detection import get_accelerator_detector

            devices = [
                {"type": d.type.value, "index": d.index}
                for d in get_accelerator_detector().available_devices
            ]
            gui, filtered = map_device(devices, config.cuda_device)
            self.fp16 = use_reduced_precision(config.precision, config.model, self.info)
            self.settings = make_settings(config, workflow, gui, self.fp16)
            with tempfile.TemporaryDirectory(prefix="manga-settings-") as folder:
                settings_path = Path(folder) / "settings.json"
                settings_path.write_text(json.dumps(self.settings), encoding="utf-8")
                previous = sys.argv
                try:
                    sys.argv = ["run_upscale.py", "--settings", str(settings_path)]
                    spec = importlib.util.spec_from_file_location(
                        "mangajanai_upstream", source / "run_upscale.py"
                    )
                    assert spec and spec.loader
                    self.module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(self.module)
                finally:
                    sys.argv = previous
            self.module.spandrel_custom.install(ignore_duplicates=True)
            from .extra_arches import install

            install()
            self.options = self.module.SettingsParser(
                {
                    "use_cpu": False,
                    "use_fp16": self.fp16,
                    "accelerator_device_index": filtered,
                    "gpu_index": config.cuda_device,
                    "budget_limit": 0,
                }
            )
            self._context()
            from packages.chaiNNer_pytorch.settings import get_settings

            actual_device = get_settings(self.module.context).device
            if str(actual_device) != f"cuda:{config.cuda_device}":
                raise RuntimeError(f"Upstream chọn sai GPU: {actual_device}")
            if config.effective_detector == "corrected":
                self.module.cv_image_is_grayscale = lambda image, threshold: detect(
                    image, threshold, "corrected"
                )
        self.split_module = importlib.import_module("nodes.impl.pytorch.auto_split")
        self.original_split = self.split_module.auto_split
        self.split_module.auto_split = self._tracked_split
        self.log_handler = logging.StreamHandler(self.log)
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        self.backend_logger = logging.getLogger("sanic.root")
        self.previous_handlers = self.backend_logger.handlers[:]
        self.backend_logger.handlers = [self.log_handler]
        self.backend_logger.setLevel(logging.DEBUG)

    def _context(self) -> None:
        self.module.context = self.module._ExecutorNodeContext(
            self.module.ProgressController(), self.options, Path()
        )

    def _tracked_split(self, image, upscale, tiler, overlap=16):
        height, width = image.shape[:2]
        tw, th = tiler.starting_tile_size(width, height, image.shape[2] if image.ndim == 3 else 1)
        done = 0
        total = math.ceil(width / tw) * math.ceil(height / th)
        self.callback(0, total, "Đang Upscale bằng model")

        def tracked(tile, region):
            nonlocal done
            result = upscale(tile, region)
            if isinstance(result, self.split_module.Split):
                # Restart the image instead of upstream's buggy mid-row OOM restart.
                raise TileOOM(max(16, min(tw, th) // 2))
            done += 1
            self.callback(done, max(done, total), "Đang Upscale bằng model")
            return result

        return self.original_split(image, tracked, tiler, overlap)

    def process(
        self,
        source: Path,
        destination: Path,
        chain: dict,
        progress: Callable[[int, int, str], None],
    ) -> dict:
        import torch

        self.callback = progress
        model_path = str((self.config.models / chain["ModelFilePath"]).resolve())
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Thiếu model: {model_path}; chạy models download")
        hit = model_path in self.cache
        with contextlib.redirect_stdout(self.log), contextlib.redirect_stderr(self.log):
            self._context()
            for key, model in self.cache.items():
                if key != model_path:
                    model.to("cpu")
            if not hit:
                while len(self.cache) >= 2:
                    self.cache.popitem(last=False)
                torch.cuda.empty_cache()
            progress(0, 1, "Chuẩn bị ảnh và model AI")
            queue: Queue = Queue()
            chain = chain.copy()
            device = self.config.cuda_device
            free, total = torch.cuda.mem_get_info(device)
            # PyTorch can reuse its idle cache without asking the driver for more VRAM.
            reusable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
            free = min(total, free + max(0, reusable))
            tile = initial_tile_size(self.config.model, self.config.tile, free)
            learned = self.tile_limits.get(model_path)
            if learned:
                tile = min(tile, learned) if tile else learned
            chain["ModelTileSize"] = str(tile) if tile else "Auto (Estimate)"
            self.module.preprocess_worker_image(
                queue,
                str(source),
                str(destination),
                True,
                self.config.scale,
                0,
                0,
                [chain],
                self.cache,
                self.config.threshold,
            )
            image, _, _, gray, width, height, tile_size, model = queue.get_nowait()
            if image is None or model is None:
                raise RuntimeError(
                    "Backend không chọn/load được AI model; dừng, không tạo ảnh resize"
                )
            if not np.isfinite(image).all():
                raise ValueError(
                    "Upstream preprocessing sinh NaN/Inf (có thể ảnh một màu + auto levels)"
                )
            self.cache.move_to_end(model_path)
            retries_before = self.oom_retries
            for attempt in range(8):
                try:
                    result = self.module.ai_upscale_image(image, tile_size, model)
                    break
                except TileOOM as exc:
                    self.oom_retries += 1
                    if exc.tile < 32 or attempt == 7:
                        raise RuntimeError(
                            "CUDA OOM kể cả tile tối thiểu; chọn GPU nhiều VRAM hơn"
                        ) from exc
                    tile_size = self.module.get_tile_size(str(exc.tile))
                    self.tile_limits[model_path] = exc.tile
                    torch.cuda.empty_cache()
                    progress(
                        0, 1, f"Thiếu bộ nhớ card · đang thử cách dùng ít bộ nhớ hơn (lần {attempt + 1})"
                    )
            if not np.isfinite(result).all():
                raise ValueError("AI output chứa NaN/Inf; thử --precision fp32")
            if gray:
                result = self.module.convert_image_to_grayscale(result)
            progress(0, 1, "Hoàn thiện và lưu ảnh")
            self.module.save_image(
                result,
                str(destination),
                self.config.format,
                self.config.quality,
                self.config.lossless,
                width,
                height,
                self.config.scale,
                0,
                0,
                gray,
            )
            progress(1, 1, "Đã lưu tạm · đang kiểm tra")
        return {
            "model_cache_hit": hit,
            "oom_retries": self.oom_retries - retries_before,
            "tile": int(tile_size),
            "precision": {
                "torch.float32": "fp32", "torch.float16": "fp16", "torch.bfloat16": "bf16",
            }[str(model.dtype)],
        }

    def close(self) -> None:
        self.split_module.auto_split = self.original_split
        self.backend_logger.handlers = self.previous_handlers
        self.cache.clear()
        self.log.close()
