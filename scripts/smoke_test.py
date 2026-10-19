"""Opt-in REAL CUDA smoke test. Downloads official weights; never uses a fake backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manga_colab.config import Config
from manga_colab.gpu import gpu_info
from manga_colab.pipeline import run
from manga_colab.model_selector import EXTRA_MODELS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(".runtime/smoke"))
    parser.add_argument("--scale", type=float, default=2)
    parser.add_argument("--model", choices=["auto", *EXTRA_MODELS], default="auto")
    parser.add_argument("--models", type=Path, help="Reuse an existing verified model cache")
    args = parser.parse_args()
    gpu_info(required=True)
    root = args.directory.resolve()
    source = root / ("input" if args.model == "auto" else f"input-{args.model}")
    source.mkdir(parents=True, exist_ok=True)
    gray = Image.new("L", (64, 80), 255)
    draw = ImageDraw.Draw(gray)
    draw.rectangle((5, 5, 58, 74), outline=20, width=2)
    draw.text((10, 15), "MANGA", fill=35)
    for y in range(35, 68, 4):
        for x in range(10, 56, 4):
            draw.ellipse((x, y, x + 1, y + 1), fill=90)
    gray.save(source / "manga.png")
    color = Image.new("RGB", (64, 64), (40, 100, 160))
    draw = ImageDraw.Draw(color)
    draw.ellipse((12, 8, 55, 52), fill=(220, 130, 40))
    draw.line((4, 60, 58, 3), fill=(250, 50, 100), width=3)
    if args.model != "eula-digimanga":
        color.save(source / "color.png")
    output_root = root if args.model == "auto" else root / args.model
    output = output_root / f"scale-{args.scale:g}"
    config = Config(
        source,
        output,
        args.models or root / "models",
        scale=args.scale,
        tile=64,
        overwrite=True,
        model=args.model,
    )
    status = run(config)
    if status:
        return status
    result = json.loads((output / ".manga-colab/report.json").read_text(encoding="utf-8"))
    assert result["summary"]["successful"] == (1 if args.model == "eula-digimanga" else 2)
    assert len(result["summary"]["models"]) == (2 if args.model == "auto" else 1)
    print(f"REAL CUDA smoke test passed: {args.model}, target {args.scale:g}x.")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
