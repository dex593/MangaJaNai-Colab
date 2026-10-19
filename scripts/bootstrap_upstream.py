from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manga_colab.config import ROOT, UPSTREAM_COMMIT, UPSTREAM_REPO

PATCHES = {
    "nodes/impl/pytorch/auto_split.py": [
        (
            'if "allocate" in str(e) or "CUDA" in str(e) or "out of memory" in str(e).lower():',
            'if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():',
        ),
        ("            safe_cuda_cache_empty()", "            safe_accelerator_cache_empty(device)"),
    ],
}


def main() -> None:
    checkout = ROOT / ".upstream/MangaJaNaiConverterGui"
    created = not checkout.exists()
    if created:
        checkout.parent.mkdir(exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", "--filter=blob:none", UPSTREAM_REPO, str(checkout)],
            check=True,
        )
    actual = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if actual.stdout.strip() != UPSTREAM_COMMIT:
        if (
            not created
            and subprocess.check_output(
                ["git", "-C", str(checkout), "status", "--porcelain"], text=True
            ).strip()
        ):
            raise RuntimeError(
                "Upstream có chỉnh sửa local; giữ lại và dùng checkout sạch để đổi pin"
            )
        subprocess.run(["git", "-C", str(checkout), "fetch", "origin", UPSTREAM_COMMIT], check=True)
    # A newly cloned --no-checkout tree has no files, even when HEAD already matches.
    source = checkout / "MangaJaNaiConverterGui/backend/src"
    if not (source / "run_upscale.py").exists() or actual.stdout.strip() != UPSTREAM_COMMIT:
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", UPSTREAM_COMMIT], check=True
        )
    for name, replacements in PATCHES.items():
        path = source / name
        text = path.read_text(encoding="utf-8")
        for before, after in replacements:
            if after in text:
                continue
            if text.count(before) != 1:
                raise RuntimeError(f"Patch không khớp source đã pin: {name}")
            text = text.replace(before, after)
        path.write_text(text, encoding="utf-8", newline="\n")
    print(f"Upstream ready: {UPSTREAM_COMMIT}; 2 documented error-handling fixes")


if __name__ == "__main__":
    main()
