#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python - <<'PY'
import sys
import torch
import torchvision
if not (3, 11) <= sys.version_info[:2] < (3, 13):
    raise SystemExit("chainner_ext 0.3.10 chưa hỗ trợ Python 3.13. Chọn Runtime → Change runtime type → Runtime Version 2026.07 (Python 3.12), GPU, rồi chạy lại. Không tự thay Python/PyTorch của runtime.")
if not torch.cuda.is_available():
    raise SystemExit("Chưa có CUDA GPU. Colab: Runtime → Change runtime type → GPU.")
print("Giữ nguyên runtime:", sys.version, torch.__version__, torchvision.__version__, torch.version.cuda)
PY
# An overlay keeps Colab's CUDA/PyTorch installation intact.
python -m venv --without-pip --system-site-packages .venv-colab
python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
lines=[]
for name in ('torch', 'torchvision', 'torchaudio'):
    try:
        lines.append(f'{name}=={version(name)}')
    except PackageNotFoundError:
        continue
Path('.venv-colab/torch-constraints.txt').write_text('\n'.join(lines)+'\n')
PY
.venv-colab/bin/python -m pip install --disable-pip-version-check -c .venv-colab/torch-constraints.txt -e '.[backend]'
.venv-colab/bin/python scripts/bootstrap_upstream.py
.venv-colab/bin/python -m manga_colab doctor
printf '%s\n' 'Ready: .venv-colab/bin/python -m manga_colab run --help'
