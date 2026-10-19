#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
printf '%s\n' '@@MANGA_COLAB_STEP@@1/4 · Kiểm tra máy và card đồ họa'
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
printf '%s\n' '@@MANGA_COLAB_STEP@@2/4 · Chuẩn bị nơi cài chương trình'
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
printf '%s\n' '@@MANGA_COLAB_STEP@@3/4 · Tải và cài thành phần hỗ trợ (có thể mất vài phút)'
.venv-colab/bin/python -m pip install --disable-pip-version-check --progress-bar off -c .venv-colab/torch-constraints.txt -e '.[backend]'
printf '%s\n' '@@MANGA_COLAB_STEP@@4/4 · Chuẩn bị bộ xử lý ảnh và kiểm tra lần cuối'
.venv-colab/bin/python scripts/bootstrap_upstream.py
.venv-colab/bin/python -m manga_colab doctor
printf '%s\n' 'Ready: .venv-colab/bin/python -m manga_colab run --help'
