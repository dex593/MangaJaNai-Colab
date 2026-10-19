#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
.venv-colab/bin/python -m manga_colab doctor "$@"
