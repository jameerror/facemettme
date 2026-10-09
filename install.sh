#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mode="${1:-cpu}"
if [[ "$mode" != cpu && "$mode" != gpu ]]; then
  echo 'Usage: bash install.sh [cpu|gpu]' >&2
  exit 2
fi
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
if [[ "$mode" == gpu ]]; then
  .venv/bin/python -m pip uninstall -y onnxruntime onnxruntime-gpu
  .venv/bin/python -m pip install 'onnxruntime-gpu[cuda,cudnn]>=1.21,<2'
fi
.venv/bin/python setup_models.py
echo 'Ready. Use bash start.sh, or follow LINUX_CLOUD.md for cloud access.'
