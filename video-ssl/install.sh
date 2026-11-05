#!/usr/bin/env bash
set -euo pipefail
cd /workspace/video-ssl
export UV_CACHE_DIR=/workspace/.cache/uv
if [ ! -x /workspace/.venvs/ml-pentesting/bin/python ]; then
  uv venv /workspace/.venvs/ml-pentesting --python 3.12
fi
uv pip install --python /workspace/.venvs/ml-pentesting/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /workspace/.venvs/ml-pentesting/bin/python -r requirements.lock
