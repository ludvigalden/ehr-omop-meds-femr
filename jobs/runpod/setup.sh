#!/bin/sh
# Environment setup for a rented-GPU mapping job.
#
# Installs uv, restores the locked environment with the embed and hf extras,
# swaps the lock's CPU-only torch for the CUDA build of the same version, and
# verifies that a CUDA GPU is actually usable before anything expensive
# starts. Fails closed on any missing prerequisite, naming the reason.
#
# OMOPMAP_TORCH_INDEX selects the PyTorch wheel index for the CUDA build
# [https://download.pytorch.org/whl/cu132]. The job scripts run with
# UV_NO_SYNC=1 so `uv run` keeps the CUDA build instead of re-syncing to the
# lock.
#
# HF_TOKEN, when a step needs it, is read from the environment only. It is
# never echoed, logged, or written to disk by this script.
set -eu

# Repo root: two levels above this script's directory.
repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_root"

# --- uv ---
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing a user-local copy" >&2
  installer=$(mktemp)
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://astral.sh/uv/install.sh -o "$installer"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$installer" https://astral.sh/uv/install.sh
  else
    echo "FAIL: neither curl nor wget is available to fetch the uv installer" >&2
    exit 1
  fi
  sh "$installer"
  rm -f "$installer"
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- locked environment with the GPU-path extras ---
uv sync --frozen --extra embed --extra hf

# --- CUDA torch: same version as the lock, from the PyTorch CUDA index ---
torch_index=${OMOPMAP_TORCH_INDEX:-https://download.pytorch.org/whl/cu132}
torch_version=$(uv run --no-sync python -c \
  "from importlib.metadata import version; print(version('torch').split('+')[0])")
uv pip install --reinstall-package torch --index "$torch_index" "torch==$torch_version"

# --- GPU gate: fail closed when there is no usable CUDA device ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FAIL: nvidia-smi not found; the job requires a GPU machine" >&2
  exit 1
fi
nvidia-smi -L

uv run --no-sync python - <<'EOF'
import sys

import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("FAIL: torch.cuda is not available; the job requires a GPU machine")
print("gpu:", torch.cuda.get_device_name(0))
EOF

echo "setup OK"
