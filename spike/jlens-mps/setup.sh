#!/usr/bin/env bash
# Set up the spike environment. Idempotent — safe to re-run.
#
# Uses uv if present (much faster), falls back to python -m venv + pip.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_MIN="3.10"   # jlens requires >=3.10

if command -v uv >/dev/null 2>&1; then
  echo "==> creating venv with uv"
  uv venv --python 3.11 .venv
  PIP=(uv pip install --python .venv/bin/python)
else
  echo "==> uv not found; using python3 -m venv"
  python3 -m venv .venv
  PIP=(.venv/bin/python -m pip install --upgrade)
  "${PIP[@]}" pip >/dev/null
  PIP=(.venv/bin/python -m pip install)
fi

echo "==> installing torch"
# On macOS the default PyPI wheel is the one with MPS support. Do NOT point
# this at the CPU-only index; that is a Linux-container workaround and it
# will silently cost you the GPU.
"${PIP[@]}" torch

echo "==> installing transformers, huggingface_hub, jlens"
"${PIP[@]}" "transformers>=5.5" huggingface_hub numpy
"${PIP[@]}" "git+https://github.com/anthropics/jacobian-lens.git"

echo "==> verifying"
.venv/bin/python - <<'PY'
import torch, transformers, jlens
print(f"  torch        {torch.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  mps available: {torch.backends.mps.is_available()}")
print(f"  mps built:     {torch.backends.mps.is_built()}")
print("  jlens imports OK")
PY

echo
echo "Done. Next:"
echo "  ./run.sh                    # the full spike on Qwen3-8B"
echo "  ./run.sh --model qwen3-1.7b # a fast first pass (~3.6 GB download)"
