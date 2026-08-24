#!/usr/bin/env bash
# Run the spike with the environment variables that matter on Apple Silicon.
set -euo pipefail
cd "$(dirname "$0")"

# Fall back to CPU for any op MPS has not implemented, rather than raising.
# If the run only succeeds with this set, that is itself a finding — note
# which op fell back, because it will be slow.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

exec .venv/bin/python smoke.py "$@"
