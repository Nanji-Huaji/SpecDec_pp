#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_PYTHON="${SPECDEC_MAIN_PYTHON:-$ROOT_DIR/.venv/bin/python}"
DATASET_PYTHON="${SPECDEC_DATASET_PYTHON:-$ROOT_DIR/.venv-vllm/bin/python}"

if [ ! -x "$MAIN_PYTHON" ]; then
  echo "Main Python not found: $MAIN_PYTHON" >&2
  exit 1
fi

if [ ! -x "$DATASET_PYTHON" ]; then
  echo "Dataset Python not found: $DATASET_PYTHON" >&2
  echo "Run scripts/setup_vllm_env.sh first or set SPECDEC_DATASET_PYTHON." >&2
  exit 1
fi

export SPECDEC_DATASET_PYTHON="$DATASET_PYTHON"
exec "$MAIN_PYTHON" "$ROOT_DIR/prepare_acc_head.py" "$@"
