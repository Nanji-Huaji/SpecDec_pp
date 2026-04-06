#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_ENV="$ROOT_DIR/.venv-vllm"
SITE_PACKAGES="$VLLM_ENV/lib/python3.10/site-packages"

uv venv --clear "$VLLM_ENV" --python 3.10

uv pip install --python "$VLLM_ENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

uv pip install --python "$VLLM_ENV/bin/python" \
  "vllm==0.5.4" \
  "datasets==4.8.4" \
  "transformers==4.44.2" \
  "huggingface-hub<1" \
  "sentencepiece" \
  "protobuf" \
  "numpy<2" \
  "setuptools" \
  "wheel"

mkdir -p "$SITE_PACKAGES/pyairports"
printf 'from .airports import AIRPORT_LIST\n' > "$SITE_PACKAGES/pyairports/__init__.py"
printf 'AIRPORT_LIST = []\n' > "$SITE_PACKAGES/pyairports/airports.py"

"$VLLM_ENV/bin/python" -c "import torch, vllm; from pyairports.airports import AIRPORT_LIST; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('vllm', vllm.__version__); print('pyairports entries', len(AIRPORT_LIST))"
