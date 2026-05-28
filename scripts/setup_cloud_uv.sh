#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_FOR_VENV="${PYTHON_FOR_VENV:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --upgrade --user uv
  export PATH="${HOME}/.local/bin:${PATH}"
fi

cd "${ROOT_DIR}"
uv venv --python "${PYTHON_FOR_VENV}" "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

uv pip install torch --index-url "${TORCH_INDEX_URL}"
uv pip install -r requirements-cloud.txt

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
