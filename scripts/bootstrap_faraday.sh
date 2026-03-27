#!/usr/bin/env bash
#
# Reference: requirements_server.txt
# Purpose: create a Python 3.8 environment suitable for Faraday/Linux runs.
#

set -euo pipefail

ENV_NAME="${ENV_NAME:-ecgdl}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found in PATH."
    exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" python=3.8 -y
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install -r requirements_server.txt

python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY
