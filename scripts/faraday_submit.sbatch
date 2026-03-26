#!/bin/bash
#
# Reference: scripts/run_remote.sh
#

#SBATCH --job-name=ecgdl
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/ecg_classification_DL}"
ENV_NAME="${ENV_NAME:-ecgdl}"

mkdir -p "${REPO_DIR}/logs"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        conda activate "${ENV_NAME}"
    else
        echo "Conda environment '${ENV_NAME}' not found; using the current shell environment."
    fi
fi

cd "${REPO_DIR}"

bash scripts/run_remote.sh "$@"
