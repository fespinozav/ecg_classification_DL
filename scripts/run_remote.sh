#!/usr/bin/env bash
#
# Reference: repository entrypoint `run.py`
#

set -euo pipefail

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/ecgdl_mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/ecgdl_cache}"
export ECGDL_USE_WANDB="${ECGDL_USE_WANDB:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

mkdir -p "${MPLCONFIGDIR}"
mkdir -p "${XDG_CACHE_HOME}"

python run.py "$@"
