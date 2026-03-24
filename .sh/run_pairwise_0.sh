#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export SCRIPT_TAG="${SCRIPT_TAG:-run_pairwise_0}"
export GPU_ID="${GPU_ID:-1}"

exec bash ./pairwise_sweep_worker.sh
